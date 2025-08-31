import argparse
import os

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import numpy as np

from model.resnet import VNet
import model.teachernet as teachernet
import model.newresnet as newresnet

from helper.utils import test, kd_loss_fn, load_teacher, accuracy, adjust_learning_rate
from helper.VNetLearner import VNetLearner

from cifar import build_dataset, build_dummy_dataset
parser = argparse.ArgumentParser(description='Meta-Weight-Net KD Training')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_epoch', default=[5, 80, 120, 175], nargs='+', type=int,
                    help='epochs to decay learning rate')
parser.add_argument('--temperature', default=4, type=float, help='temperature for softmax')
parser.add_argument('--normalize_logits', default=False, type=bool, help='normalize logits by std')
parser.add_argument('--print_freq', default=150, type=int)
parser.add_argument('--prefetch', type=int, default=0)
parser.add_argument('--teacher_ckpt', default='teacher_resnet32_cifar10.pth', type=str)
parser.add_argument('--name_file_log', default='log/log_loss.json', type=str, help='file to save log')
parser.add_argument('--log_weight_path', default='log/log_weight.json', type=str, help='file to save log weight')
parser.add_argument('--log_weight_freq', default=10, type=int, help='log weight after n epochs')
parser.add_argument('--l_meta', default='hard', help='mix/hard/soft')
parser.add_argument('--input_vnet', default='loss', type=str, help='input to vnet (loss/logits_teacher/logit_st/loss_ce)')
parser.add_argument('--norm_bf_feed_vnet', default=False, type=bool, help='normalize before feeding to vnet')
parser.add_argument('--debug', default=False, type=bool, help='gen dummy dataset for debug')

parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor, larger means more imbalance')
parser.add_argument('--n_omits', default=0, type=int, help='number of classes to omit')

parser.add_argument('--scheduler_vnet', default=False, type=bool, help='use scheduler for vnet optimizer')
parser.add_argument('--hidden_vnet', default=[100], nargs='+', type=int,
                    help='hidden layers for vnet')
parser.set_defaults(augment=True)
args = parser.parse_args()

# test: python MW_Net.py --debug True --dataset cifar10 

"""Save args to a json file"""
if not os.path.exists('log'):
    os.makedirs('log')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.seed = 12
args.n_classes = 10 if args.dataset == 'cifar10' else 100
torch.manual_seed(args.seed)


def build_student():
    return newresnet.meta_resnet8x4(num_classes=args.n_classes).to(device)


def train(train_loader, valid_loader, model, teacher, vnet_learner, optimizer_model, scheduler_vnet, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

    n_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs_teacher = teacher(inputs)

        # Step 1: Meta-model (for meta-update) 
        meta_model = build_student()
        meta_model.load_state_dict(model.state_dict())
        meta_model.to(device)

        outputs_student = meta_model(inputs)

        v_lambda, hard_loss, soft_loss = vnet_learner(outputs_student, outputs_teacher, targets, epoch)

        w_hard = v_lambda[:, 0:1] # shape (batch_size, 1)
        w_soft = v_lambda[:, 1:2] # shape (batch_size, 1)
        l_f_meta = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / w_hard.size(0)

        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(lr_inner=args.lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(valid_loader_iter)
            # shape inputs_val: (batch_size, 3, 32, 32), targets_val: (batch_size)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            inputs_val, targets_val = next(valid_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        with torch.no_grad():
            outputs_teacher_val = teacher(inputs_val)
        outputs_val_student = meta_model(inputs_val)
        
        hard_loss, soft_loss = kd_loss_fn(outputs_val_student, outputs_teacher_val, targets_val, args)
        if args.l_meta == 'mix':
            l_g_meta = torch.mean(hard_loss + soft_loss)  # l_g_meta = hard_loss + soft_loss
        elif args.l_meta == 'hard':
            l_g_meta = torch.mean(hard_loss)  # l_g_meta = hard_loss
        else: # args.l_meta == 'soft'
            l_g_meta = torch.mean(soft_loss)  # l_g_meta = soft_loss

        vnet_learner.optimizer_vnet.zero_grad()
        l_g_meta.backward()
        vnet_learner.optimizer_vnet.step()

        # Step 2: Main model update
        outputs_student = model(inputs)
        v_lambda, hard_loss, soft_loss = vnet_learner(outputs_student, outputs_teacher, targets, epoch, no_grad=True)           

        w_hard = v_lambda[:, 0:1]
        w_soft = v_lambda[:, 1:2]
        loss = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / w_hard.size(0)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


        prec_meta = accuracy(outputs_val_student.data, targets_val.data, topk=(1,))[0]
        prec_train = accuracy(outputs_student.data, targets.data, topk=(1,))[0]

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

        total_prec_train += prec_train.item()
        total_prec_meta += prec_meta.item()

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size,
                      train_loss / (batch_idx + 1), meta_loss / (batch_idx + 1), prec_train, prec_meta))
            
        n_batches = batch_idx + 1

    if args.scheduler_vnet:
        scheduler_vnet.step(meta_loss / n_batches)

    log = {'train': {'loss_train': float(train_loss / n_batches), 'acc_train': float(total_prec_train / n_batches), 'loss_meta': float(meta_loss / n_batches), 'acc_meta': float(total_prec_meta / n_batches)}}
    # save log to json file
    with open(args.name_file_log, 'r+') as f:
        data = json.load(f)
        data[str(epoch + 1)] = log
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

def main():
    # Set up logging
    if not os.path.exists(args.name_file_log):
        with open(args.name_file_log, 'w') as f:
            json.dump({}, f, indent=4)

    if not os.path.exists(args.log_weight_path):
        with open(args.log_weight_path, 'w') as f:
            json.dump({}, f, indent=4)

    
    if args.debug:
        print("Debug mode: using dummy dataset")
        train_loader, valid_loader, test_loader = build_dummy_dataset(args=args)
    else:
        train_loader, valid_loader, test_loader = build_dataset(args=args)
    model = build_student()
    teacher = load_teacher(args)

    if args.input_vnet == 'loss' or args.input_vnet == 'logit_st' or args.input_vnet == 'loss_ce':
        # vnet = VNet(2, [200, 100], 2).to(device)  # input=2 (hard/soft loss), output=2 (weight cho mỗi loss)
        vnet = VNet(2, args.hidden_vnet, 2).to(device)
    elif args.input_vnet == 'logits_teacher':
        # vnet = VNet(args.n_classes, [200, 100], 2).to(device)  # input=100 (features), output=2 (weight cho mỗi loss)
        vnet = VNet(args.n_classes, args.hidden_vnet, 2).to(device)
    elif args.input_vnet == 'feature_teacher':
        vnet = VNet(512, args.hidden_vnet, 2).to(device)

    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
    scheduler_vnet = ReduceLROnPlateau(optimizer_vnet, factor=0.5, patience=50)

    vnet_learner = VNetLearner(vnet, optimizer_vnet, args)

    best_acc = 0
    at_e = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch, args, optimizer_vnet)
        train(train_loader, valid_loader, model, teacher, vnet_learner, optimizer_model, scheduler_vnet, epoch)
        test_acc = test(model, test_loader, epoch, args)

        if test_acc >= best_acc:
            best_acc = test_acc
            at_e = epoch
            ckpt = {
                'student': model.state_dict(),
                'vnet': vnet.state_dict(),
                'acc@1': best_acc,
                'epoch': epoch + 1,
                'config': vars(args)
            }
            torch.save(ckpt, f'8x4_{args.dataset}.pth')
        
    print(f'Best acc: {best_acc} at epoch {at_e + 1}')

if __name__ == '__main__':
    main()
