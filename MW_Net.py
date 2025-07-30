import argparse
import os

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import numpy as np

from model.resnet import VNet
import model.teachernet as teachernet
import model.newresnet as newresnet

from helper.utils import test, kd_loss_fn, load_teacher, accuracy, adjust_learning_rate

from cifar import build_dataset, build_dummy_dataset
parser = argparse.ArgumentParser(description='Meta-Weight-Net KD Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=200, type=int, help='epochs to run')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_epoch', default=[5, 80, 120, 175], nargs='+', type=int,
                    help='epochs to decay learning rate')
parser.add_argument('--temperature', default=4, type=float, help='temperature for softmax')
parser.add_argument('--normalize_logits', default=False, type=bool, help='normalize logits by std')
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--prefetch', type=int, default=0)
parser.add_argument('--teacher_ckpt', default='teacher_resnet32_cifar10.pth', type=str)
parser.add_argument('--name_file_log', default='log/log_loss.json', type=str, help='file to save log')
parser.add_argument('--log_weight_path', default='log/log_weight.json', type=str, help='file to save log weight')
parser.add_argument('--log_weight_freq', default=10, type=int, help='log weight after n epochs')
parser.add_argument('--l_meta', default='mix', help='mix/only hard/only soft')
parser.add_argument('--input_vnet', default='loss', type=str, help='input to vnet (loss/logits_teacher/logit_st)')
parser.add_argument('--debug', default=False, type=bool, help='gen dummy dataset for debug')
parser.set_defaults(augment=True)
args = parser.parse_args()

"""Save args to a json file"""
# check if log directory exists, if not create it
if not os.path.exists('log'):
    os.makedirs('log')
# if not os.path.exists('log/args.json'):
#     with open('log/args.json', 'w') as f:
#         json.dump(vars(args), f, indent=4)
# else:
#     with open('log/args.json', 'r+') as f:
#         data = json.load(f)
#         data.update(vars(args))
#         f.seek(0)
#         json.dump(data, f, indent=4)
#         f.truncate()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.seed = 12
args.n_classes = 10 if args.dataset == 'cifar10' else 100
torch.manual_seed(args.seed)


def build_student():
    return newresnet.meta_resnet8x4(num_classes=args.n_classes).to(device)


def train(train_loader, valid_loader, model, teacher, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

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

        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets, args.temperature, args.normalize_logits)
        
        if args.input_vnet == 'loss':
            cost = torch.stack([hard_loss, soft_loss], dim=1) # shape [batch, 2]
            v_lambda = vnet(cost.data)
        elif args.input_vnet == 'logits_teacher':
            v_lambda = vnet(outputs_teacher.data)
        elif args.input_vnet == 'logit_st':
            out_s = outputs_student[torch.arange(outputs_student.size(0)), targets]
            out_t = outputs_teacher[torch.arange(outputs_teacher.size(0)), targets]
            out = torch.stack([out_s, out_t], dim=1) # shape [batch, 2]
            v_lambda = vnet(out.data)

        # v_lambda = vnet(cost.data)
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
        
        hard_loss, soft_loss = kd_loss_fn(outputs_val_student, outputs_teacher_val, targets_val, args.temperature, args.normalize_logits)
        if args.l_meta == 'mix':
            l_g_meta = torch.mean(hard_loss + soft_loss)  # l_g_meta = hard_loss + soft_loss
        elif args.l_meta == 'hard':
            l_g_meta = torch.mean(hard_loss)  # l_g_meta = hard_loss
        else: # args.l_meta == 'soft'
            l_g_meta = torch.mean(soft_loss)  # l_g_meta = soft_loss

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # Step 2: Main model update
        outputs_student = model(inputs)
        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets, args.temperature, args.normalize_logits)
        cost = torch.stack([hard_loss, soft_loss], dim=1)

        out_s = outputs_student[torch.arange(outputs_student.size(0)), targets]
        out_t = outputs_teacher[torch.arange(outputs_teacher.size(0)), targets]
        with torch.no_grad():
            # v_lambda = vnet(cost)
            if args.input_vnet == 'loss':
                # hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets, args.temperature, args.normalize_logits)
                # cost = torch.stack([hard_loss, soft_loss], dim=1) # shape [batch, 2]
                v_lambda = vnet(cost)
            elif args.input_vnet == 'logits_teacher':
                v_lambda = vnet(outputs_teacher)
            elif args.input_vnet == 'logit_st':
                out = torch.stack([out_s, out_t], dim=1) # shape [batch, 2]
                v_lambda = vnet(out)            

        w_hard = v_lambda[:, 0:1]
        w_soft = v_lambda[:, 1:2]
        loss = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / len(cost)

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
            
    # end of epoch
    if epoch % args.log_weight_freq == 0:
        if args.input_vnet == 'loss':
            loss_weights = v_lambda.cpu().numpy().tolist()
            cost_weights = cost.detach().cpu().numpy().tolist()
            log = {str(epoch + 1): {'v_lambda': loss_weights, 'cost': cost_weights}}
        elif args.input_vnet == 'logits_teacher':
            weights = v_lambda.cpu().numpy().tolist()
            pred_teacher = outputs_teacher.argmax(dim=1).detach().cpu().numpy().tolist()
            # variance of teacher logits
            variance_teacher = outputs_teacher.var(dim=1).detach().cpu().numpy().tolist()
            log = {str(epoch + 1): {'v_lambda': weights, 'pred_teacher': pred_teacher, 'variance_teacher': variance_teacher}}
        elif args.input_vnet == 'logit_st':
            weights = v_lambda.cpu().numpy().tolist()
            log = {str(epoch + 1): {'v_lambda': weights, 'out_student': out_s.detach().cpu().numpy().tolist(), 'out_teacher': out_t.detach().cpu().numpy().tolist()}}

        with open(args.log_weight_path, 'r+') as f:
            data = json.load(f)
            data.update(log)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    log = {'train': {'loss_train': float(train_loss / (batch_idx + 1)), 'acc_train': float(total_prec_train / (batch_idx + 1)), 'loss_meta': float(meta_loss / (batch_idx + 1)), 'acc_meta': float(total_prec_meta / (batch_idx + 1))}}
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
        train_loader, valid_loader, test_loader = build_dummy_dataset(args=args)
    else:
        train_loader, valid_loader, test_loader = build_dataset(args=args)
    model = build_student()
    teacher = load_teacher(args)

    if args.input_vnet == 'loss' or args.input_vnet == 'logit_st':
        vnet = VNet(2, 100, 2).to(device)  # input=2 (hard/soft loss), output=2 (weight cho mỗi loss)
    elif args.input_vnet == 'logits_teacher':
        vnet = VNet(args.n_classes, [200, 100], 2).to(device)  # input=100 (features), output=2 (weight cho mỗi loss)
    elif args.input_vnet == 'feature_teacher':
        vnet = VNet(512, 200, 2).to(device)

    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)

    best_acc = 0
    at_e = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch, args, optimizer_vnet)
        train(train_loader, valid_loader, model, teacher, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model, test_loader, epoch, args)

        if test_acc >= best_acc:
            best_acc = test_acc
            at_e = epoch
            ckpt = {
                'student': model.state_dict(),
                'vnet': vnet.state_dict(),
                'acc@1': best_acc,
                'epoch': epoch + 1
            }
            torch.save(ckpt, f'8x4_{args.dataset}.pth')
        
    print(f'Best acc: {best_acc} at epoch {at_e + 1}')

if __name__ == '__main__':
    main()
