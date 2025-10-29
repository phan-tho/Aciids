import argparse
import os

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import numpy as np

import model.newresnet as newresnet

from helper.utils import load_teacher, accuracy, test

from cifar import build_dataset, build_dummy_dataset
parser = argparse.ArgumentParser(description='RW KD Training')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_epoch', default=[5, 80, 120, 175], nargs='+', type=int,
                    help='epochs to decay learning rate')
parser.add_argument('--temperature', default=4, type=float, help='temperature for softmax')

parser.add_argument('--teacher_ckpt', default='teacher_resnet32_cifar10.pth', type=str)
parser.add_argument('--l_meta', default='hard', help='mix/hard/soft')
parser.add_argument('--debug', default=False, type=bool, help='gen dummy dataset for debug')

parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor, larger means more imbalance')
parser.add_argument('--n_omits', default=0, type=int, help='number of classes to omit')

parser.add_argument('--prefetch', type=int, default=0)
parser.set_defaults(augment=True)
args = parser.parse_args()

# test: python MW_Net.py --debug True --dataset cifar10 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.seed = 12
args.n_classes = 10 if args.dataset == 'cifar10' else 100
torch.manual_seed(args.seed)

if not os.path.exists('log'):
    os.makedirs('log')

if not os.path.exists('log/args.txt'):
    with open('log/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f'{arg} = {getattr(args, arg)}\n')

def build_student():
    return newresnet.meta_resnet8x4(num_classes=args.n_classes).to(device)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_epoch:
        if epoch >= milestone:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def kd_loss_fn(student_logits, teacher_logits, target, args):
    hard_loss = F.cross_entropy(student_logits, target, reduction='none')

    log_student = F.log_softmax(student_logits / args.temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / args.temperature, dim=1)
    soft_loss = F.kl_div(log_student, soft_teacher, reduction='none').sum(1) * (args.temperature * args.temperature)

    # hard_loss shape: [batch_size]
    return hard_loss, soft_loss

def normalize_epsilon(epsilon):
    # Normalize epsilon: shape (batch_size, 2)
    for i in range(epsilon.size(0)):
        e = epsilon[i]
        sum = max(e[0], 1e-8) + max(e[1], 1e-8)
        e[0] = max(e[0], 1e-8) / sum
        e[1] = max(e[1], 1e-8) / sum
    return epsilon

def train(train_loader, valid_loader, model, teacher, optimizer_model, epoch, args):
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

    n_batches = 0

    ''' 
        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat,val_labels)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
        
        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps,min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()
    '''

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs_teacher = teacher(inputs)

        # Step 1: Meta-model (for meta-update) 
        meta_model = build_student()
        meta_model.load_state_dict(model.state_dict()).to(device)

        outputs_student = meta_model(inputs)

        epsilon = torch.zeros((inputs.size(0), 2), requires_grad=True, device=device)

        lce, lkl = kd_loss_fn(outputs_student, outputs_teacher, targets, args)
        inner_loss = torch.sum(epsilon[:, 0] * lce + epsilon[:, 1] * lkl) / inputs.size(0)

        meta_model.zero_grad()
        grads = torch.autograd.grad(inner_loss, (meta_model.params()), create_graph=True)
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

        epsilon = -torch.autograd.grad(l_g_meta, epsilon)[0]
        epsilon = normalize_epsilon(epsilon)

        outputs_new_s = model(inputs)
        lce, lkl = kd_loss_fn(outputs_new_s, outputs_teacher, targets, args)
        inner_loss = torch.sum(epsilon[:, 0] * lce + epsilon[:, 1] * lkl) / inputs.size(0)

        optimizer_model.zero_grad()
        inner_loss.backward()
        optimizer_model.step()
        

        prec_meta = accuracy(outputs_val_student.data, targets_val.data, topk=(1,))[0]
        prec_train = accuracy(outputs_new_s.data, targets.data, topk=(1,))[0]

        train_loss += inner_loss.item()
        meta_loss += l_g_meta.item()

        total_prec_train += prec_train.item()
        total_prec_meta += prec_meta.item()

        # if (batch_idx + 1) % args.print_freq == 0:
        #     print('Epoch: [%d/%d]\t'
        #           'Iters: [%d/%d]\t'
        #           'Loss: %.4f\t'
        #           'MetaLoss:%.4f\t'
        #           'Prec@1 %.2f\t'
        #           'Prec_meta@1 %.2f' % (
        #               (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size,
        #               train_loss / (batch_idx + 1), meta_loss / (batch_idx + 1), prec_train, prec_meta))
            
        n_batches = batch_idx + 1

    log = {'train': {'loss_train': float(train_loss / n_batches), 'acc_train': float(total_prec_train / n_batches), 'loss_meta': float(meta_loss / n_batches), 'acc_meta': float(total_prec_meta / n_batches)}}
    # save log to json file
    with open(args.name_file_log, 'r+') as f:
        data = json.load(f)
        data[str(epoch + 1)] = log
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

def main():
    if args.debug:
        print("Debug mode: using dummy dataset")
        train_loader, valid_loader, test_loader = build_dummy_dataset(args=args)
    else:
        train_loader, valid_loader, test_loader = build_dataset(args=args)
    model = build_student()
    teacher = load_teacher(args)

    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0
    at_e = 0
    for epoch in range(args.epochs):
        if epoch == args.lr_decay_epoch[0]:
            # save state dict, optimizer of current model to continual training
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
            }, f'ckpt_epoch{epoch}_before_lr_decay.pth')

        adjust_learning_rate(optimizer_model, epoch, args)
        train(train_loader, valid_loader, model, teacher, optimizer_model, epoch, args)
        test_acc = test(model, test_loader, epoch, args)

        if test_acc >= best_acc:
            best_acc = test_acc
            at_e = epoch
            ckpt = {
                'student': model.state_dict(),
                'acc@1': best_acc,
                'epoch': epoch + 1,
                'config': vars(args)
            }
            torch.save(ckpt, f'8x4_{args.dataset}.pth')
        
    print(f'Best acc: {best_acc} at epoch {at_e + 1}')

if __name__ == '__main__':
    main()
