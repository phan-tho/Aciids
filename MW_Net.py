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

from helper.utils import test, train, kd_loss_fn, load_teacher, accuracy, adjust_learning_rate
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
parser.add_argument('--input_vnet', default='loss', type=str, help='input to vnet (loss/logits_teacher/logit_st/loss_ce/ce_student/ce_s+tgt_logit_t/logit_st+ce_student)')
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

if not os.path.exists('log/args.txt'):
    with open('log/args.txt', 'w') as f:
        for arg in vars(args):
            f.write(f'{arg} = {getattr(args, arg)}\n')

def build_student():
    return newresnet.meta_resnet8x4(num_classes=args.n_classes).to(device)

def build_vnet():
    if args.input_vnet == 'logits_teacher':
        # vnet = VNet(args.n_classes, [200, 100], 2).to(device)  # input=100 (features), output=2 (weight cho mỗi loss)
        vnet = VNet(args.n_classes, args.hidden_vnet, 2)
    elif args.input_vnet == 'feature_teacher':
        vnet = VNet(512, args.hidden_vnet, 2)
    elif args.input_vnet == 'ce_student':
        vnet = VNet(1, args.hidden_vnet, 2)
    elif args.input_vnet == 'logit_st+ce_student':
        vnet = VNet(3, args.hidden_vnet, 2)
    else:
        vnet = VNet(2, args.hidden_vnet, 2)
        # if args.input_vnet == 'loss' or args.input_vnet == 'logit_st' or args.input_vnet == 'loss_ce':
        # vnet = VNet(2, [200, 100], 2).to(device)  # input=2 (hard/soft loss), output=2 (weight cho mỗi loss)
    return vnet.to(device)

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
    vnet = build_vnet()

    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
    scheduler_vnet = ReduceLROnPlateau(optimizer_vnet, factor=0.5, patience=50)

    vnet_learner = VNetLearner(vnet, optimizer_vnet, args)

    best_acc = 0
    at_e = 0
    for epoch in range(args.epochs):
        if epoch == args.lr_decay_epoch[0]:
            # save state dict, optimizer of current model to continual training
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_model_state_dict': optimizer_model.state_dict(),
                'vnet_state_dict': vnet.state_dict(),
                'optimizer_vnet_state_dict': optimizer_vnet.state_dict(),
            }, f'ckpt_epoch{epoch}_before_lr_decay.pth')

        adjust_learning_rate(optimizer_model, epoch, args, optimizer_vnet)
        train(train_loader, valid_loader, model, teacher, vnet_learner, optimizer_model, scheduler_vnet, epoch, args)
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
