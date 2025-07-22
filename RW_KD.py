import argparse
import os
import time

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import numpy as np

import model.teachernet as teachernet

from cifar import build_dataset, build_dummy_dataset
from helper.utils import test, kd_loss_fn, load_teacher, accuracy, adjust_learning_rate
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.datasets import MNIST

import torchopt
# from helpers.argument import parse_args
# from helpers.model import LeNet5
# from helpers.utils import get_imbalance_dataset, plot, set_seed

parser = argparse.ArgumentParser(description='ReWeight KD Training')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=200, type=int, help='epochs to run')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_epoch', default=[5, 80, 120, 175], nargs='+', type=int,
                    help='epochs to decay learning rate')
parser.add_argument('--reload_at_decay_epoch', default=False, type=bool)
parser.add_argument('--temperature', default=4, type=float, help='temperature for softmax')
parser.add_argument('--normalize_logits', default=False, type=bool, help='normalize logits by std')
parser.add_argument('--print_freq', default=100, type=int)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0)
parser.add_argument('--teacher_ckpt', default='teacher_resnet32_cifar10.pt', type=str)
parser.add_argument('--name_file_log', default='log/log_loss.json', type=str, help='file to save log')
parser.add_argument('--log_weight_path', default='log/log_weight.json', type=str, help='file to save log weight')
parser.add_argument('--log_weight_freq', default=10, type=int, help='log weight after n epochs')
parser.add_argument('--l_meta', default='hard', help='mix/only hard/only soft')
parser.set_defaults(augment=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

def main():
    # Set up logging
    if not os.path.exists('log'):
        os.makedirs('log')
    # if not os.path.exists('log/args.json'):
    #     with open('log/args.json', 'w') as f:
    #         json.dump(vars(args), f, indent=4)
    if not os.path.exists(args.name_file_log):
        with open(args.name_file_log, 'w') as f:
            json.dump({}, f, indent=4)
    if not os.path.exists(args.log_weight_path):
        with open(args.log_weight_path, 'w') as f:
            json.dump({}, f, indent=4)

    # Load dataset
    train_loader, valid_loader, test_loader = build_dataset(args=args)
    # train_loader, valid_loader, test_loader = build_dummy_dataset(args=args)

    model = teachernet.resnet8x4(num_classes=(100 if args.dataset == 'cifar100' else 10)).to(device)
    teacher = load_teacher(args)

    model_optimizer = torchopt.MetaSGD(model, lr=args.lr)
    real_model_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                           momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc = 0
    at_e = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(real_model_optimizer, epoch, args)
        train(train_loader, valid_loader, model, teacher, model_optimizer, real_model_optimizer, epoch)
        test_acc = test(model, test_loader, epoch, args)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            at_e = epoch

            ckpt = {
                'student': model.state_dict(),
                'acc@1': best_acc,
                'epoch': epoch + 1
            }
            torch.save(ckpt, f'8x4_{args.dataset}.pth')

    print(f"Best accuracy: {best_acc} at epoch {at_e}")


def normalize_epsilon(epsilon):
    # Normalize epsilon: shape (batch_size, 2)
    for i in range(epsilon.size(0)):
        e = epsilon[i]
        sum = max(e[0], 1e-8) + max(e[1], 1e-8)
        e[0] = max(e[0], 1e-8) / sum
        e[1] = max(e[1], 1e-8) / sum
    return epsilon

def train(train_loader, valid_loader, model, teacher, model_optimizer, real_model_optimizer, epoch):
    train_loss = 0
    meta_loss = 0
    valid = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

    model.train()
    for idx, (train_x, train_y) in enumerate(train_loader):
        try:
            valid_x, valid_y = next(valid)
        except BaseException:
            valid = iter(valid_loader)
            valid_x, valid_y = next(valid)
        train_x, train_y, valid_x, valid_y = (
            train_x.to(args.device),
            train_y.to(args.device),
            valid_x.to(args.device),
            valid_y.to(args.device),
        )

        # reset meta-parameter weights
        # model.reset_meta(size=train_x.size(0)).   epsilon = 0
        epsilon = torch.zeros((train_x.size(0), 2), requires_grad=True, device=device)

        net_state_dict = torchopt.extract_state_dict(model)
        optim_state_dict = torchopt.extract_state_dict(model_optimizer)

        for _ in range(1):
            outputs_student = model(train_x)
            with torch.no_grad():
                outputs_teacher = teacher(train_x)
            lce, lkl = kd_loss_fn(outputs_student, outputs_teacher, train_y, args.temperature, args.normalize_logits)
            inner_loss = torch.sum(epsilon[:, 0] * lce + epsilon[:, 1] * lkl)
            inner_loss /= train_x.size(0)
            model_optimizer.step(inner_loss)

        # calculate outer_loss, derive meta-gradient and normalize
        # loss meta
        outputs_student_val = model(valid_x)
        with torch.no_grad():
            outputs_teacher_val = teacher(valid_x)
        lce, lkl = kd_loss_fn(outputs_student_val, outputs_teacher_val, valid_y, args.temperature, args.normalize_logits)
        if args.l_meta == 'mix':
            outer_loss = torch.mean(lce + lkl)
        elif args.l_meta == 'hard':
            outer_loss = torch.mean(lce)
        elif args.l_meta == 'soft':
            outer_loss = torch.mean(lkl)
        epsilon = -torch.autograd.grad(outer_loss, epsilon)[0]

        # normalize epsiplon: shape (batch_size, 2)
        epsilon = normalize_epsilon(epsilon)

        # reset the model and model optimizer
        torchopt.recover_state_dict(model, net_state_dict)
        torchopt.recover_state_dict(model_optimizer, optim_state_dict)

        # reuse inner_adapt to conduct real update based on learned meta weights
        # inner_loss = model.inner_loss(train_x, train_y)
        for _ in range(1):
            outputs_student = model(train_x)
            lce, lkl = kd_loss_fn(outputs_student, outputs_teacher, train_y, args.temperature, args.normalize_logits)
            inner_loss = torch.sum(epsilon[:, 0] * lce + epsilon[:, 1] * lkl)
            inner_loss /= train_x.size(0)
            # inner_loss = inner_loss(model, teacher, train_x, train_y, epsilon)
            real_model_optimizer.zero_grad()
            inner_loss.backward()
            real_model_optimizer.step()

        
        # ---------Log----------

        prec_meta = accuracy(outputs_student_val.data, valid_y.data, topk=(1,))[0]
        prec_train = accuracy(outputs_student.data, train_y.data, topk=(1,))[0]

        train_loss += inner_loss.item()
        meta_loss += outer_loss.item()

        total_prec_train += prec_train.item()
        total_prec_meta += prec_meta.item()

        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, idx + 1, len(train_loader.dataset)/args.batch_size,
                      train_loss / (idx + 1), meta_loss / (idx + 1), prec_train, prec_meta))
    # end of epoch
    # save epsilon
    if epoch % args.log_weight_freq == 0:
        log = {str(epoch + 1): epsilon.cpu().numpy().tolist()}
        with open(args.log_weight_path, 'r+') as f:
            data = json.load(f)
            data.update(log)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

    log = {'train': {'loss_train': float(train_loss / (idx + 1)), 'acc_train': float(total_prec_train / (idx + 1)), 'loss_meta': float(meta_loss / (idx + 1)), 'acc_meta': float(total_prec_meta / (idx + 1))}}
    # save log to json file
    with open(args.name_file_log, 'r+') as f:
        data = json.load(f)
        data[str(epoch + 1)] = log
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

if __name__ == '__main__':
    main()