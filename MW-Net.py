import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
import numpy as np

from resnet import StudentResNet, VNet, TeacherResNet32 # StudentResNet là ResNet8 (num_blocks=[1,1,1])
from cifar import CIFAR10, CIFAR100  
parser = argparse.ArgumentParser(description='Meta-Weight-Net KD Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=120, type=int, help='epochs to run')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0)
parser.add_argument('--teacher_ckpt', default='teacher_resnet32_cifar10.pt', type=str)
parser.set_defaults(augment=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if args.dataset == 'cifar10':
        train_data = CIFAR10(
            root='../data', train=True, valid=False, num_valid=args.num_valid, transform=train_transform, download=True, seed=args.seed)
        valid_data = CIFAR10(
            root='../data', train=True, valid=True, num_valid=args.num_valid, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_data = CIFAR100(
            root='../data', train=True, valid=False, num_valid=args.num_valid, transform=train_transform, download=True, seed=args.seed)
        valid_data = CIFAR100(
            root='../data', train=True, valid=True, num_valid=args.num_valid, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    return train_loader, valid_loader, test_loader

def build_student():
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = StudentResNet(num_classes=num_classes, num_blocks=[1, 1, 1]).to(device)
    return model

def load_teacher():
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    import teachernet as teachernet
    model = teachernet.resnet32x4(num_classes=10 if args.dataset == 'cifar10' else 100)
    ckt = torch.load(args.teacher_ckpt, map_location=device)
    model.load_state_dict(ckt['net'])
    model.to(device)    
    return model

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def kd_loss_fn(student_logits, teacher_logits, target, T=4):
    """Returns hard_loss (CE with gt) and soft_loss (KL with teacher) per sample"""
    hard_loss = F.cross_entropy(student_logits, target, reduction='none')
    log_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(log_student, soft_teacher, reduction='none').sum(1) * (T*T)
    return hard_loss, soft_loss

def train(train_loader, valid_loader, model, teacher, vnet, optimizer_model, optimizer_vnet, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

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
        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets)
        cost = torch.stack([hard_loss, soft_loss], dim=1) # shape [batch, 2]
        v_lambda = vnet(cost.data)
        w_hard = v_lambda[:, 0:1] # shape (batch_size, 1)
        w_soft = v_lambda[:, 1:2] # shape (batch_size, 1)
        l_f_meta = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / len(cost)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(lr_inner=args.lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            inputs_val, targets_val = next(valid_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        with torch.no_grad():
            outputs_teacher_val = teacher(inputs_val)
        outputs_val_student = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(outputs_val_student, targets_val)
        prec_meta = accuracy(outputs_val_student.data, targets_val.data, topk=(1,))[0]

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # Step 2: Main model update
        outputs_student = model(inputs)
        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets)
        cost = torch.stack([hard_loss, soft_loss], dim=1)
        with torch.no_grad():
            v_lambda = vnet(cost)
        w_hard = v_lambda[:, 0:1]
        w_soft = v_lambda[:, 1:2]
        loss = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / len(cost)
        prec_train = accuracy(outputs_student.data, targets.data, topk=(1,))[0]

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()
        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size,
                      (train_loss / (batch_idx + 1)), (meta_loss / (batch_idx + 1)), prec_train, prec_meta))

def main():
    train_loader, valid_loader, test_loader = build_dataset()
    model = build_student()
    teacher = load_teacher()
    vnet = VNet(2, 100, 2).to(device)  # input=2 (hard/soft loss), output=2 (weight cho mỗi loss)
    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)

    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader, valid_loader, model, teacher, vnet, optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
    print('best accuracy:', best_acc)

if __name__ == '__main__':
    main()
