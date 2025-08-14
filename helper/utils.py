import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import model.teachernet as teachernet


def test(model, test_loader, epoch, args):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    # save test loss and acc to json file
    log = {'loss_test': float(test_loss), 'acc_test': float(acc)}
    with open(args.name_file_log, 'r+') as f:
        data = json.load(f)
        data[str(epoch + 1)]['test'] = log
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

    return acc

def kd_loss_fn(student_logits, teacher_logits, target, args):
    if args.normalize_logits:
        # devide logits by standard deviation
        student_logits = student_logits / student_logits.std(dim=1, keepdim=True)
        teacher_logits = teacher_logits / teacher_logits.std(dim=1, keepdim=True)
    hard_loss = F.cross_entropy(student_logits, target, reduction='none')

    log_student = F.log_softmax(student_logits / args.temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / args.temperature, dim=1)
    soft_loss = F.kl_div(log_student, soft_teacher, reduction='none').sum(1) * (args.temperature * args.temperature)

    # adapt commented code above when args.use_wsl is True
    if args.use_wsl:
        fc_s_auto = student_logits.detach()
        fc_t_auto = teacher_logits.detach()
        log_softmax_s = F.log_softmax(fc_s_auto, dim=1)
        log_softmax_t = F.log_softmax(fc_t_auto, dim=1)

        one_hot_label = F.one_hot(target, num_classes=100).float()
        # one_hot_label shape (batch_size, num_classes)
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).to(student_logits.device)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        soft_loss = focal_weight.squeeze() * soft_loss

    return hard_loss, soft_loss


def load_teacher(args):
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    model = teachernet.resnet32x4(num_classes=10 if args.dataset == 'cifar10' else 100)
    try:
        ckt = torch.load(args.teacher_ckpt, map_location=args.device)
        model.load_state_dict(ckt['net'])
        print('load teacher acc', ckt['acc@1'])
    except Exception as e:
        print('Error loading teacher model:', e)
    model.to(args.device)
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

def adjust_learning_rate(optimizer, epoch, args, optimizer_vnet=None):
    # if model is not None. load state dict best currently model at epoch in lr_decay_epoch
    # lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))

    for e in args.lr_decay_epoch:
        if epoch == e:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    if optimizer_vnet is not None:
        if epoch == 200:
            for param_group in optimizer_vnet.param_groups:
                param_group['lr'] *= 0.1