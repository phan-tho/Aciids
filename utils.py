import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import teachernet


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

def kd_loss_fn(student_logits, teacher_logits, target, T, normalize):
    """Returns hard_loss (CE with gt) and soft_loss (KL with teacher) per sample"""
    if normalize:
        # devide logits by stadnard deviation
        student_logits = student_logits / student_logits.std(dim=1, keepdim=True)
        teacher_logits = teacher_logits / teacher_logits.std(dim=1, keepdim=True)
    hard_loss = F.cross_entropy(student_logits, target, reduction='none')
    log_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(log_student, soft_teacher, reduction='none').sum(1) * (T*T)
    return hard_loss, soft_loss




def load_teacher(args):
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    model = teachernet.resnet32x4(num_classes=10 if args.dataset == 'cifar10' else 100)
    ckt = torch.load(args.teacher_ckpt, map_location=args.device)
    model.load_state_dict(ckt['net'])
    print('load teacher acc', ckt['acc@1'])
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

def adjust_learning_rate(optimizer, epoch, args):
    # if model is not None. load state dict best currently model at epoch in lr_decay_epoch
    # lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))

    for e in args.lr_decay_epoch:
        if epoch == e:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1