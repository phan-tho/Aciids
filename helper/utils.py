import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
from model import newresnet
import model.teachernet as teachernet

def train(train_loader, valid_loader, model, teacher, vnet_learner, optimizer_model, scheduler_vnet, epoch, args):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

    n_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        with torch.no_grad():
            outputs_teacher = teacher(inputs)

        # Step 1: Meta-model (for meta-update) 
        meta_model = newresnet.meta_resnet8x4(num_classes=args.n_classes).to(args.device)
        meta_model.load_state_dict(model.state_dict())
        meta_model.to(args.device)

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
        inputs_val, targets_val = inputs_val.to(args.device), targets_val.to(args.device)
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

def test(model, test_loader, epoch, args):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.args.device), targets.to(args.args.device)
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

    # hard_loss shape: [batch_size]
    return hard_loss, soft_loss


def load_teacher(args):
    # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    model = teachernet.resnet32x4(num_classes=10 if args.dataset == 'cifar10' else 100)
    try:
        ckt = torch.load(args.teacher_ckpt, map_location=args.args.device)
        model.load_state_dict(ckt['net'])
        print('load teacher acc', ckt['acc@1'])
    except Exception as e:
        print('Error loading teacher model:', e)
    model.to(args.args.device)
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
    if optimizer_vnet is not None and epoch == 80:
        from collections import defaultdict
        optimizer_vnet.state = defaultdict(dict)
    
    for e in args.lr_decay_epoch:
        if epoch == e:
            args.lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_vnet.param_groups:
                param_group['lr'] *= 0.8