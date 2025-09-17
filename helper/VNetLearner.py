import json
import torch

from helper.utils import kd_loss_fn

class VNetLearner:
    def __init__(self, vnet, optimizer_vnet, args):
        self.vnet = vnet
        self.optimizer_vnet = optimizer_vnet
        self.args = args
        self.last_epoch = 0

    def __call__(self, outputs_student, outputs_teacher, targets, epoch, no_grad=False):
        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets, self.args)

        if self.args.input_vnet == 'loss':
            cost = torch.stack([hard_loss, soft_loss], dim=1) # shape [batch, 2]
            v_lambda = self.vnet(cost.data)
        elif self.args.input_vnet == 'logits_teacher':
            v_lambda = self.vnet(outputs_teacher.data)
        elif self.args.input_vnet == 'logit_st':
            if self.args.norm_bf_feed_vnet:
                # devide logits by standard deviation
                outputs_student = outputs_student / outputs_student.std(dim=1, keepdim=True)
                outputs_teacher = outputs_teacher / outputs_teacher.std(dim=1, keepdim=True)
            out_s = outputs_student[torch.arange(outputs_student.size(0)), targets]
            out_t = outputs_teacher[torch.arange(outputs_teacher.size(0)), targets]
            out = torch.stack([out_s, out_t], dim=1) # shape [batch, 2]
            v_lambda = self.vnet(out.data)
        elif self.args.input_vnet == 'loss_ce':
            ce_teacher = torch.functional.F.cross_entropy(outputs_teacher, targets, reduction='none') # shape [batch]
            ce = torch.stack([hard_loss, ce_teacher], dim=1) # shape [batch, 2]
            v_lambda = self.vnet(ce.data)
        elif self.args.input_vnet == 'ce_student':
            v_lambda = self.vnet(hard_loss.unsqueeze(1).data)
        elif self.args.input_vnet == 'ce_s+tgt_logit_t':
            out_t = outputs_teacher[torch.arange(outputs_teacher.size(0)), targets]
            out = torch.stack([hard_loss, out_t], dim=1) # shape [batch, 2]
            v_lambda = self.vnet(out.data)

        if no_grad:
            v_lambda = v_lambda.detach()

        if self.last_epoch != epoch:
            self.last_epoch = epoch
            if epoch % self.args.log_weight_freq == 0:
                if self.args.input_vnet == 'loss':
                    loss_weights = v_lambda.detach().cpu().numpy().tolist()
                    cost_weights = cost.detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': loss_weights, 'cost': cost_weights}}
                elif self.args.input_vnet == 'logits_teacher':
                    weights = v_lambda.detach().cpu().numpy().tolist()
                    pred_teacher = outputs_teacher.argmax(dim=1).detach().cpu().numpy().tolist()
                    # variance of teacher logits
                    variance_teacher = outputs_teacher.var(dim=1).detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': weights, 'pred_teacher': pred_teacher, 'variance_teacher': variance_teacher}}
                elif self.args.input_vnet == 'logit_st':
                    weights = v_lambda.detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': weights, 'out_student': out_s.detach().cpu().numpy().tolist(), 'out_teacher': out_t.detach().cpu().numpy().tolist()}}
                elif self.args.input_vnet == 'loss_ce':
                    weights = v_lambda.detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': weights, 'ce_teacher': ce_teacher.detach().cpu().numpy().tolist(), 'ce_student': hard_loss.detach().cpu().numpy().tolist()}}
                elif self.args.input_vnet == 'ce_student':
                    weights = v_lambda.detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': weights, 'ce_student': hard_loss.detach().cpu().numpy().tolist()}}
                elif self.args.input_vnet == 'ce_s+tgt_logit_t':
                    weights = v_lambda.detach().cpu().numpy().tolist()
                    log = {str(epoch + 1): {'v_lambda': weights, 'ce_student': hard_loss.detach().cpu().numpy().tolist(), 'tgt_logit_teacher': out_t.detach().cpu().numpy().tolist()}}
                with open(self.args.log_weight_path, 'r+') as f:
                    data = json.load(f)
                    data.update(log)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()

        return v_lambda, hard_loss, soft_loss