import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Giả định: Model của bạn được import từ file resnet.py ---
# Hãy đảm bảo bạn có file resnet.py với 2 kiến trúc này
try:
    from resnet import resnet32x4, resnet8x4
except ImportError:
    print("LỖI: Không tìm thấy file resnet.py hoặc các model resnet32x4, resnet8x4.")
    print("Vui lòng tạo file resnet.py chứa các định nghĩa model.")
    exit()
# -----------------------------------------------------------


def get_img_num_per_cls(dataset, num_classes, imb_factor):
    """
    Tạo một danh sách số lượng mẫu cho mỗi lớp theo phân phối long-tailed
    (giảm theo hàm mũ).
    """
    if dataset == 'cifar10':
        img_max = 50000 / num_classes
    elif dataset == 'cifar100':
        img_max = 50000 / num_classes
    else:
        raise NotImplementedError("Chỉ hỗ trợ cifar10 và cifar100")

    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor**(-cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


class ImbalancedCIFAR(Dataset):
    """
    Tạo Dataset CIFAR mất cân bằng từ Dataset gốc.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 dataset_name='cifar10', imb_factor=1.0):

        if dataset_name == 'cifar10':
            self.num_classes = 10
            base_dataset = torchvision.datasets.CIFAR10(root=root, train=train,
                                                        download=download)
        elif dataset_name == 'cifar100':
            self.num_classes = 100
            base_dataset = torchvision.datasets.CIFAR100(root=root, train=train,
                                                         download=download)
        else:
            raise NotImplementedError

        self.transform = transform
        self.data = []
        self.targets = []

        # Lấy số lượng mẫu mỗi lớp
        self.cls_num_list = get_img_num_per_cls(dataset_name, self.num_classes, imb_factor)
        print(f"Phân phối lớp (mất cân bằng, imb_factor={imb_factor}):")
        print(self.cls_num_list)

        # Lấy chỉ số cho mỗi lớp
        targets_np = np.array(base_dataset.targets, dtype=np.int64)
        indices_per_class = [
            np.where(targets_np == i)[0] for i in range(self.num_classes)
        ]

        # Xáo trộn và cắt bớt
        for cls_idx in range(self.num_classes):
            np.random.shuffle(indices_per_class[cls_idx])
            selected_indices = indices_per_class[cls_idx][:self.cls_num_list[cls_idx]]
            self.data.extend(base_dataset.data[selected_indices])
            self.targets.extend([cls_idx] * len(selected_indices))

        # Chuyển đổi sang np.array để tương thích
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Chuyển đổi từ HWC (numpy) sang CHW (tensor)
        if self.transform:
            img = self.transform(img)

        return img, target

    def get_cls_num_list(self):
        return self.cls_num_list


@torch.no_grad()
def get_teacher_prior(model_teacher, val_loader, num_classes, device):
    """
    Tính toán prior của teacher p^t(y)[cite: 219, 220].
    Đây là kỳ vọng của p^t(y|x) trên một validation set *cân bằng*.
    """
    print("Đang tính toán prior của teacher p^t(y) trên val set cân bằng...")
    model_teacher.eval()
    all_preds = []

    for inputs, _ in tqdm(val_loader, desc="Teacher Prior"):
        inputs = inputs.to(device)
        outputs = model_teacher(inputs)
        all_preds.append(F.softmax(outputs, dim=1))

    # Tính trung bình softmax trên toàn bộ dataset [cite: 220]
    p_t_y = torch.cat(all_preds, dim=0).mean(dim=0)
    print("Tính toán p^t(y) hoàn tất.")
    return p_t_y.detach()


class CCKDLoss(nn.Module):
    """
    Triển khai Class-Conditional Knowledge Distillation Loss (CCKD).
    Loss này tối ưu KL(q || r), trong đó:
    - q: Phân phối target đã de-bias của teacher [cite: 175]
    - r: Phân phối dự đoán đã de-bias của student [cite: 183]

    Triển khai theo phương trình (8), sử dụng Cross-Entropy (tương đương KL
    khi bỏ qua hằng số H(q)) và nhân với T^2 để scale gradient.
    """
    def __init__(self, log_ptr_y, log_pte_y, log_pt_y, T=2.0):
        super(CCKDLoss, self).__init__()
        self.T = T
        # Các log-priors cần thiết cho phương trình (8) 
        self.log_ptr_y = log_ptr_y # p_tr^s(y) [cite: 175]
        self.log_pte_y = log_pte_y # p_te^s(y) [cite: 224]
        self.log_pt_y = log_pt_y   # p^t(y) [cite: 218]

    def forward(self, student_logits, teacher_logits):
        # 1. Tính target q(y|x) từ Teacher [cite: 175, 193]
        # q \propto p^t(y|x) / p^t(y) -> logit = z_t - log(p^t(y))
        q_logits = teacher_logits.detach() - self.log_pt_y
        q_dist_temp = F.softmax(q_logits / self.T, dim=1)

        # 2. Tính phân phối r(y|x) từ Student [cite: 183, 193]
        # r \propto p_te^s(y|x) * p_tr^s(y) / p_te^s(y)
        # Logit = z_s + log(p_tr^s(y)) - log(p_te^s(y))
        r_logits = student_logits + self.log_ptr_y - self.log_pte_y
        r_log_dist_temp = F.log_softmax(r_logits / self.T, dim=1)

        # 3. Tính loss (Cross-Entropy H(q, r)) [cite: 193, 505, 506]
        # L_cckd = - sum(q_i * log(r_i))
        # Nhân T^2 để scale gradient (theo chuẩn Hinton KD)
        loss = - (q_dist_temp * r_log_dist_temp).sum(dim=1).mean() * (self.T * self.T)
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description='Triển khai CCKD (Paper)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset (cifar10 hoặc cifar100)')
    parser.add_argument('--imb_factor', type=float, default=0.1,
                        help='Tỷ lệ mất cân bằng (vd: 0.01 -> 100)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='Số epochs')
    parser.add_argument('--lr_decay_epoch', type=str, default='160,180',
                        help='Các epoch giảm lr (vd: 160,180)')
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Đường dẫn đến checkpoint của teacher (resnet32x4)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--T', type=float, default=2.0,
                        help='Nhiệt độ T cho distillation')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Hệ số cân bằng cho CCKD loss')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    """Giảm learning rate tại các mốc epoch"""
    lr = args.lr
    decay_epochs = [int(e) for e in args.lr_decay_epoch.split(',')]
    for e in decay_epochs:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(model_student, model_teacher, train_loader, optimizer,
                    criterion_sup, criterion_kd, alpha, device):
    model_student.train()
    model_teacher.eval() # Teacher luôn ở chế độ eval

    total_loss = 0
    total_loss_sup = 0
    total_loss_kd = 0

    pbar = tqdm(train_loader, desc="Training Epoch")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        student_logits = model_student(inputs)
        with torch.no_grad():
            teacher_logits = model_teacher(inputs)

        # Tính loss
        # 1. Supervised Loss (Logit Adjustment) 
        loss_sup = criterion_sup(student_logits, targets)

        # 2. CCKD Loss 
        loss_kd = criterion_kd(student_logits, teacher_logits)

        # Total loss [cite: 267]
        loss = loss_sup + alpha * loss_kd

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss_sup += loss_sup.item()
        total_loss_kd += loss_kd.item()

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         sup=f"{loss_sup.item():.3f}",
                         kd=f"{loss_kd.item():.3f}")

    avg_loss = total_loss / len(train_loader)
    avg_loss_sup = total_loss_sup / len(train_loader)
    avg_loss_kd = total_loss_kd / len(train_loader)
    return avg_loss, avg_loss_sup, avg_loss_kd


@torch.no_grad()
def test(model_student, test_loader, log_ptr_y, device):
    """
    Đánh giá model trên test set CÂN BẰNG.
    Sử dụng logic của Logit Adjustment (LA) để đánh giá[cite: 483]:
    Logit dự đoán = logit thô - log(p_tr)
    (Vì test prior p_te là uniform, log(p_te) là hằng số, có thể bỏ qua)
    """
    model_student.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_student(inputs)

        # Điều chỉnh logit để đánh giá trên test set cân bằng
        # [cite: 483] (tham chiếu cho Logit Adjustment)
        # Tương đương với: z - log(p_tr) + log(p_te)
        # log(p_te) là hằng số nên không ảnh hưởng argmax
        eval_logits = outputs - log_ptr_y

        _, predicted = torch.max(eval_logits.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def main():
    args = parse_args()
    print(args)

    # Setup seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Chuẩn bị Data
    num_classes = 100 if args.dataset == 'cifar100' else 10
    
    transform_train = transforms.Compose([
        transforms.ToPILImage(), # Vì data đang là numpy HWC
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), # Dữ liệu test gốc đã là PIL
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dataset train mất cân bằng
    train_dataset = ImbalancedCIFAR(
        root='./data', train=True, download=True,
        transform=transform_train,
        dataset_name=args.dataset,
        imb_factor=(args.imb_factor) # imb_factor = N_min / N_max
    )
    
    # Dataset test/val CÂN BẰNG
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform_test)
    else:
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                     download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=False, num_workers=4)

    # 2. Tính toán các Priors
    cls_num_list = train_dataset.get_cls_num_list()
    cls_num_tensor = torch.tensor(cls_num_list).float().to(device)

    # p_tr^s(y): Phân phối của training set [cite: 175]
    p_tr_y = cls_num_tensor / cls_num_tensor.sum()
    log_ptr_y = torch.log(p_tr_y).to(device)

    # p_te^s(y): Phân phối của test set (cân bằng) [cite: 224]
    p_te_y = torch.ones(num_classes).float().to(device) / num_classes
    log_pte_y = torch.log(p_te_y).to(device)

    # 3. Load Models
    model_student = resnet8x4(num_classes=num_classes).to(device)
    model_teacher = resnet32x4(num_classes=num_classes).to(device)

    if not os.path.exists(args.teacher_ckpt_path):
        print(f"LỖI: Không tìm thấy checkpoint của teacher tại: {args.teacher_ckpt_path}")
        return

    print(f"Đang load checkpoint của teacher từ: {args.teacher_ckpt_path}")
    model_teacher.load_state_dict(torch.load(args.teacher_ckpt_path,
                                             map_location=device))
    model_teacher.eval()

    # 4. Tính p^t(y) (Teacher Prior) [cite: 219]
    # (Sử dụng test_loader CÂN BẰNG làm validation set)
    p_t_y = get_teacher_prior(model_teacher, test_loader, num_classes, device)
    log_pt_y = torch.log(p_t_y).to(device)

    # 5. Setup Loss và Optimizer
    
    # Supervised Loss (Logit Adjustment) 
    # L_sup = CE(z_s + log(p_tr))
    class LogitAdjustmentLoss(nn.Module):
        def __init__(self, log_ptr_y):
            super(LogitAdjustmentLoss, self).__init__()
            self.log_ptr_y = log_ptr_y
        
        def forward(self, logits, target):
            adjusted_logits = logits + self.log_ptr_y
            return F.cross_entropy(adjusted_logits, target)

    criterion_sup = LogitAdjustmentLoss(log_ptr_y).to(device)

    # CCKD Loss 
    criterion_kd = CCKDLoss(log_ptr_y, log_pte_y, log_pt_y, T=args.T).to(device)
    
    # Optimizer cho Student
    optimizer = optim.SGD(model_student.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    # 6. Training
    print("Bắt đầu training...")
    best_acc = 0.0

    for epoch in range(1, args.epoch + 1):
        adjust_learning_rate(optimizer, epoch, args)
        
        loss, sup_loss, kd_loss = train_one_epoch(
            model_student, model_teacher, train_loader, optimizer,
            criterion_sup, criterion_kd, args.alpha, device
        )
        
        acc = test(model_student, test_loader, log_ptr_y, device)
        
        print(f"Epoch: {epoch}/{args.epoch} | "
              f"Loss: {loss:.4f} (Sup: {sup_loss:.4f} + KD: {kd_loss:.4f}) | "
              f"Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            # (Bạn có thể thêm logic lưu checkpoint ở đây)

    print(f"Training hoàn tất. Best Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()