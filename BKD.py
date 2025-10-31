import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np

# Giả định các hàm này đã được định nghĩa trong file của bạn
from helper.utils import load_teacher, accuracy, test, adjust_learning_rate
# Thay thế 'from cifar import build_dataset' bằng hàm dưới đây cho dữ liệu Long-tailed
# from cifar import build_dataset 

# ==============================================================================
# 1. Hàm tạo Dataset Long-tailed (Thay thế build_dataset trong cifar.py)
# ==============================================================================

def build_dataset_long_tail(dataset_name, batch_size, imb_factor):
    """
    Tạo dataset CIFAR-10/100 Long-tailed theo phân phối lũy thừa.
    
    Args:
        dataset_name (str): 'cifar10' hoặc 'cifar100'.
        batch_size (int): Kích thước batch.
        imb_factor (int): Tỷ lệ mất cân bằng (rho) = n_max / n_min.
        
    Returns:
        train_loader (DataLoader): DataLoader cho tập huấn luyện Long-tailed.
        test_loader (DataLoader): DataLoader cho tập kiểm tra (Balanced).
        num_classes (int): Số lượng lớp.
        cls_num_list (list): Danh sách số lượng mẫu trong mỗi lớp (dùng cho Loss BKD).
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    print(f"==> Đang tải dataset {dataset_name}...")
    
    # 1. Định nghĩa Transform và Tải Dataset Gốc (Ví dụ dùng CIFAR)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == 'cifar10':
        train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_dataset_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError("Dataset không được hỗ trợ hoặc bạn cần triển khai logic tải dữ liệu riêng.")
        
    # 2. Tạo phân phối Long-tailed (Exponential Imbalance)
    
    # Số lượng mẫu tối đa (của lớp nhiều nhất)
    # CIFAR có 50000/C mẫu mỗi lớp. 
    n_max = 50000 // num_classes
    
    # Tính số lượng mẫu n_k cho mỗi lớp theo công thức lũy thừa
    # n_k = n_max * (imb_factor)^(-k/(C-1)) 
    cls_num_list = []
    
    # Lấy list targets gốc
    train_targets = np.array(train_dataset_full.targets)
    
    indices = []
    for i in range(num_classes):
        # Tính n_k:
        if imb_factor > 1:
            n_k = int(n_max * (1.0 / imb_factor)**(i / (num_classes - 1)))
        else:
            n_k = n_max
            
        cls_num_list.append(n_k)
        
        # Lấy indices của n_k mẫu đầu tiên thuộc lớp i
        cls_indices = np.where(train_targets == i)[0]
        indices.extend(cls_indices[:n_k])
        
    # Tạo Long-tailed Dataset từ các indices đã chọn
    train_dataset_long_tail = Subset(train_dataset_full, indices)
    
    print(f"  Số mẫu mỗi lớp (n_k): {cls_num_list}")
    print(f"  Tổng số mẫu tập Train: {len(train_dataset_long_tail)} (imb_factor={imb_factor})")
    
    # 3. Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset_long_tail, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Trả về các giá trị theo yêu cầu của bạn (train, test, valid)
    # và cls_num_list để tính BKD Loss
    return train_loader, test_loader, num_classes, cls_num_list


# ==============================================================================
# 2. Định nghĩa Balanced Knowledge Distillation Loss (BKDLoss)
# ==============================================================================

class BKDLoss(nn.Module):
    """
    Hàm Loss cho Balanced Knowledge Distillation (BKD) theo công thức của paper (Eq. 8).
    L_BKD = L_CE + T^2 * D_KL(Q' || P)
    Trong đó, Q' là soft-target của Teacher được cân bằng lớp và chuẩn hóa.
    """
    def __init__(self, cls_num_list, beta=0.9999, temperature=2.0):
        super(BKDLoss, self).__init__()
        self.T = temperature
        self.beta = beta
        self.CE_loss = nn.CrossEntropyLoss()
        
        # Khởi tạo weights omega_i theo công thức Effective Number (Eq. 2)
        # omega_i = (1 - beta) / (1 - beta^n_i)
        
        num_classes = len(cls_num_list)
        weights = []
        for n_i in cls_num_list:
            # Tương đương với 1/E_n (E_n là Effective Number)
            effective_num = 1.0 - (self.beta ** n_i)
            weights.append((1.0 - self.beta) / effective_num)
            
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        # Sử dụng KLDivLoss vì nó cho $D_{KL}(P || Q) = \sum P \log (P/Q)$ (P: log-softmax, Q: softmax)
        # Tuy nhiên, ta cần $D_{KL}(Q' || P) = \sum Q' \log (Q'/P)$
        # Cần triển khai thủ công để đảm bảo thứ tự Q' và P
        # KLDivLoss(P_log, Q_target) = \sum Q_target * (\log Q_target - P_log)
        self.KD_loss = nn.KLDivLoss(reduction='sum') # Sẽ chia lại cho batch_size ở forward

    def forward(self, student_logits, teacher_logits, targets):
        
        # Đảm bảo weights ở cùng device
        weights = self.weights.to(student_logits.device)
        batch_size = student_logits.size(0)

        # 1. Instance-Balanced Classification Loss (L_CE)
        # Sử dụng CrossEntropyLoss tiêu chuẩn (mỗi mẫu có trọng số như nhau)
        L_CE = self.CE_loss(student_logits, targets)

        # 2. Class-Balanced Distillation Loss (L_KD^CB) - dựa trên Eq. 8 và ý tưởng chuẩn hóa
        
        # Tính soft-target cho Teacher và Student
        Q = torch.softmax(teacher_logits / self.T, dim=1) # Teacher Soft Target (Q_hat)
        P_log = torch.log_softmax(student_logits / self.T, dim=1) # Student Log-Softmax (log P)

        # Tạo Tensor trọng số omega cho từng mẫu trong batch
        # Omega chỉ phụ thuộc vào lớp (target)
        # omega_i cho mỗi lớp i: weights[i]
        omega_i_batch = weights[targets] # (B,)
        
        # Áp dụng trọng số lớp omega_i vào soft-target Q (Weighted Q: Q_CB = omega_i * Q)
        # Nhân element-wise Q với omega_i_batch mở rộng (B, C)
        Q_CB = Q * omega_i_batch.unsqueeze(1) # (B, C)
        
        # Chuẩn hóa (Normalize): Q' = Q_CB / sum(Q_CB) 
        # Chuẩn hóa trên mỗi hàng (mỗi mẫu)
        sum_Q_CB = Q_CB.sum(dim=1, keepdim=True) # (B, 1)
        Q_prime = Q_CB / (sum_Q_CB + 1e-8) # (B, C) - Q' là target distribution

        # Tính KL Divergence: L_KL = T^2 * D_KL(Q' || P) 
        # Trong PyTorch: KLDivLoss(log_P, Q_target) = \sum Q_target * (\log Q_target - log_P)
        # Q_target: Q'
        # log_P: P_log
        L_KD_CB = self.KD_loss(P_log, Q_prime) / batch_size
        
        # L_KD_CB phải nhân với T^2 để scale gradient
        L_KD_CB = L_KD_CB * (self.T ** 2)

        # 3. Tổng Loss: L_BKD = L_CE + L_KD^CB 
        total_loss = L_CE + L_KD_CB
        
        # (Optional: In ra giá trị loss để debug)
        # print(f"L_CE: {L_CE.item():.4f}, L_KD_CB: {L_KD_CB.item():.4f}")
        
        return total_loss

# ==============================================================================
# 3. Định nghĩa Arguments (Arguments)
# ==============================================================================
def parse_args():
    """Định nghĩa và parse các arguments từ command line."""
    parser = argparse.ArgumentParser(description='PyTorch Balanced Knowledge Distillation (BKD) Training')
    
    # Cấu hình chung
    parser.add_argument('--dataset', type=str, default='cifar100', help='Tên dataset: cifar10/cifar100/...')
    parser.add_argument('--model', type=str, default='resnet8x4', help='Tên kiến trúc student model')
    parser.add_argument('--teacher_model', type=str, default='resnet32', help='Tên kiến trúc teacher model (paper dùng model cùng size)')
    parser.add_argument('--teacher_ckpt_path', type=str, default='./checkpoints/teacher/resnet32_cifar100_ce_best.pth', help='Đường dẫn đến checkpoint của teacher (đã train bằng CE)')
    parser.add_argument('--gpu', type=int, default=0, help='Chỉ số GPU để sử dụng')
    
    # Tham số Long-tailed
    parser.add_argument('--imb_factor', type=int, default=100, help='Tỷ lệ mất cân bằng rho (n_max / n_min)')
    
    # Tham số BKD
    parser.add_argument('--temperature', type=float, default=2.0, help='Nhiệt độ T cho Softmax trong BKD (Paper dùng T=2.0)')
    parser.add_argument('--beta', type=float, default=0.9999, help='Tham số beta cho Effective Number (Paper dùng beta=0.9999)')
    
    # Cấu hình huấn luyện
    parser.add_argument('--epoch', type=int, default=200, help='Tổng số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=128, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate ban đầu (Paper dùng 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum cho SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='160,180', help='Các epoch để giảm LR (ví dụ: 160,180) - Paper dùng 160, 180 cho 200 epochs')
    
    args = parser.parse_args()
    
    # Chuyển chuỗi epoch giảm LR thành list số nguyên
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args

# ==============================================================================
# 4. Hàm Huấn Luyện (Train)
# ==============================================================================

def train_bkd(epoch, train_loader, student_model, teacher_model, criterion, optimizer, args):
    """Một epoch huấn luyện BKD"""
    student_model.train()
    teacher_model.eval() 
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        
        # Forward qua Student Model
        student_logits = student_model(inputs)
        
        # Forward qua Teacher Model (Không tính gradient)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        # Tính tổng Loss BKD
        # criterion là BKDLoss
        loss = criterion(student_logits, teacher_logits, targets)
        
        # Backward và tối ưu hóa
        loss.backward()
        optimizer.step()

        # Hiển thị tiến trình
        # if batch_idx % 100 == 0:
            # print(f'Epoch [{epoch}/{args.epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} finished. Time: {epoch_time:.2f}s")
    
# ==============================================================================
# 5. Hàm Main
# ==============================================================================

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    train_loader, test_loader, num_classes, cls_num_list = \
        build_dataset_long_tail(args.dataset, args.batch_size, args.imb_factor)
    
    teacher_model = load_teacher(args).to(device)
    
    from model.teachernet import resnet8x4 
    student_model = resnet8x4(num_classes=num_classes).to(device)

    # 4. Định nghĩa Loss, Optimizer, Scheduler
    # Khởi tạo BKD Loss với danh sách số lượng mẫu mỗi lớp
    criterion = BKDLoss(
        cls_num_list=cls_num_list, 
        beta=args.beta, 
        temperature=args.temperature
    ).to(device)
    
    optimizer = optim.SGD(student_model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    
    # 5. Huấn luyện
    best_acc = 0
    print("==> Bắt đầu huấn luyện Student với BKD Loss...")
    for epoch in range(1, args.epoch + 1):
        adjust_learning_rate(optimizer, epoch)
        
        train_bkd(epoch, train_loader, student_model, teacher_model, criterion, optimizer, args)
        
        val_acc = test(student_model, test_loader, epoch, args)
        
        # 7. Lưu checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'*** Lưu checkpoint tốt nhất với Acc: {best_acc:.2f}% (Epoch {epoch}) ***')
            state = {
                'model': student_model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
            }
            save_dir = './checkpoints/student_bkd'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state, os.path.join(save_dir, f'best_bkd_{args.model}_{args.dataset}_rho{args.imb_factor}.pth'))

    print(f"Accuracy tốt nhất: {best_acc:.2f}%")

if __name__ == '__main__':
    # For long-tailed CIFAR-10, we first train the student model with vanilla knowledge distillation 
    # before the 160th epoch, and then deploy our BKD
    main()