from PIL import Image
import os
import numpy as np
import sys
import pickle
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Subset


class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root='', train=True, valid=False, num_valid=1000,
                 transform=None, target_transform=None, download=False, seed=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train      # True = train set, False = test set
        self.valid = valid      # True = valid set, False = train set
        self.num_valid = num_valid
        self.num_classes = 10
        np.random.seed(seed)
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                file = os.path.join(root, self.base_folder, fentry[0])
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] >= 3 else pickle.load(fo)
                self.data.append(entry['data'])
                self.labels += entry['labels']
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1)) # HWC
            labels_np = np.array(self.labels)

            # Split balanced valid/train set
            valid_per_class = self.num_valid // self.num_classes
            valid_indices = []
            train_indices = []
            for cls in range(self.num_classes):
                cls_indices = np.where(labels_np == cls)[0]
                np.random.shuffle(cls_indices)
                valid_cls_idx = cls_indices[:valid_per_class]
                train_cls_idx = cls_indices[valid_per_class:]
                valid_indices.extend(valid_cls_idx)
                train_indices.extend(train_cls_idx)
            np.random.shuffle(valid_indices)
            np.random.shuffle(train_indices)
            if self.valid:
                self.data = self.data[valid_indices]
                self.labels = list(labels_np[valid_indices])
            else:
                self.data = self.data[train_indices]
                self.labels = list(labels_np[train_indices])
        else:
            file = os.path.join(root, self.base_folder, self.test_list[0][0])
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] >= 3 else pickle.load(fo)
            self.data = entry['data']
            self.labels = entry['labels']
            self.data = self.data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))  # HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)
        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    def __init__(self, root='', train=True, valid=False, num_valid=5000,
                 transform=None, target_transform=None, download=False, seed=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.valid = valid
        self.num_valid = num_valid
        self.num_classes = 100
        np.random.seed(seed)
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        if self.train:
            self.data = []
            self.labels = []
            for fentry in self.train_list:
                file = os.path.join(root, self.base_folder, fentry[0])
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] >= 3 else pickle.load(fo)
                self.data.append(entry['data'])
                self.labels += entry['fine_labels']
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1)) # HWC
            labels_np = np.array(self.labels)
            # Split balanced valid/train set
            valid_per_class = self.num_valid // self.num_classes
            valid_indices = []
            train_indices = []
            for cls in range(self.num_classes):
                cls_indices = np.where(labels_np == cls)[0]
                np.random.shuffle(cls_indices)
                valid_cls_idx = cls_indices[:valid_per_class]
                train_cls_idx = cls_indices[valid_per_class:]
                valid_indices.extend(valid_cls_idx)
                train_indices.extend(train_cls_idx)
            np.random.shuffle(valid_indices)
            np.random.shuffle(train_indices)
            if self.valid:
                self.data = self.data[valid_indices]
                self.labels = list(labels_np[valid_indices])
            else:
                self.data = self.data[train_indices]
                self.labels = list(labels_np[train_indices])
        else:
            file = os.path.join(root, self.base_folder, self.test_list[0][0])
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1') if sys.version_info[0] >= 3 else pickle.load(fo)
            self.data = entry['data']
            self.labels = entry['fine_labels']
            self.data = self.data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))  # HWC

def build_dummy_dataset(args):
    """Build a dummy dataset for testing purposes."""
    # cretae some random data have shape (16, 3, 32, 32)
    # create dalaloader for train, valid, test
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100        

    train_data = torch.randn(16, 3, 32, 32)
    train_labels = torch.randint(0, num_classes, (16,))

    train_data = list(zip(train_data, train_labels))

    if args.n_omits > 0:
        train_data = omit_last_classes(train_data, args.n_omits)
    if args.imb_factor > 1:
        train_data = make_imbalanced_dataset(train_data, args)

    all_label = np.array([label for (_, label) in train_data])
    cnt_cls = np.array([sum(all_label == i) for i in range(args.n_classes)])
    print("Class distribution:", cnt_cls)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=4)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=4)

    return train_loader, valid_loader, test_loader

def build_dataset(args):
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

    if args.n_omits > 0:
        train_data = omit_last_classes(train_data, args.n_omits)
    if args.imb_factor > 1:
        train_data = make_imbalanced_dataset(train_data, args)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    return train_loader, valid_loader, test_loader


def make_imbalanced_dataset(dataset, args):
    rng = np.random.default_rng(args.seed)
    targets = np.array([label for (_, label) in dataset])
    cnt_cls = np.array([sum(targets == i) for i in range(args.n_classes)])
    img_max = min(cnt_cls)

    imb_factor = 1/args.imb_factor

    img_num_per_cls = []
    for cls_idx in range(args.n_classes):
        num = img_max * (imb_factor**(cls_idx / (args.n_classes - 1.0)))
        img_num_per_cls.append(int(num))

    indices_list = []
    for cls_idx, num_samples in enumerate(img_num_per_cls):
        cls_indices = np.where(targets == cls_idx)[0]
        sampled_indices = rng.choice(cls_indices, size=num_samples, replace=False)
        indices_list.append(sampled_indices)

    all_indices = np.concatenate(indices_list)
    return Subset(dataset, all_indices)

def omit_last_classes(dataset, n_omits):
    targets = np.array([label for (_, label) in dataset])
    max_class = max(targets) + 1
    keep_classes = set(range(max_class - n_omits))

    keep_indices = [i for i, t in enumerate(targets) if t in keep_classes]
    return Subset(dataset, keep_indices)