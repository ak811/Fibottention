import os
import scipy.io
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import glob

class CIFAR10CDataset(Dataset):
    def __init__(self, data_dir, corruption_type, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.transform = transform

        self.images = np.load(os.path.join(self.data_dir, f'{self.corruption_type}.npy'))
        self.labels = np.load(os.path.join(self.data_dir, 'labels.npy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.int64)


class CIFAR100CDataset(Dataset):
    def __init__(self, data_dir, corruption_type, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.transform = transform

        self.images = np.load(os.path.join(self.data_dir, f'{self.corruption_type}.npy'))
        self.labels = np.load(os.path.join(self.data_dir, 'labels.npy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.int64)

class TinyImageNetCDataset(Dataset):
    def __init__(self, data_dir, corruption_type, transform=None):
        self.data_dir = data_dir
        self.corruption_type = corruption_type
        self.transform = transform

        self.wnids = self.load_wnids()

        self.class_mapping = self.create_class_mapping()

        self.image_paths = []
        self.labels = []

        for severity in range(1, 6):
            severity_dir = os.path.join(self.data_dir, self.corruption_type, str(severity))

            class_folders = sorted(glob.glob(os.path.join(severity_dir, '*')))

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
        
            if class_name in self.wnids:
                if class_name in self.class_mapping:
                    label = self.class_mapping[class_name]

                    print(f"Class '{class_name}' found in wnids and mapped to label {label}")
                    
                    image_files = glob.glob(os.path.join(class_folder, '*.JPEG'))
                    self.image_paths.extend(image_files)
                    self.labels.extend([label] * len(image_files))
                else:
                    print(f"Class '{class_name}' is in wnids but not in class_mapping, skipping")
                    continue
            else:
                print(f"Skipping class '{class_name}' as it's not in the wnids list")

    def load_wnids(self):
        wnids_file = os.path.join('image_classification/exp/tinyimagenet', 'wnids.txt')
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return wnids

    def create_class_mapping(self):
        class_mapping = {}
        severity_dir = os.path.join(self.data_dir, self.corruption_type, '1')
        class_folders = sorted(glob.glob(os.path.join(severity_dir, '*')))

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            if class_name in self.wnids:
                class_mapping[class_name] = self.wnids.index(class_name)
            else:
                print(f"Class {class_name} not found in wnids")

        print(f"Class Mapping: {class_mapping}")
        return class_mapping

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.int64)

class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['class_label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.int64)


class OxfordFlowers102Dataset(Dataset):
    def __init__(self, image_dir, label_file, split_file, is_train, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        mat = scipy.io.loadmat(label_file)
        self.labels = mat['labels'][0] - 1

        split_mat = scipy.io.loadmat(split_file)
        if is_train:
            self.image_ids = split_mat['trnid'][0]
        else:
            self.image_ids = split_mat['tstid'][0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx]).zfill(5)
        image_path = os.path.join(self.image_dir, f'image_{image_id}.jpg')
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[self.image_ids[idx] - 1]
        return image, torch.tensor(label, dtype=torch.int64)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == "c10":
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "c100":
        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "c10c":
        if is_train:
            dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform)
        else:
            data_dir = '/data/CIFAR-10-C/'
            corruption_type = 'gaussian_noise'
            dataset = CIFAR10CDataset(data_dir, corruption_type, transform=transform)
    elif args.dataset == "c100c":
        if is_train:
            dataset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform)
        else:
            data_dir = '/data/CIFAR-100-C/'
            corruption_type = 'gaussian_noise'
            dataset = CIFAR100CDataset(data_dir, corruption_type, transform=transform)
    elif args.dataset == "tinyimagenet":
        root = os.path.join('/data/tiny-imagenet-200', 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.dataset == "tinyimagenetc":
        if is_train:
            root = os.path.join('/data/tiny-imagenet-200', 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            data_dir = '/data/Tiny-ImageNet-C/'
            corruption_type = 'gaussian_noise'
            dataset = TinyImageNetCDataset(data_dir, corruption_type, transform=transform)
    elif args.dataset == "imagenet":
        root = '../../imagenet'
        dataset = torchvision.datasets.ImageNet(
            root=root, split='train' if is_train else 'val', download=False, transform=transform)
    elif args.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            root='./data', train=is_train, download=True, transform=transform)
    elif args.dataset == "flower102":
        image_dir = os.path.join('/data/flower102/jpg')
        label_file = os.path.join('/data/flower102/imagelabels.mat')
        split_file = os.path.join('/data/flower102/setid.mat')
        dataset = OxfordFlowers102Dataset(image_dir, label_file, split_file, is_train, transform=transform)
    elif args.dataset == "bucket":

        ds = load_dataset("bghira/photo-concept-bucket")

        unique_classes = ds['train'].unique('class_label')
        print(len(unique_classes))

        train_data, test_data = train_test_split(ds['train'], test_size=0.2, random_state=42)

        if is_train:
            dataset = HuggingFaceDataset(train_data, transform)
        else:
            dataset = HuggingFaceDataset(test_data, transform)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if args.dataset == "c10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "c100":
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
    elif args.dataset == "mnist":
        mean = (0.1307,)
        std = (0.3081,)

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
