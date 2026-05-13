import random
from PIL import ImageOps
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageNet
from torch.utils.data import DataLoader
import torch


class GaussianBlur(object):
    """Gaussian blur augmentation from DINO."""

    def __init__(self, p=0.5, kernel_size=23, sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return transforms.GaussianBlur(self.kernel_size, sigma)(img)
        return img


class Solarization(object):
    """Solarization augmentation from DINO."""

    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img, threshold=128)
        return img


class MultiCropTransform:
    """Multi-crop augmentation: 2 global + 6 local crops in DINO style."""

    def __init__(self, num_local_crops=6):
        self.num_local_crops = num_local_crops

        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.4, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.4, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                96,
                scale=(0.05, 0.4),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

    def __call__(self, image):
        global_views = [
            self.global_transform1(image),
            self.global_transform2(image)
        ]
        local_views = [
            self.local_transform(image)
            for _ in range(self.num_local_crops)
        ]

        return torch.stack(global_views), torch.stack(local_views)


class StandardTransform:
    """Standard transform for evaluation"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    def __call__(self, image):
        return self.transform(image)


def get_cifar100_loaders(batch_size=32, num_workers=4, num_train=0.8):
    """Get CIFAR-100 train/val/test loaders"""
    
    train_transform = MultiCropTransform()
    eval_transform = StandardTransform()
    
    train_dataset = CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=eval_transform
    )
    
    # Split train into train/val
    train_size = int(len(train_dataset) * num_train)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader