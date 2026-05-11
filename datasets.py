from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageNet
from torch.utils.data import DataLoader
import torch


class MultiCropTransform:
    """Multi-crop augmentation: 2 global + 6-8 local crops"""
    
    def __init__(self, num_local_crops=8):
        self.num_local_crops = num_local_crops
        
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.05, 0.4),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    def __call__(self, image):
        global_views = [
            self.global_transform(image),
            self.global_transform(image)
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
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader