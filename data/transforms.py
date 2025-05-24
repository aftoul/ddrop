import torch
from torchvision import transforms, datasets
from torchvision.transforms import v2

def get_transforms(dataset):
    if dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
        ])
    elif dataset == 'imagenet':
        train_transform = v2.Compose([
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.PILToTensor(),
            v2.RandAugment(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return train_transform, test_transform
