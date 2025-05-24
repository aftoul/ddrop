from torch.utils.data import DataLoader, default_collate
from torchvision import datasets
from .transforms import get_transforms
from torchvision.transforms import v2

def get_dataloaders(args):
    train_transform, test_transform = get_transforms(args.dataset)
    collate_fn = default_collate
    if args.dataset in ['cifar10', 'cifar100']:
        Dataset = getattr(datasets, args.dataset.upper())
        train_set = Dataset(root=args.data_dir, train=True, download=True, transform=train_transform)
        test_set = Dataset(root=args.data_dir, train=False, transform=test_transform)
    elif args.dataset == 'imagenet':
        train_set = datasets.ImageFolder(f'{args.data_dir}/train', transform=train_transform)
        test_set = datasets.ImageFolder(f'{args.data_dir}/val', transform=test_transform)
        
        # Add CutMix/MixUp
        cutmix = v2.CutMix(alpha=1.0, num_classes=1000)
        mixup = v2.MixUp(alpha=0.2, num_classes=1000)
        cutmix_or_mixup = v2.RandomApply([v2.RandomChoice([cutmix, mixup], p=[0.4, 0.6])], p=0.9)
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=args.test_batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader
