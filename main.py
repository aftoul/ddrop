import argparse
import torch
from models.model_factory import create_model
from data.dataloaders import get_dataloaders
from utils.train_utils import train_loop, test, setup_optimizer
from models.ddrop_layers import apply_ddrop, remove_ddrop
from utils.pruning import prune_model

def main():
    parser = argparse.ArgumentParser(description='DDrop Training Framework')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--data-dir', type=str, default='./data')
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet18', 'resnet34', 'resnet50', 
                                 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
                                 'vit', 'swin'])
    parser.add_argument('--prob', type=float, nargs=2, default=[0.2, 1.0])
    parser.add_argument('--schedule', type=str, default='constant',
                        choices=['constant', 'cosine', 'linear'])
    parser.add_argument('--total_steps', type=int, default=None)
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight-decay', type=float, default=4e-3)
    parser.add_argument('--before-epochs', type=int, default=0)
    
    # Pruning parameters
    parser.add_argument('--amount', type=float, default=0.7)
    
    # Fine-tuning parameters
    parser.add_argument('--ft-epochs', type=int, default=30)
    parser.add_argument('--ft-lr', type=float, default=2e-2)
    
    # Misc
    parser.add_argument('--seed', type=int, default=5)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.total_steps is None:
        args.total_steps = args.epochs // 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data and model
    train_loader, test_loader = get_dataloaders(args)
    model = create_model(args, device)

    # Initial training phase
    if args.before_epochs > 0:
        optimizer, scheduler = setup_optimizer(model, train_loader, args, phase='initial')
        train_loop(model, device, train_loader, test_loader, optimizer, scheduler, args.before_epochs, args)

    # Apply DDrop
    apply_ddrop(model, args)
    model.to(device)

    # Main training phase
    optimizer, scheduler = setup_optimizer(model, train_loader, args, phase='main')
    train_loop(model, device, train_loader, test_loader, optimizer, scheduler, args.epochs, args)

    # Pruning phase
    remove_ddrop(model, args)
    prune_model(model, args)
    model.to(device)
    test(model, device, test_loader, 'Pruned', args)

    # Fine-tuning phase
    optimizer, scheduler = setup_optimizer(model, train_loader, args, phase='finetune')
    train_loop(model, device, train_loader, test_loader, optimizer, scheduler, args.ft_epochs, args)

if __name__ == '__main__':
    main()
