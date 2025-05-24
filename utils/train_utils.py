import torch
from tqdm import tqdm
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from transformers import get_cosine_schedule_with_warmup

def train_loop(model, device, train_loader, test_loader, optimizer, scheduler, epochs, args):
    model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(
                    enabled=device.type=='cuda',
                    dtype=torch.bfloat16,
                    device_type=device.type):
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            break
        for m in model.modules():
            if hasattr(m, 'ddrop_step'):
                m.ddrop_step()
        test(model, device, test_loader, epoch, args)
        break

def test(model, device, test_loader, epoch, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            break

    if isinstance(epoch, int):
        prefix = f'Epoch {epoch}'
    else:
        prefix = epoch
    print(f'{prefix}: Accuracy {100.*correct/total:.2f}%')

def setup_optimizer(model, train_loader, args, phase='main'):
    params = [
        {'params': [p for n,p in model.named_parameters() if 'weight' in n], 'weight_decay': args.weight_decay},
        {'params': [p for n,p in model.named_parameters() if 'bias' in n], 'weight_decay': 0.0}
    ]
    
    if phase == 'finetune':
        lr = args.ft_lr
        epochs = args.ft_epochs
    else:
        lr = args.lr
        epochs = args.epochs
    if args.model in ['vit', 'swin']:
        optimizer = AdamW(params, lr=lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=len(train_loader)*(epochs/10),
            num_training_steps=len(train_loader)*epochs
        )
    else:
        optimizer = SGD(params, lr=lr, momentum=0.9)
        scheduler = OneCycleLR(
            optimizer, lr, 
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            anneal_strategy='cos'
        )
    return optimizer, scheduler
