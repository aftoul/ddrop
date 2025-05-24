import torchvision.models as models
from .ddrop_layers import DummyNorm
from torchvision.ops.misc import MLP
import re
from torch import nn

def create_model(args, device):
    if args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        if args.model.startswith('resnet'):
            from . import small_resnets
            model = getattr(small_resnets, args.model)(
                    num_classes=num_classes,
                    factor=1 if num_classes == 10 else 2
                    )
        elif args.model.startswith('vgg'):
            from . import vgg_models
            model = getattr(vgg_models, args.model)(num_classes=num_classes)
    else:
        num_classes = 1000
        # ImageNet Models
        if args.model == 'vit':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            for name, module in model.named_modules():
                if isinstance(module, MLP):
                    module[0] = nn.Sequential(
                        module[0], 
                        DummyNorm(module[0].out_features, p=args.prob)
                    )
        elif args.model == 'swin':
            model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
            for name, module in model.named_modules():
                if isinstance(module, MLP):
                    module[0] = nn.Sequential(
                        module[0],
                        DummyNorm(module[0].out_features, p=args.prob)
                    )
        elif args.model == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif args.model == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif args.model == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.model == 'vgg11_bn':
            model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        elif args.model == 'vgg13_bn':
            model = models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
        elif args.model == 'vgg16_bn':
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        elif args.model == 'vgg19_bn':
            model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
    
    model = model.to(device)
    return model
