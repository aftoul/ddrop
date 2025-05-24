import torch
import torch.nn.utils.prune as prune
from torch import nn
import re
from models.ddrop_layers import DummyNorm

def prune_conv_using_batchnorm(conv_module, bn_module, amount):
    bn_weights = bn_module.weight.data.abs()
    num_prune = int(amount * bn_weights.shape[0])
    _, prune_indices = torch.topk(bn_weights, num_prune, largest=False)
    
    # Create masks
    weight_mask = torch.ones_like(conv_module.weight.data)
    weight_mask[prune_indices] = 0
    bias_mask = torch.ones_like(conv_module.bias.data) if conv_module.bias else None
    
    # Apply pruning
    prune.custom_from_mask(conv_module, 'weight', weight_mask)
    if bias_mask is not None:
        prune.custom_from_mask(conv_module, 'bias', bias_mask)
    
    # Zero BN parameters
    bn_module.weight.data[prune_indices] = 0
    bn_module.bias.data[prune_indices] = 0
    bn_module.running_mean[prune_indices] = 0
    bn_module.running_var[prune_indices] = 0

def prune_linear_using_dummynorm(linear_module, dn_module, amount):
    dn_weights = dn_module.weight.data.abs()
    num_prune = int(amount * dn_weights.shape[0])
    _, prune_indices = torch.topk(dn_weights, num_prune, largest=False)
    
    weight_mask = torch.ones_like(linear_module.weight.data)
    weight_mask[prune_indices] = 0
    bias_mask = torch.ones_like(linear_module.bias.data) if linear_module.bias is None else None
    
    prune.custom_from_mask(linear_module, 'weight', weight_mask)
    if bias_mask is not None:
        prune.custom_from_mask(linear_module, 'bias', bias_mask)
    
    dn_module.weight.data[prune_indices] = 0

def prune_linear_using_layernorm(linear_module, ln_module, amount=0.2):
    ln_weights = ln_module.weight.data.abs()
    num_units = ln_weights.shape[0]
    num_prune = int(amount * num_units)
    _, prune_indices = torch.topk(ln_weights, num_prune, largest=False)
    weight_mask = torch.ones_like(linear_module.weight.data)
    weight_mask[prune_indices, :] = 0
    if linear_module.bias is not None:
        bias_mask = torch.ones_like(linear_module.bias.data)
        bias_mask[prune_indices] = 0
    else:
        bias_mask = None
    prune.custom_from_mask(linear_module, name='weight', mask=weight_mask)
    if bias_mask is not None:
        prune.custom_from_mask(linear_module, name='bias', mask=bias_mask)
    ln_module.weight.data[prune_indices] = 0
    ln_module.bias.data[prune_indices] = 0

def prune_model(model, args):
    if args.model in ['vit', 'swin']:
        # Prune linear layers with DummyNorm
        for name, module in model.named_modules():
            if isinstance(module, DummyNorm):
                linear_name = re.sub(r'\.1$', '.0', name)
                linear_module = model.get_submodule(linear_name)
                prune_linear_using_dummynorm(linear_module, module, args.amount)
    elif 'resnet' in args.model:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if not 'conv' in name:
                    continue
                bn_name = name.replace('conv', 'bn')
                bn_module = model.get_submodule(bn_name)
                if bn_module:
                    prune_conv_using_batchnorm(module, bn_module, args.amount)
    elif 'vgg' in args.model:
        with torch.no_grad():
            for n, module in model.named_modules():
                if n in ['features.0', 'fc']:
                    continue
                if isinstance(module, nn.Linear):
                    try:
                        lin_name, lin_num = n.rsplit('.', 1)
                        ln_name = lin_name + '.' + str(int(lin_num) + 1)
                        ln_module = model.get_submodule(ln_name)
                        if not isinstance(ln_module, nn.LayerNorm):
                            continue
                        prune_linear_using_layernorm(module, ln_module, amount)
                        module.weight_orig *= module.weight_mask
                    except:
                        pass
                if isinstance(module, nn.Conv2d):
                    try:
                        conv_name, conv_num = n.rsplit('.', 1)
                        bn_name = conv_name + '.' + str(int(conv_num) + 1)
                        bn_module = model.get_submodule(bn_name)
                        if not isinstance(bn_module, nn.BatchNorm2d):
                            continue
                        prune_conv_using_batchnorm(module, bn_module, amount)
                        module.weight_orig *= module.weight_mask
                    except:
                        pass
