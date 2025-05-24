import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

def get_probs(w, p):
    if len(w.shape) > 1:
        w = w.reshape(w.shape[0], -1)
        w = w.square().sum(-1)
    a = w.argsort(-1) / (w.shape[-1] - 1)
    probs = p[0] + a * (p[1] - p[0])
    return probs

class ScheduledDropMixin:
    def setup_schedule(self, p, schedule=None, total_steps=1000):
        self.p = p
        self.schedule = schedule
        self.steps = 0
        self.total_steps = total_steps

    def ddrop_step(self):
        self.steps = min(self.steps + 1, self.total_steps)

    def current_probs(self):
        if self.schedule is None:
            return self.p
        scale = {
            "cosine": math.cos(0.5 * math.pi * self.steps / self.total_steps),
            "linear": 1 - self.steps / self.total_steps,
        }.get(self.schedule, 1.0)
        pmin, pmax = self.p
        pmin = pmin + (1 - pmin) * scale
        pmax = pmax + (1 - pmax) * scale
        return pmin, pmax

class DDropMixin(ScheduledDropMixin):
    def apply_ddrop(self):
        self.original_forward = self.forward
        self.forward = self.ddrop_forward
    def remove_ddrop(self):
        if hasattr(self, 'original_forward'):
            self.forward = self.original_forward

class DDropLinear(nn.Linear, DDropMixin):
    def __init__(self, in_features, out_features, bias=True, p=(0, 0.95), schedule=None, total_steps=1000):
        super().__init__(in_features, out_features, bias)
        self.setup_schedule(p, schedule, total_steps)

    def ddrop_forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if not self.training:
            return y
        probs = get_probs(self.weight, self.current_probs())
        mask = torch.bernoulli(probs.reshape(1, -1).expand(y.shape[0], -1))
        return torch.einsum('b...o,bo->b...o', y, mask)

    @classmethod
    def from_original(cls, layer, **kwargs):
        new = cls(layer.in_features, layer.out_features, layer.bias is not None, **kwargs)
        new.load_state_dict(layer.state_dict())
        return new

class DDropConv2d(nn.Conv2d, DDropMixin):
    def __init__(self, *args, p=(0, 0.95), schedule=None, total_steps=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_schedule(p, schedule, total_steps)

    def ddrop_forward(self, x):
        y = super().forward(x)
        if not self.training:
            return y
        probs = get_probs(self.weight.data.abs(), self.current_probs())
        mask = torch.bernoulli(probs.reshape(1, -1).expand(y.shape[0], -1))
        return torch.einsum('bo...,bo->bo...', y, mask)

    @classmethod
    def from_original(cls, layer, **kwargs):
        new = cls(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            layer.stride, layer.padding, layer.dilation,
            layer.groups, layer.bias is not None, **kwargs)
        new.load_state_dict(layer.state_dict())
        return new

class DDropBatchNorm2d(nn.BatchNorm2d, DDropMixin):
    def __init__(self, num_features, p=(0, 0.95), schedule=None, total_steps=1000, **kwargs):
        super().__init__(num_features, **kwargs)
        self.setup_schedule(p, schedule, total_steps)

    def ddrop_forward(self, x):
        y = super().forward(x)
        if not self.training:
            return y
        probs = get_probs(self.weight.data.abs(), self.current_probs())
        mask = torch.bernoulli(probs.reshape(1, -1).expand(x.shape[0], -1))
        mask = mask.view([x.shape[0], x.shape[1]] + [1]*(x.ndim - 2))
        return y * mask

    @classmethod
    def from_original(cls, layer, **kwargs):
        new = cls(layer.num_features, **kwargs)
        new.load_state_dict(layer.state_dict())
        return new

class DummyNorm(nn.Module, DDropMixin):
    def __init__(self, dim, p=(0, 0.95), schedule=None, total_steps=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.setup_schedule(p, schedule, total_steps)

    def forward(self, x):
        y = self.weight * x
        return y

    def ddrop_forward(self, x):
        y = self.weight * x
        if not self.training:
            return y
        probs = get_probs(self.weight, self.current_probs())
        mask = torch.bernoulli(probs.reshape(1, -1).expand(x.shape[0], -1))
        return y * mask.reshape([x.shape[0]] + [1]*(x.ndim-2) + [x.shape[-1]])

    @classmethod
    def from_original(cls, layer, **kwargs):
        new = cls(layer.weight.shape[0], **kwargs)
        new.weight.data.copy_(layer.weight.data)
        return new

def apply_resnet(model, args):
    for name, module in model.named_modules():
        # Skip the top-level conv and fc
        if not isinstance(module, nn.BatchNorm2d) or 'bn' not in name:
            continue
        if name == 'bn1':
            continue
        if name.endswith('bn2') and args.model in ('resnet18', 'resnet34'):
            continue
        if name.endswith('bn3'):
            continue
        parent = model
        subpath = name.split('.')
        for p in subpath[:-1]:
            parent = getattr(parent, p)
        key = subpath[-1]

        old_module = getattr(parent, key)
        new_module = None

        new_module = DDropBatchNorm2d.from_original(
                old_module, p=args.prob, schedule=args.schedule, total_steps=args.total_steps)

        if new_module is not None:
            setattr(parent, key, new_module)

def apply_vgg(model, args):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for i, sub in enumerate(module):
                if isinstance(sub, nn.Linear):
                    module[i] = DDropLinear.from_original(sub, p=args.prob, schedule=args.schedule, total_steps=args.total_steps)
                elif isinstance(sub, nn.BatchNorm2d):
                    module[i] = DDropBatchNorm2d.from_original(sub, p=args.prob, schedule=args.schedule, total_steps=args.total_steps)

def apply_transformer(model, args):
    """
    Inject DummyNorm layers into transformer blocks.
    Replaces MLP Linear layers with Sequential(Linear, DummyNorm).
    """

    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) or len(module) < 2:
            continue
        first = module[0]
        if not isinstance(first, nn.Linear):
            continue
        dummy = DummyNorm(first.out_features, p=args.prob, schedule=args.schedule, total_steps=args.total_steps)
        dummy.to(first.weight.device)
        dummy = dummy.to(first.weight.dtype)
        new_seq = nn.Sequential(first, dummy)

        # Replace in parent
        parent = model
        keys = name.split('.')
        for k in keys[:-1]:
            parent = getattr(parent, k)
        setattr(parent, keys[-1], new_seq)

def apply_ddrop(model, args):
    """Applies DDrop modifications based on model architecture"""
    if 'resnet' in args.model:
        apply_resnet(model, args)
    elif args.model in ['vit', 'swin']:
        apply_transformer(model, args)
    elif args.model.startswith('vgg'):
        apply_vgg(model, args)

def remove_ddrop(model, args):
    for m in model.modules():
        if hasattr(model, 'remove_ddrop'):
            self.remove_ddrop()
