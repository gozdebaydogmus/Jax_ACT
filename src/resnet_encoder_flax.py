# resnet_encoder_flax.py
import jax.numpy as jnp
import flax.linen as nn
from typing import Literal

IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD  = jnp.array([0.229, 0.224, 0.225])

def normalize_imagenet(x: jnp.ndarray) -> jnp.ndarray:
    return (x - IMAGENET_MEAN) / IMAGENET_STD

class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    @nn.compact
    def __call__(self, x, *, train: bool):
        identity = x
        y = nn.Conv(self.features, (3,3), (self.stride,self.stride), padding="SAME", use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.features, (3,3), (1,1), padding="SAME", use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        if identity.shape[-1] != self.features or self.stride != 1:
            identity = nn.Conv(self.features, (1,1), (self.stride,self.stride), use_bias=False)(identity)
            identity = nn.BatchNorm(use_running_average=not train)(identity)
        return nn.relu(identity + y)

def _layer(x, f, n, s, train, name):
    for i in range(n):
        x = BasicBlock(f, stride=(s if i==0 else 1), name=f"{name}_b{i+1}")(x, train=train)
    return x

class ResNetEncoder(nn.Module):
    variant: Literal["resnet18","resnet34"] = "resnet18"
    d_model: int = 256
    imagenet_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # x: [B,H,W,3], float in [0,1]
        if self.imagenet_norm:
            x = normalize_imagenet(x)
        x = nn.Conv(64, (7,7), (2,2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3,3), (2,2), padding="SAME")
        layers = (2,2,2,2) if self.variant=="resnet18" else (3,4,6,3)
        x = _layer(x,  64, layers[0], 1, train, "l1")
        x = _layer(x, 128, layers[1], 2, train, "l2")
        x = _layer(x, 256, layers[2], 2, train, "l3")
        x = _layer(x, 512, layers[3], 2, train, "l4")
        # Global avg pool + proj
        x = jnp.mean(x, axis=(1,2))       # [B, 512]
        x = nn.Dense(self.d_model)(x)     # [B, D]
        return x
