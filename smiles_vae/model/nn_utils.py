import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def set_seed(seed):
    """
    Sets the seed for generating random numbers in for pseudo-random number 
    generators in Python.random, numpy, and PyTorch.

    Args:
        seed (int): The desired seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    Args:
        model: An nn.Module.

    Returns:
        Number of trainable parameters (int).
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def vae_loss(recon_x, x, mu, logvar, voc, kld_weight=1, verbose=False):
    # if use reduction='mean', then KLD must be 0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar).sum(1).mean()
    BCE = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)),
                          x.view(-1),
                          ignore_index=voc.vocab['<PAD>'],
                          reduction='sum',
                          )
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if verbose:
        print(f'BCE: {BCE}')
        print(f'KLD: {KLD}')
        print(f'BCE/KLD: {BCE/KLD}')
    return kld_weight * KLD + BCE
