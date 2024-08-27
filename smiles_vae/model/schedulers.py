import math
from typing import List, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    PyTorch's CosineAnnealingWarmRestarts anneals the initial learning rate 
    in a cosine manner until it hits a restart after which the learning rate is
    set back to the initial value and the cycle happens again. This scheduler adds 
    damping such that the maximum learning rate decreases after each restart.
    """
    def __init__(self,
                 optimizer,
                 T_0,
                 T_mult=1,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False,
                 decay=1,
                 ):
        super().__init__(optimizer,
                         T_0,
                         T_mult=T_mult,
                         eta_min=eta_min,
                         last_epoch=last_epoch,
                         verbose=verbose,
                         )
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).

    Args:
        optimizer: A PyTorch optimizer.
        warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        total_epochs: The total number of epochs.
        steps_per_epoch: The number of steps (batches) per epoch.
        init_lr: The initial learning rate.
        max_lr: The maximum learning rate (achieved after warmup_epochs).
        final_lr: The final learning rate (achieved after total_epochs).
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        Args:
            current_step: Optionally specify what step to set the learning rate to.
                          If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]



class KLDAnnealer:
    """
    https://github.com/molecularsets/moses/issues/42
    Training the encoder and decoder right off the bat variationally (KL-term constant) can lead to a lot of
    instability while training. So a good first step is to train them as a AE and at some moment slowly switch
    on the KL term. It allows the model to arrive to a 'decent' spot (trained as AE) before going VAE.
    A similar pattern lies with ORGAN, you need to train pre-train your generators so that they are in a
    decent spot before competing with the discriminator. There are many ways of doing this, most are just
    engineering, hence MOSES's approach also works.
    """
    def __init__(self,
                 start_epoch=0,
                 end_epoch=30,
                 kld_w_start=0,
                 kld_w_end=1,
                 ):
        """
        Class for using a piece-wise linear function to schedule
        the weighting of KLD penalty when training a VAE.

        Args:
            start_epoch: Epoch to start increasing KLD weight from.
            end_epoch: Epoch to stop increasing KLD weight from.
            kld_w_start: Initial KLD weight value.
            kld_w_end:: Final KLD weight value.
        """
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.kld_w_start = kld_w_start
        self.kld_w_end = kld_w_end

        self.inc = (self.kld_w_end - self.kld_w_start) / (self.end_epoch - self.start_epoch)
    
    def __call__(self, epoch):
        if epoch < self.start_epoch:
            return 0  # could also do self.kld_w_start depending on desired behavior
        elif epoch >= self.end_epoch:
            return self.kld_w_end
        else:
            m = (epoch - self.start_epoch)
            return self.kld_w_start + m * self.inc
