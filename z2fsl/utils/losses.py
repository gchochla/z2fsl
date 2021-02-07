"""Losses."""

import torch
import torch.nn as nn
import torch.autograd as autograd


class EpisodeCrossEntropyLoss(nn.Module):
    """FSL episode classification loss.

    Attributes:
        criterion (`nn.Module`): module that computes loss.
        reduction (`str`): how to reduce vector results.
    """

    def __init__(self, reduction='mean'):
        """Init.

        Args:
            reduction (`str`, optional): how to reduce
                vector results, default `'mean'`.
        """

        assert reduction in ('mean', 'sum')

        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.reduction = reduction

    def forward(self, logits):
        """Computes loss.

        Args:
            logits (`list`): list of torch.Tensor logits. Index
                i corresponds to i-th class.

        Returns:
            Loss.
        """

        loss = []
        for i, class_logits in enumerate(logits):
            labels = i * torch.ones(
                class_logits.size(0), dtype=torch.long, device=class_logits.device
            )
            loss.append(self.criterion(class_logits, labels))

        if self.reduction == 'mean':
            return sum(loss) / len(loss)
        elif self.reduction == 'sum':
            return sum(loss)


def gradient_penalty(critic, real_samples, gen_samples, real_cond=None, gen_cond=None):
    """Computes the gradient penalty of a critic in
    the WGAN framework to enforce the Lipschitz constraint.

    Args:
        critic (`nn.Module`): WGAN critic.
        real_samples (`torch.Tensor`): real samples
            for the critic.
        gen_samples (`torch.Tensor`): generated samples
            for the critic.
        real_cond (`torch.Tensor`, optional): real conditioning
            variable for the critic.
        gen_cond (`torch.Tensor`, optional): generated
            conditioning variable for the critic. If
            not set, real_cond is solely used.

    Returns:
        The gradient penalty as defined in
        `Improved Training of Wasserstein GANs`.
    """

    alpha = torch.empty(real_samples.size(0), 1, device=real_samples.device).uniform_()
    inter_samples = alpha * real_samples + (1 - alpha) * gen_samples.detach()
    inter_samples = nn.Parameter(inter_samples)  # necessary for autograd
    if real_cond is not None:
        if gen_cond is None:
            gen_cond = real_cond
        inter_cond = alpha * real_cond + (1 - alpha) * gen_cond
        outs = critic(inter_samples, inter_cond)
    else:
        outs = critic(inter_samples)
    nabla = autograd.grad(outs.sum(), inter_samples, create_graph=True)[0]
    return ((nabla.norm(2, dim=-1) - 1) ** 2).mean()


def kl_divergence(mus, logvars):
    """Computes KL divergence of N(mus, e^{logvar/2}) from N(0, I).

    Args:
        mus (`torch.Tensor`): size (batch_size, cond_dim) with the means.
        logvars (`torch.Tensor`): size (batch_size, cond_dim) with the
            logarithm of the std devs.

    Returns:
        Loss.
    """

    return (
        0.5
        * (
            logvars.exp().sum(dim=-1) + (mus ** 2).sum(dim=-1) - logvars.sum(dim=-1) - mus.size(-1)
        ).mean()
    )
