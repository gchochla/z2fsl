"""Various major components."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP.

    Attributes:
        layers (`nn.Module`): sequence of layers.
    """

    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers=None,
        dropout=0,
        hidden_actf=nn.LeakyReLU(0.2),
        output_actf=nn.ReLU(),
        noise_std=0,
    ):
        """Init.

        Args:
            in_features (`int`): input dimension.
            out_features (`int`): final output dimension.
            hidden_layers (`list` of `int`s | `int`| `None`, optional):
                list of hidden layer sizes of arbitrary length or int for one
                hidden layer, default `None` (no hidden layers).
            dropout(`float`, optional): dropout probability, default `0`.
            hidden_actf (activation function, optional): activation
                function of hidden layers, default `nn.LeakyReLU(0.2)`.
            output_actf (activation function, optional): activation
                function of output layers, default `nn.ReLU()`.
            noise_std (`float`, optional): std dev of gaussian noise to add
                to MLP result, default `0` (no noise).
        """

        if hidden_layers is None:
            hidden_layers = []
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        super().__init__()

        hidden_layers = [in_features] + hidden_layers + [out_features]

        layers = []
        for i, (in_f, out_f) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            layers.append(nn.Linear(in_f, out_f))

            if i != len(hidden_layers) - 2:
                # up to second-to-last layer
                layers.append(hidden_actf)
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(output_actf)  # ok to use relu, resnet feats are >= 0

        self.layers = nn.Sequential(*layers)
        self.noise_aug = GaussianNoiseAugmentation(std=noise_std)

    def forward(self, x):
        """Forward propagation.

        Args:
            x (`torch.Tensor`): input of size (batch, in_features).

        Returns:
            A torch.Tensor of size (batch, out_features).
        """
        return self.noise_aug(self.layers(x))

    def init_diagonal(self):
        """Sets weights of linear layers to approx I
        and biases to 0.
        """

        def init_fn(mod):
            """Function to pass to .apply()."""
            classname = mod.__class__.__name__
            if classname.find('Linear') != -1:
                init = torch.randn_like(mod.weight) / 10
                init[range(mod.in_features), range(mod.in_features)] = 1
                mod.weight = nn.Parameter(init, requires_grad=True)
                if mod.bias is not None:
                    mod.bias = nn.Parameter(torch.zeros_like(mod.bias), requires_grad=True)

        self.apply(init_fn)


class ConcatMLP(MLP):
    """Simple MLP that accepts two inputs
    which are to be concatenated."""

    def forward(self, x, y):
        """Concatenates input and noise and forward propagates.
        x's and input's dimensions must add up to in_features.

        Args:
            x (`torch.Tensor`): input.
            noise (`torch.Tensor`): randomly sampled noise.

        Returns:
            A torch.Tensor of size (batch, out_features).
        """

        x = torch.cat((x, y), dim=1)
        return super().forward(x)


class GaussianNoiseAugmentation(nn.Module):
    """Module that adds gaussian noise to tensor
    (for robustness, ...).

    Attributes:
        mean(torch.Tensor): scalar mean of Gaussian of noise to add.
        std(torch.Tensor): scalar std dev of Gaussian of noise to add.
    """

    def __init__(self, mean=0.0, std=1.0):
        """Init.

        Args:
            mean (`float`, optional): mean of Gaussian of noise to add,
                default `0`.
            std(`float`, optional): std of Gaussian of noise to add,
                default `1`. `0` deactivates module.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        """Forward propagation.

        Args:
            x (`torch.Tensor`): input.

        Returns:
            Input plus noise if training or just input
            if evaluating.
        """
        if self.training and self.std != 0:
            return x + torch.empty_like(x).normal_(self.mean, self.std)
        return x

    def __repr__(self):
        """Repr.

        Print mean and std along with classname.
        """
        if self.std > 0:
            return '{}(mean={}, std={})'.format(self._get_name(), self.mean, self.std)
        return self._get_name() + '(Deactivated)'


class CVAE(nn.Module):
    """Conditional VAE.

    Attributes:
        encoder (`nn.Module`): encodes features to concatenated
            mean and logarithm of variance.
        decoder (`nn.Module`): decodes latent variable and
            conditioning variable to feature.
    """

    def __init__(self, feature_dim, latent_dim, cond_dim, hidden_layers):
        """Init.

        Args:
            feature_dim (`int`): dimensionality of features.
            latent_dim (`int`): dimensionality of latent variables.
            cond_dim (`int`): dimensionality of conditioning variables.
            hidden_layers (`list` | `int` | `None`): hidden layers of
                encoder and decoder.
        """
        super().__init__()

        # handle cases to add another layer at encoder
        if hidden_layers:
            if isinstance(hidden_layers, int):
                hidden_layers = [hidden_layers]
        else:
            hidden_layers = []
        self.encoder = ConcatMLP(
            feature_dim + cond_dim,
            2 * latent_dim,
            hidden_layers=list(reversed(hidden_layers)),
            output_actf=nn.Identity(),
        )
        self.decoder = ConcatMLP(
            latent_dim + cond_dim,
            feature_dim,
            hidden_layers=hidden_layers,
            output_actf=nn.Sigmoid(),
        )

    def forward(self, x, cond):
        """Forward propagation.

        Args:
            x (`torch.Tensor`): input feature.
            cond (`torch.Tensor`): conditioning variable.

        Returns:
            The reconstruction of `x`, the mean of the latent
            variable and the logarithm of its variance.
        """

        latent_code, mean, logvar = self.encode(x, cond)
        x_hat = self.decode(cond, latent_code)
        return x_hat, mean, logvar

    def decode(self, cond, noise):
        """Decodes latent variable and conditioning variable.

        Args:
            cond (`torch.Tensor`): conditioning variable.
            noise (`torch.Tensor`): latent variable.

        Returns:
            The reconstructed feature.
        """
        x = self.decoder(cond, noise)
        return x

    def encode(self, feature, cond):
        """Encodes features.

        Args:
            feature (`torch.Tensor`): feature to be encoded.

        Returns:
            The latent variable, its mean and the logarithm
            of its variance.
        """
        latent_stats = self.encoder(feature, cond)
        split_dim = latent_stats.size(1) // 2
        mean, logvar = latent_stats[:, :split_dim], latent_stats[:, split_dim:]
        latent_code = mean + torch.empty_like(logvar).normal_() * (logvar / 2).exp()
        return latent_code, mean, logvar
