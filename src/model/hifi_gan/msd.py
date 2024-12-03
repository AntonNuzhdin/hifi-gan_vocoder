import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


def init_conv(in_channels, out_channels, kernel_size, stride, padding, groups=1, use_spectral_norm=False):
    conv = nn.Conv1d(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride,
        padding=padding, groups=groups
    )
    if use_spectral_norm:
        return spectral_norm(conv)
    else:
        return weight_norm(conv)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_spectral_norm=False):
        super().__init__()
        self.conv = init_conv(
            in_channels, out_channels,
            kernel_size, stride, padding,
            groups, use_spectral_norm
        )
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.conv(x))


class SubDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_specs = [
            (1, 128, 15, 1, 7, 1),
            (128, 128, 41, 2, 20, 4),
            (128, 256, 41, 2, 20, 16),
            (256, 512, 41, 4, 20, 16),
            (512, 1024, 41, 4, 20, 16),
            (1024, 1024, 41, 1, 20, 16),
            (1024, 1024, 5, 1, 2, 1),
        ]
        for in_ch, out_ch, k_size, stride, pad, groups in layer_specs:
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k_size,
                    stride=stride,
                    padding=pad,
                    groups=groups,
                    use_spectral_norm=use_spectral_norm
                )
            )
        self.final_conv = init_conv(
            in_channels=1024, out_channels=1,
            kernel_size=3, stride=1, padding=1,
            use_spectral_norm=use_spectral_norm
        )

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        x = self.final_conv(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_maps


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SubDiscriminator(use_spectral_norm=True),
            SubDiscriminator(),
            SubDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, x):
        disc_outputs = []
        disc_features = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pools[i - 1](x)
            output, feats = disc(x)
            disc_outputs.append(output)
            disc_features.extend(feats)
        return disc_outputs, disc_features
