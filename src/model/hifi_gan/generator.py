import torch.nn as nn
from typing import List


class ResBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        dilations: List[List[int]]
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding="same"
                        )
                    )
                )
                for dilation in dilation_group
            ])
            for dilation_group in dilations
        ])

    def forward(self, x):
        for block in self.blocks:
            residual = x
            for layer in block:
                x = layer(x)
            x = x + residual
        return x


class MRF(nn.Module):
    def __init__(
        self,
        channels,
        kr,
        Dr
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, kr[n], Dr[n])
            for n in range(len(kr))  # dilation_group [[], [], []]
        ])

    def forward(self, x):
        out = 0
        for block in self.res_blocks:
            out += block(x)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hu: int,
        ku: List[int],
        kr: List[int],
        Dr: List[List[List[int]]]
    ):
        super().__init__()
        self.preprocess = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hu,
                kernel_size=7,
                padding="same"
            )
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.ConvTranspose1d(
                    in_channels=hu // (2 ** i),
                    out_channels=hu // (2 ** (i + 1)),
                    kernel_size=ku[i],
                    stride=ku[i] // 2,
                    padding=ku[i] // 4
                ),
                MRF(
                    hu // (2 ** (i + 1)),
                    kr,
                    Dr
                )
            )
            for i in range(len(ku))
        ])
        self.postprocess = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=hu // (2 ** len(ku)),
                out_channels=1,
                kernel_size=7,
                padding="same"
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        x = self.postprocess(x)
        return x
