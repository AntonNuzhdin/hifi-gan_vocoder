import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period, in_channels=1, base_channels=32, num_blocks=4, kernel_size=5, stride=2):
        super().__init__()
        self.period = period
        self.blocks = nn.ModuleList()
        channels = in_channels
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            channels, base_channels, kernel_size=(kernel_size, 1),
                            stride=(stride, 1),
                            padding=(kernel_size // 2, 0))),
                    nn.LeakyReLU(0.1)))
            channels = base_channels
            base_channels *= 4
        self.final_conv = weight_norm(nn.Conv2d(channels, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x):
        # Reshape 1D -> 2D (batch_size, 1, T // period, period)
        batch_size, channels, length = x.shape
        if length % self.period != 0:
            pad_size = self.period - (length % self.period)
            x = F.pad(x, (0, pad_size), mode="reflect")
        x = x.view(batch_size, channels, -1, self.period)
        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)
        output = self.final_conv(x)
        feature_maps.append(output)
        return output, feature_maps


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11], in_channels=1):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([PeriodSubDiscriminator(p, in_channels) for p in periods])

    def forward(self, x):
        outputs = []
        all_feature_maps = []
        for sub_discriminator in self.sub_discriminators:
            output, feature_maps = sub_discriminator(x)
            outputs.append(output)
            all_feature_maps.extend(feature_maps)
        return outputs, all_feature_maps
