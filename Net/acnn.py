"""Auto Encoder"""
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.down_size = tuple([d // (2 ** 3) for d in input_size])

        self.encoder_cnn = nn.Sequential(
            Conv(in_channels, 16, stride=2),
            nn.ReLU(),
            Conv(16, 16),
            nn.ReLU(),
            Conv(16, 32, stride=2),
            nn.ReLU(),
            Conv(32, 32),
            nn.ReLU(),
            Conv(32, 1, stride=2),
            nn.ReLU(),
        )

        self.encoder_linear = nn.Linear(self.down_size[0]**2, 32)

        self.decoder_linear = nn.Sequential(
            nn.Linear(32, self.down_size[0]**2),
            nn.ReLU()
        )

        self.decoder_cnn = nn.Sequential(
            UpConv(1, 32),
            nn.ReLU(),
            Conv(32, 32),
            nn.ReLU(),
            UpConv(32, 16),
            nn.ReLU(),
            Conv(16, 16),
            nn.ReLU(),
            UpConv(16, 16),
            nn.ReLU(),
            Conv(16, out_channels, use_batchnorm=False),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_linear(x.view(-1, self.down_size[0]**2))
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.decoder_cnn(x.view(-1, 1, *self.down_size))
        return x


class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm)
        )

    def forward(self, x):
        x = self.block(x)
        return x
