import torch.nn as nn


# Residual Block used in the lolGenerator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# Generator for Low-Light Image Enhancement (LOL dataset)
class lolGenerator(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(lolGenerator, self).__init__()

        # Initial conv block
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        x = self.output_layer(x)
        return x



class lolDiscriminator(nn.Module):
    def __init__(self):
        super(lolDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # input: [B, 3, H, W]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, H/2, W/2]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, H/4, W/4]
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [B, 256, H/8, W/8]
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # -> [B, 512, H/8 - 1, W/8 - 1]
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # -> [B, 1, H/8 - 2, W/8 - 2]
            # No sigmoid here — use BCEWithLogitsLoss instead
        )

    def forward(self, x):
        return self.model(x)



class dark_faceGenerator(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1),  # [B, 3, H, W] -> [B, 64, H/2, W/2]
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class dark_faceDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(dark_faceDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),  # or use BCEWithLogitsLoss without sigmoid
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


class lolv2Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self.contract_block(input_channels, features),
            self.contract_block(features, features * 2),
            self.contract_block(features * 2, features * 4),
            self.contract_block(features * 4, features * 8),
        )
        self.decoder = nn.Sequential(
            self.expand_block(features * 8, features * 4),
            self.expand_block(features * 4, features * 2),
            self.expand_block(features * 2, features),
            nn.ConvTranspose2d(
                features, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def contract_block(
        self, in_channels, out_channels, kernel_size=4, stride=2, padding=1
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def expand_block(
        self, in_channels, out_channels, kernel_size=4, stride=2, padding=1
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        # Ensure output matches input size
        out = nn.functional.interpolate(
            out, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return out


class lolv2Discriminator(nn.Module):
    def __init__(self, input_channels=3, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                input_channels, features, kernel_size=4, stride=2, padding=1
            ),  # no batchnorm here
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),  # Optional, can use BCEWithLogitsLoss instead
        )

    def forward(self, x):
        return self.model(x)
