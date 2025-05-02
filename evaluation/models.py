import torch
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
            nn.InstanceNorm2d(channels),
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
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),  # Output normalized to [-1, 1]
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
            nn.Conv2d(
                3, 64, kernel_size=4, stride=2, padding=1
            ),  # -> [B, 64, H/2, W/2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # -> [B, 128, H/4, W/4]
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),  # -> [B, 256, H/8, W/8]
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                256, 512, kernel_size=4, stride=1, padding=1
            ),  # -> [B, 512, H/8 - 1, W/8 - 1]
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                512, 1, kernel_size=4, stride=1, padding=1
            ),  # -> [B, 1, H/8 - 2, W/8 - 2]
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


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super(UNetBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class lolv2Generator(nn.Module):
    def __init__(self):
        super(lolv2Generator, self).__init__()

        # Encoder
        self.enc1 = UNetBlock(3, 64, use_norm=False)
        self.enc2 = UNetBlock(64, 128)
        self.pool1 = nn.MaxPool2d(2)

        self.enc3 = UNetBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2)

        self.enc4 = UNetBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(512, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Final conv
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()  # output in [-1, 1]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.tanh(self.final_conv(d1))


class lolv2Discriminator(nn.Module):
    def __init__(self):
        super(lolv2Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # No norm in first layer
            nn.LeakyReLU(0.2, inplace=True),
            discriminator_block(64, 128, 2),
            discriminator_block(128, 256, 2),
            discriminator_block(256, 512, 1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),  # Output PatchGAN map
        )

    def forward(self, x):
        return self.model(x)  # Output shape: [B, 1, H/16 - 1, W/16 - 1]
