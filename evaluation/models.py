import torch.nn as nn


class lolGenerator(nn.Module):
    def __init__(self) -> None:
        super(lolGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class lolDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(lolDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


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
