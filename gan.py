import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

import torch
import torch.nn as nn

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
torch.set_default_device(device)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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


# Define hyperparameters
learning_rate = 0.0002
batch_size = 32
num_epochs = 100

# Create instances of the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Define loss function
criterion = nn.BCELoss()

# Define image transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


# Define custom dataset
class ImageDataset(Dataset):
    def __init__(
        self,
        dark_images_dir: Path | str,
        bright_images_dir: Path | str,
        transform: transforms.Compose | None = None,
    ):
        self.dark_images_dir = dark_images_dir
        self.bright_images_dir = bright_images_dir
        self.dark_image_files = os.listdir(dark_images_dir)
        self.bright_image_files = os.listdir(bright_images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dark_image_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        dark_image_path = os.path.join(self.dark_images_dir, self.dark_image_files[idx])
        bright_image_path = os.path.join(
            self.bright_images_dir, self.bright_image_files[idx]
        )

        dark_image = Image.open(dark_image_path).convert("RGB")
        bright_image = Image.open(bright_image_path).convert("RGB")

        if self.transform:
            dark_image = self.transform(dark_image)
            bright_image = self.transform(bright_image)

        return dark_image, bright_image


# Create dataset and dataloader
dark_images_dir = Path.cwd() / "data" / "lol_dataset" / "train485" / "low"
bright_images_dir = Path.cwd() / "data" / "lol_dataset" / "train485" / "high"
dataset = ImageDataset(dark_images_dir, bright_images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Training loop
for epoch in range(num_epochs):
    for i, (dark_images, bright_images) in enumerate(dataloader):
        dark_images = dark_images.to(device)
        bright_images = bright_images.to(device)

        # Train discriminator
        optimizer_d.zero_grad()

        # Real images
        real_output = discriminator(bright_images)
        loss_d_real = criterion(real_output, torch.ones_like(real_output))

        # Fake images
        fake_images = generator(dark_images)
        fake_output = discriminator(fake_images.detach())
        loss_d_fake = criterion(fake_output, torch.zeros_like(fake_output))

        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        output_g = discriminator(fake_images)
        loss_g = criterion(output_g, torch.ones_like(output_g))
        loss_g.backward()
        optimizer_g.step()

        # Print progress
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
            )

torch.save(
    generator.state_dict(),
    Path.cwd() / "weights" / "lol_dataset" / "generator_weights.pth",
)
torch.save(
    discriminator.state_dict(),
    Path.cwd() / "weights" / "lol_dataset" / "discriminator_weight.pth",
)
