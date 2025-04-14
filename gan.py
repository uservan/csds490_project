import argparse

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image
import os
from pathlib import Path

import torch
import torch.nn as nn

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
torch.set_default_device(device)


class Generator(nn.Module):
    def __init__(self) -> None:
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


class NewGenerator(nn.Module):
    def __init__(self) -> None:
        super(NewGenerator, self).__init__()
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


class Discriminator(nn.Module):
    def __init__(self) -> None:
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


DATA_PATH = Path.cwd() / "data"
WEIGHTS_PATH = Path.cwd() / "weights"
GENERATOR_WEIGHTS = "generator_weights.pth"
DISCRIMINATOR_WEIGHTS = "discriminator_weight.pth"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


BATCH_SIZE = 16

# Create instances of the generator and discriminator
# Replace Generator with NewGenerator below if you want to use it.
generator = NewGenerator()
discriminator = Discriminator()


def training(data_set_subdir: str | Path) -> None:
    # Define hyperparameters
    learning_rate = 0.0002
    num_epochs = 100

    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Define loss function
    criterion = nn.BCELoss()

    # Define image transformations

    # Create dataset and dataloader
    dark_images_dir = DATA_PATH / data_set_subdir / "train" / "low"
    bright_images_dir = DATA_PATH / data_set_subdir / "train" / "high"
    dataset = ImageDataset(
        dark_images_dir, bright_images_dir, transform=IMAGE_TRANSFORM
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

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
        WEIGHTS_PATH / data_set_subdir / GENERATOR_WEIGHTS,
    )
    torch.save(
        discriminator.state_dict(),
        WEIGHTS_PATH / data_set_subdir / DISCRIMINATOR_WEIGHTS,
    )


def testing(data_set_subdir: str | Path) -> None:
    generator.load_state_dict(
        torch.load(WEIGHTS_PATH / data_set_subdir / GENERATOR_WEIGHTS)
    )
    discriminator.load_state_dict(
        torch.load(WEIGHTS_PATH / data_set_subdir / DISCRIMINATOR_WEIGHTS)
    )

    # Create dataset and dataloader
    dark_images_dir = DATA_PATH / data_set_subdir / "test" / "low"
    bright_images_dir = DATA_PATH / data_set_subdir / "test" / "high"
    dataset = ImageDataset(
        dark_images_dir, bright_images_dir, transform=IMAGE_TRANSFORM
    )
    dataloader = DataLoader(dataset, batch_size=1)

    # Testing loop
    for i, (dark_images, bright_images) in enumerate(dataloader):
        dark_images = dark_images.to(device)
        bright_images = bright_images.to(device)

        generated_images = generator(dark_images)

        grid = make_grid(
            [
                dark_images[0],
                generated_images[0],
                bright_images[0],
            ],
            nrow=3,
        )
        save_image(grid, Path("outputs") / data_set_subdir / f"comparison_{i:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", choices=["lol", "ve_lol_cap", "ve_lol_synth"])

    args = parser.parse_args()

    data_set_dir: str | Path

    match args.dataset:
        case "lol":
            data_set_dir = "lol_dataset"
        case "ve_lol_cap":
            data_set_dir = Path("ve_lol_dataset") / "capture"
        case "ve_lol_synth":
            data_set_dir = Path("ve_lol_dataset") / "synthetic"
        case _:
            raise ValueError(
                f"{args.dataset} was not one of expected [lol, ve_lol_cap, ve_lol_synth]"
            )

    if args.train:
        training(data_set_dir)
    else:
        testing(data_set_dir)


if __name__ == "__main__":
    main()
