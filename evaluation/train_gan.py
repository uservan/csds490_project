import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from common import (
    DISCRIMINATOR_MAPPING,
    DISCRIMINATOR_WEIGHTS,
    GENERATOR_MAPPING,
    GENERATOR_WEIGHTS,
    WEIGHTS_PATH,
    DataSets,
    GroundTruthDataSets,
    NoGroundTruthDataSets,
    device,
)

# Import to have import here after common Tensor be on cuda
from custom_datasets import GroundTruthImageDataset, NoGroundTruthImageDataset
from loss import unsupervised_loss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

BATCH_SIZE = 8


def ground_truth_training(dataset_name: GroundTruthDataSets) -> None:
    generator = GENERATOR_MAPPING[dataset_name]
    discriminator = DISCRIMINATOR_MAPPING[dataset_name]

    # Define hyperparameters
    learning_rate = 0.0002
    num_epochs = 100

    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create dataset and dataloader
    dataset = GroundTruthImageDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Training loop
    print("Starting")
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
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
        )

    torch.save(
        generator.state_dict(),
        WEIGHTS_PATH / dataset_name / GENERATOR_WEIGHTS,
    )
    torch.save(
        discriminator.state_dict(),
        WEIGHTS_PATH / dataset_name / DISCRIMINATOR_WEIGHTS,
    )


def ground_truth_testing(dataset_name: GroundTruthDataSets, algorithm_name: DataSets) -> None:
    generator = GENERATOR_MAPPING[algorithm_name]
    generator.load_state_dict(
        torch.load(WEIGHTS_PATH / algorithm_name / GENERATOR_WEIGHTS)
    )

    # Create dataset and dataloader
    dataset = GroundTruthImageDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=1)

    # Testing loop
    for i, (dark_images, bright_images) in enumerate(dataloader):
        if i >= 20:
            break

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
        save_image(
            grid,
            Path.cwd().parent / ("gan_outputs") / f"trained_on_{algorithm_name}" / dataset_name / f"comparison_{i:03d}.png",
        )
        save_image(
            generated_images[0],
            Path.cwd().parent / ("gan_outputs") / f"trained_on_{algorithm_name}" / dataset_name / f"{i:03d}.png",
        )


def no_ground_truth_training(dataset_name: NoGroundTruthDataSets):
    generator = GENERATOR_MAPPING[dataset_name]
    discriminator = DISCRIMINATOR_MAPPING[dataset_name]

    # Define hyperparameters
    learning_rate = 0.0002
    num_epochs = 100

    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Define loss function
    criterion = unsupervised_loss

    # Create dataset and dataloader
    dataset = NoGroundTruthImageDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=4)

    # Training loop
    print("Starting")
    for epoch in range(num_epochs):
        for i, dark_images in enumerate(dataloader):
            dark_images = dark_images.to(device)

            # Train discriminator
            optimizer_d.zero_grad()

            # Real images
            # real_output = discriminator(bright_images)
            # loss_d_real = criterion(real_output, torch.ones_like(real_output))

            # Fake images
            fake_images = generator(dark_images)
            fake_output = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_output)

            # Total discriminator loss
            # loss_d = loss_d_real + loss_d_fake
            loss_d = loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            output_g = discriminator(fake_images)
            loss_g = criterion(output_g)
            loss_g.backward()
            optimizer_g.step()

            if (i % 100) == 0:
                # Print progress
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
                )

    torch.save(
        generator.state_dict(),
        WEIGHTS_PATH / dataset_name / GENERATOR_WEIGHTS,
    )
    torch.save(
        discriminator.state_dict(),
        WEIGHTS_PATH / dataset_name / DISCRIMINATOR_WEIGHTS,
    )


def no_ground_truth_testing(dataset_name: NoGroundTruthDataSets, algorithm_name: DataSets):
    generator = GENERATOR_MAPPING[algorithm_name]
    generator.load_state_dict(
        torch.load(WEIGHTS_PATH / algorithm_name / GENERATOR_WEIGHTS)
    )

    # Create dataset and dataloader
    dataset = NoGroundTruthImageDataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=1)

    # Testing loop
    for i, dark_images in enumerate(dataloader):
        if i >= 20:
            break

        dark_images = dark_images.to(device)

        generated_images = generator(dark_images)

        grid = make_grid(
            [
                dark_images[0],
                generated_images[0],
            ],
            nrow=2,
        )
        save_image(
            generated_images[0],
            Path.cwd().parent / ("gan_outputs") / f"trained_on_{algorithm_name}" / dataset_name / f"{i:03d}.png",
        )
        save_image(
            grid,
            Path.cwd().parent / ("gan_outputs") / f"trained_on_{algorithm_name}" / dataset_name / f"comparison_{i:03d}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dataset", choices=["lol", "lolv2", "dark_face"])

    args = parser.parse_args()

    match args.dataset:
        case "lol":
            dataset_name: DataSets = "lol_dataset"
            ground_truth = True
        case "lolv2":
            dataset_name = "LOL-v2"
            ground_truth = True
        case "dark_face":
            dataset_name = "Dark_Face"
            ground_truth = False
        case _:
            raise ValueError(
                f"{args.dataset} was not one of expected [lol, ve_lol_cap, ve_lol_synth]"
            )

    algorithm = "lol_dataset"
    # algorithm = "LOL-v2"

    if args.train:
        if ground_truth:
            ground_truth_training(dataset_name)
        else:
            no_ground_truth_training(dataset_name)
    else:
        if ground_truth:
            ground_truth_testing(dataset_name, algorithm)
        else:
            no_ground_truth_testing(dataset_name, algorithm)


if __name__ == "__main__":
    main()
