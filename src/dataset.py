import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.utils.data import default_collate
from torchvision.transforms import v2

import tqdm


def get_dataloaders(data_dir, batch_size=64, cutmixup=False):
    """
    1. Load the dataset
    2. Apply transforms, resize, flip, rotate, scale, normalize
    3. Split the dataset into train and val
    4. Create a dataloader for the train and val
    5. Return the train and val dataloaders
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # Horizontal Flip
            transforms.RandomHorizontalFlip(),
            # Rotation
            transforms.RandomRotation(20),
            # Scaling
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            # Normalize around mean and std for better convergence
            transforms.Normalize(
                mean=[0.4547, 0.4337, 0.4011], std=[0.2266, 0.2237, 0.2316]
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4547, 0.4337, 0.4011], std=[0.2266, 0.2237, 0.2316]
            ),
        ]
    )

    # Load the full dataset to get the size
    full_dataset = ImageFolder(root=data_dir)
    class_names = full_dataset.classes

    # Split indices only (so you can reuse them)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size]
    )

    # Now create two separate datasets with their own transforms
    train_dataset = ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=data_dir, transform=val_transforms)

    # Subset them using the same indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cutmixup_da if cutmixup else None,
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=cutmixup_da if cutmixup else None
    )

    return train_loader, test_loader, class_names


def cutmixup_da(batch, num_classes=40):
    """
    Implements cutmix or mixup data augmentation
    """
    cutmix = v2.CutMix(alpha=0.6, num_classes=num_classes)
    mixup = v2.MixUp(alpha=0.2, num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup(*default_collate(batch))


def calculate_mean_std(dataset):
    """
    Compute mean and std, useful for normalizing
    e.g. calculate_mean_std(full_dataset)
    usage:
        transforms.Normalize(
            mean=[0.4547, 0.4337, 0.4011],
            std=[0.2266, 0.2237, 0.2316]
        )
    """
    means = []
    stds = []

    for img, _ in tqdm(dataset, desc="Computing mean/std"):
        means.append(torch.mean(img, dim=(1, 2)))  # mean per channel
        stds.append(torch.std(img, dim=(1, 2)))  # std per channel

    means = torch.stack(means).mean(0)
    stds = torch.stack(stds).mean(0)

    print(f"Mean per channel(R, G, B): {means}")
    print(f"Std per channel(R, G, B): {stds}")
