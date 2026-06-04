"""
CIFAR-10 data loaders shared by all stages.

Transforms match the original scripts exactly: random crop + horizontal flip
for training, plain normalisation for testing.
"""
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model import MEAN, STD


def _train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def _test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def get_loaders(data_dir, batch_size, num_workers, test_batch_size=256):
    """Train + test DataLoaders for full training / fine-tuning."""
    trainset = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=_train_transform())
    testset = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=_test_transform())

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0)
    test_loader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0)
    return train_loader, test_loader


def get_test_loader(data_dir, batch_size=256):
    """Test loader only (used by evaluate.py for accuracy)."""
    testset = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=_test_transform())
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_calibration_loader(data_dir, n_batches, batch_size=32):
    """A small subset of the training set, used to calibrate PTQ."""
    trainset = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=_test_transform())
    subset = Subset(trainset, range(n_batches * batch_size))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
