import os
import sys
import time
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from custom.filtered_leaky_relu import FilteredLeakyReLU
from custom.filtered_downsampler import FilteredDownsampler

SEED = 420
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
BATCH_SIZE = 64 if DEVICE == 'cuda' else 32


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            FilteredLeakyReLU(
                sampling_rate=28,
                cutoff=10,
                half_width=4,
            ),
            FilteredDownsampler(
                sampling_rate=28,
                cutoff=7,
                half_width=2,
                down_factor=2
            )
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            FilteredLeakyReLU(
                sampling_rate=14,
                cutoff=7,
                half_width=2,
            ),
            FilteredDownsampler(
                sampling_rate=14,
                cutoff=4,
                half_width=1,
                down_factor=2
            )
        )
        self.head = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.reshape(x.size(0), -1)
        output = self.head(x)
        return output


def test_forward_backward():
    model = ConvNet()
    model.to(DEVICE)

    x = torch.zeros((32, 1, 28, 28), dtype=torch.float32)
    x = x.to(DEVICE)

    out = model(x)
    out.mean().backward()

    assert True


@pytest.mark.training_test
def test_validation_accuracy():
    val_accuracy = run_training_and_validation(BATCH_SIZE, NUM_EPOCHS, DEVICE, SEED)
    assert val_accuracy >= 0.985


def run_training_and_validation(batch_size: int, num_epochs: int, device: str, seed: int):
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset_dir = os.path.expanduser(os.path.join('.cache', 'mnist'))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        download = True
    else:
        download = False

    train_dataset = datasets.MNIST(dataset_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(dataset_dir, train=False, download=download, transform=transform)

    if device == 'cuda':
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True}
    else:
        cuda_kwargs = {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **cuda_kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, **cuda_kwargs)

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    val_accuracy = 0.
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_epoch_loop(model, device, train_loader, optimizer, scheduler)
        val_accuracy = validation_loop(model, device, test_loader)
        scheduler.step()

    return val_accuracy


def train_epoch_loop(model: nn.Module, device: str, train_loader, optimizer, scheduler):
    model.train()
    sys.stdout.flush()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    scheduler.step()


def validation_loop(model: nn.Module, device: str, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return accuracy
