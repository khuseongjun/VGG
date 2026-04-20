import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloader(batch_size=128):
    print("[Step0] Loading dataset...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=True,
        download=False,
        transform=None
    )

    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size

    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    train_subset.dataset.transform = transform_train
    val_subset.dataset.transform = transform_test

    testset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=False,
        download=False,
        transform=transform_test
    )

    print("Dataset loaded successfully")

    trainloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    valloader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    print("DataLoader created")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(trainloader)}")
    print(f"Val batches: {len(valloader)}")
    print(f"Test batches: {len(testloader)}")

    return trainloader, valloader, testloader