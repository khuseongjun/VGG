import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_dataloader(batch_size=128):
    print("Loading dataset...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=True,
        download=False,
        transform=None
    )

    indices = torch.randperm(len(full_trainset)).tolist()

    train_size = int(0.8 * len(full_trainset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=True,
        download=False,
        transform=transform_train
    )

    val_set = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=True,
        download=False,
        transform=transform_test
    )

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)

    testset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data',
        train=False,
        download=False,
        transform=transform_test
    )

    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Dataset loaded successfully")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(trainloader)}")
    print(f"Val batches: {len(valloader)}")
    print(f"Test batches: {len(testloader)}")

    return trainloader, valloader, testloader