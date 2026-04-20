import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(batch_size=128):
    print("[Step1] Loading dataset...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data', train=True, download=False, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='/data/seongjun0308/repos/VGG/data', train=False, download=False, transform=transform_test
    )

    print("Dataset loaded successfully")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )

    print("DataLoader created")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    return trainloader, testloader