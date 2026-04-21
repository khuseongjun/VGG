import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_and_eval(model, trainloader, valloader, testloader, device, epochs=5):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(trainloader),
        epochs=epochs
    )

    start = time.time()

    for epoch in range(epochs):
        print(f"- epoch: {epoch}/{epochs}")
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(valloader)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    end = time.time()

    print("- Eval")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    train_time = end - start

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Training Time: {train_time:.2f}s")

    return acc, train_time