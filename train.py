import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_and_eval(model, trainloader, testloader, device, epochs=5):
    print("[Step3] Training model...")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(epochs):
        print(f"- Train: Epoch {epoch+1}/{epochs} started", flush=True)
        
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    end = time.time()

    print("- Eval: Starting evaluation...")

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

    print(f"Accuracy: {acc:.2f}%")
    print(f"Training time: {train_time:.2f}s")

    return acc, train_time