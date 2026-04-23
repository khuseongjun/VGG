import torch
from dataloader import get_dataloader
from model import VGG
from train import train_and_eval
import pandas as pd
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEVICE]", device)

channels_config = [64, 128, 256, 512, 512]
stride_config = 2
crop_configs = 32

results = []
    
trainloader, valloader, testloader = get_dataloader(crop_size=crop_configs)
    
model = VGG(channels=channels_config, pool_stride=stride_config).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"Params: {params}")

acc, t = train_and_eval(
    model,
    trainloader,
    valloader,
    testloader,
    device,
    epochs=20
)

results.append({
    "cutmix": 1.0,
    "accuracy": acc,
    "params": params,
    "time": t
})

df = pd.DataFrame(results)
df.to_csv("cutmix_add.csv", index=False)

print("===== ALL DONE =====")