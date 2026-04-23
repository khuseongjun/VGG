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

crop_configs = [
    20, 24, 28, 32, 36
]

results = []

run_id = 1

for cs in crop_configs:

    print("\n" + "=" * 60)
    print(f"[RUN {run_id}] crop_size={cs}")
    print("=" * 60)

    run_id += 1
    
    trainloader, valloader, testloader = get_dataloader(crop_size=cs)
    
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
        "crop_size": cs,
        "accuracy": acc,
        "params": params,
        "time": t
    })

df = pd.DataFrame(results)
df.to_csv("crop_size.csv", index=False)

print("===== ALL DONE =====")