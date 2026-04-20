from dataloader import get_dataloader
from model import VGG
from train import train_and_eval
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[DEVICE]", device)

trainloader, testloader = get_dataloader()

channel_configs = [
    [32, 64, 128, 256, 512],
    [64, 128, 256, 512, 512],
    [64, 128, 256, 512, 1024]
]

pool_strides = [1, 2]

results = []

for idx, ch in enumerate(channel_configs):
    for ps in pool_strides:

        print("\n" + "=" * 60)
        print(f"[RUN {idx+1}] channels={ch}, stride={ps}")
        print("=" * 60)

        print("[STEP 2] Creating model...")
        model = VGG(channels=ch, pool_stride=ps)
        model = model.to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {params}")

        acc, t = train_and_eval(model, trainloader, testloader, device, 20)

        results.append({
            "channels": str(ch),
            "pool_stride": ps,
            "accuracy": acc,
            "params": params,
            "time": t
        })

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("===== ALL DONE =====")