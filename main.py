from dataloader import get_dataloader
from model import VGG
from train import train_and_eval
import pandas as pd

trainloader, testloader = get_dataloader()

channel_configs = [
    [32, 64, 128, 256, 512],
    [32, 128, 256, 512, 512],
    [64, 128, 256, 512, 512],
    [64, 128, 256, 512, 1024]
]

pool_strides = [1, 2]

results = []

for ch in channel_configs:
    for ps in pool_strides:
        print(f"Running: channels={ch}, stride={ps}")

        model = VGG(channels=ch, pool_stride=ps)

        params = sum(p.numel() for p in model.parameters())

        acc, t = train_and_eval(model, trainloader, testloader)

        results.append({
            "channels": ch,
            "pool_stride": ps,
            "accuracy": acc,
            "params": params,
            "time": t
        })

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)