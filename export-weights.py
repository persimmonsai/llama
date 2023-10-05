import torch
import numpy as np
from pathlib import Path
import sys
import json
import pickle

ckpt_dir = sys.argv[1]
out_dir = Path(ckpt_dir).expanduser() / "weights"
if out_dir.exists():
   print("directory already exists")
   sys.exit(0)

checkpoints = sorted(list(Path(ckpt_dir).glob('*.pth'))+list(Path(ckpt_dir).glob('*.pt')))
if len(checkpoints) > 1:
    print("too many checkpoint files")
    sys.exit()

print("Converting checkpoint to numpy weights")
with open(Path(ckpt_dir) / "params.json", "r") as f:
    params = json.loads(f.read())

for ckpt_file in checkpoints:
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    if 'model' in checkpoint: checkpoint = checkpoint['model']

    # split weights by head
    for layer in range(params['n_layers']):
        for wkey in ['wq', 'wk', 'wv']:
            key = f'layers.{layer}.attention.{wkey}.weight'
            wq_weight = torch.chunk(checkpoint[key], chunks=params['n_heads'], dim=0)

            del checkpoint[key]
            for i, weight in enumerate(wq_weight, start=0):
                checkpoint[f'layers.{layer}.attention.{wkey}.{i}.weight'] = weight

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in checkpoint.items():
        v = v.to(torch.float32).numpy()
        np.save(out_dir / ( k + '.npy'), v)
        with open (out_dir / ( k + '.pkl'), 'wb') as f:
            pickle.dump(v.tolist(), f)
    checkpoint = None
print(f"Weights stored in {out_dir}")
