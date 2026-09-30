"""
train_audio_cnn.py - Lightweight MFCC-CNN siren classifier (proposal 6.4)
<500K params, CPU-only. Spec augmentation during training (Section 6.1:
time shift, freq/time masking, additive noise). Early stopping on val loss.

Run:  python 03_Code/audio/train_audio_cnn.py
"""
from pathlib import Path
import json, copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "02_Data" / "splits" / "audio"
RES  = BASE / "04_Results" / "audio_cnn"
RES.mkdir(parents=True, exist_ok=True)

EPOCHS, BATCH, LR, PATIENCE = 25, 32, 1e-3, 5
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

def load(split):
    return np.load(DATA / f"X_{split}.npy"), np.load(DATA / f"y_{split}.npy")

Xtr, ytr = load("train"); Xva, yva = load("val")
mean, std = float(Xtr.mean()), float(Xtr.std())   # normalization constants

class MfccDS(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].copy().astype(np.float32)
        if self.augment:
            rng = np.random.default_rng(SEED + i)
            x = np.roll(x, rng.integers(-x.shape[1]//4, x.shape[1]//4), axis=1)
            for _ in range(2):                       # freq masking
                f = rng.integers(0, 4); r = rng.integers(0, x.shape[0]-f)
                x[r:r+f, :] = 0
            w = max(1, x.shape[1]//10)               # time masking
            t = rng.integers(0, x.shape[1]-w); x[:, t:t+w] = 0
            x = x + rng.normal(0, 0.01*np.std(x), x.shape).astype(np.float32)
        return torch.tensor((x - mean) / std).unsqueeze(0), int(self.y[i])

class SirenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

model = SirenCNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"SirenCNN parameters: {n_params:,} (target < 500,000)")

tr = DataLoader(MfccDS(Xtr, ytr, augment=True), batch_size=BATCH, shuffle=True)
va = DataLoader(MfccDS(Xva, yva), batch_size=BATCH)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()

best, best_state, bad = float("inf"), None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); tl = 0.0
    for xb, yb in tr:
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        tl += loss.item() * len(xb)
    model.eval(); vl, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in va:
            out = model(xb); vl += lossf(out, yb).item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item(); total += len(xb)
    tl, vl, acc = tl/len(Xtr), vl/len(Xva), correct/total
    flag = ""
    if vl < best: best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0; flag = " *best*"
    else: bad += 1
    print(f"Epoch {ep:2d}/{EPOCHS}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={acc:.4f}  ({time.time()-t0:.0f}s){flag}")
    if bad >= PATIENCE:
        print("Early stopping."); break

torch.save({"state": best_state, "mean": mean, "std": std,
            "classes": ["non-siren", "siren"]}, RES / "best.pt")
print(f"\nSaved: {RES / 'best.pt'}")
print("Next: python 03_Code/audio/evaluate_audio.py")
