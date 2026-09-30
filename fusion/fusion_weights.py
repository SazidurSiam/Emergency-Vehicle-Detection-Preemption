"""
fusion_weights.py - Weight sweep + threshold analysis (proposal 6.5)
Grid-searches wv in [0..1] and tau, on PAIRED test samples.

IMPORTANT (thesis honesty note): samples are paired index-wise (image i with
audio i) from INDEPENDENT test sets. This validates the fusion MECHANICS and
shows the decision-surface behavior. The REAL fusion gain experiment needs
synchronized captures - your BD street videos (frames + audio track) in Stage B.
"""
from pathlib import Path
import numpy as np, json, librosa, torch, torch.nn as nn, csv
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "02_Data" / "splits"
MET  = BASE / "04_Results" / "metrics"
MET.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

# ---- collect Pv per test image (1 = image contains emergency vehicle) -------
yolo = YOLO(str(BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt"))
imgs = sorted((DATA / "test" / "images").glob("*"))
Pv_list, yv_list = [], []
for im in imgs:
    r = yolo.predict(source=str(im), verbose=False)[0]
    Pv_list.append(max([float(b.conf) for b in r.boxes], default=0.0))
    yv_list.append(1)   # all test images contain emergency vehicles
Pv = np.array(Pv_list); yv = np.array(yv_list)

# ---- collect Pa per test audio clip ----------------------------------------
ckpt = torch.load(BASE / "04_Results" / "audio_cnn" / "best.pt", weights_only=False)
mean, std = ckpt["mean"], ckpt["std"]
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
model = SirenCNN(); model.load_state_dict(ckpt["state"]); model.eval()
Xa = (np.load(DATA / "audio" / "X_test.npy") - mean) / std
ya = np.load(DATA / "audio" / "y_test.npy")
with torch.no_grad():
    Pa = torch.softmax(model(torch.tensor(Xa).unsqueeze(1).float()), 1)[:, 1].numpy()

# ---- pair min length ---------------------------------------------------------
n = min(len(Pv), len(Pa)); Pv, yv, Pa, ya = Pv[:n], yv[:n], Pa[:n], ya[:n]
print(f"Paired {n} samples (visual test + audio test, index-wise)")
print(f"Visual-alone acc on these: {accuracy_score(yv, Pv >= 0.5):.4f}")
print(f"Audio-alone  acc on these: {accuracy_score(ya, Pa >= 0.5):.4f}")

# ---- weight sweep ------------------------------------------------------------
rows = []
for wv10 in range(0, 11):
    wv = wv10 / 10; wa = 1 - wv
    Pf = wv * Pv + wa * Pa
    yhat = ((Pf >= 0.85) & (yv == 1) | (Pa >= 0.5)).astype(int)  # fused flag vs visual gt
    rows.append({"wv": wv, "wa": wa,
                 "fused_flag_rate": float(np.mean(Pf >= 0.85))})
    print(f"wv={wv:.1f} wa={wa:.1f}  -> P(fused >= tau) = {np.mean(Pf >= 0.85):.3f}")

# ---- threshold sweep at wv=wa=0.5 -------------------------------------------
Pf = 0.5 * Pv + 0.5 * Pa
print("\nThreshold sweep (wv=wa=0.5):")
thr_rows = []
for tau in np.arange(0.50, 0.96, 0.05):
    thr_rows.append({"tau": round(float(tau), 2),
                     "flag_rate": float(np.mean(Pf >= tau))})
    print(f"  tau={tau:.2f}  P(fused >= tau) = {np.mean(Pf >= tau):.3f}")

with open(MET / "fusion_weight_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["wv", "wa", "fused_flag_rate"]); w.writeheader(); w.writerows(rows)
with open(MET / "fusion_threshold_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tau", "flag_rate"]); w.writeheader(); w.writerows(thr_rows)
print(f"\nSaved: {MET / 'fusion_weight_sweep.csv'} and fusion_threshold_sweep.csv")
print("NOTE: real synchronized fusion evaluation = Stage B (BD videos with audio).")
