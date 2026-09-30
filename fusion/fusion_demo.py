"""
fusion_demo.py - The BRAIN: weighted-voting fusion (proposal Section 6.5)
--------------------------------------------------------------------------
P_fused = wv * Pv + wa * Pa,  wv + wa = 1,  tau = 0.85

  Pv = visual confidence  (YOLOv8n: highest emergency-vehicle box confidence,
                           0 if nothing detected)
  Pa = audio confidence   (SirenCNN: softmax 'siren' probability over a 4s clip)

Usage (thesis env active, from repo root):
  python 03_Code/fusion/fusion_demo.py --image 02_Data/splits/test/images/xxx.jpg
       --audio some_4s_clip.wav
  optional: --wv 0.5 --wa 0.5 --tau 0.85

Outputs: annotated image (boxes + all 3 scores) in 04_Results/figures/
"""
from pathlib import Path
import argparse
import numpy as np
import librosa, torch, torch.nn as nn, cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[2]
FIG  = BASE / "04_Results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--wv", type=float, default=0.5)
ap.add_argument("--wa", type=float, default=0.5)
ap.add_argument("--tau", type=float, default=0.85)
args = ap.parse_args()
assert abs(args.wv + args.wa - 1.0) < 1e-6, "wv + wa must equal 1"

# ---- 1. visual pipeline -----------------------------------------------------
yolo = YOLO(str(BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt"))
res = yolo.predict(source=args.image, verbose=False)[0]
Pv = 0.0
for b in res.boxes:
    Pv = max(Pv, float(b.conf))
img = cv2.imread(args.image)

# ---- 2. audio pipeline ------------------------------------------------------
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
y, sr = librosa.load(args.audio, sr=22050, duration=4.0, mono=True)
y = np.pad(y, (0, max(0, 88200 - len(y))))[:88200]
m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
with torch.no_grad():
    out = model(torch.tensor(((m - mean) / std)).unsqueeze(0).unsqueeze(0).float())
Pa = float(torch.softmax(out, 1)[0, 1])

# ---- 3. fusion + decision ---------------------------------------------------
P_fused = args.wv * Pv + args.wa * Pa
decision = "EMERGENCY VEHICLE DETECTED -> PREEMPT SIGNAL" if P_fused >= args.tau else "normal traffic operation"

annot = res.plot()  # YOLO's own annotated frame
annot = cv2.cvtColor(annot, cv2.COLOR_RGB2BGR)
for i, txt in enumerate([f"Pv (visual)   = {Pv:.3f}",
                         f"Pa (audio)    = {Pa:.3f}",
                         f"P_fused       = {P_fused:.3f}  (wv={args.wv}, wa={args.wa})",
                         f"Decision: {decision}"]):
    cv2.putText(annot, txt, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if P_fused >= args.tau else (0, 0, 255), 2)
out_path = FIG / "fusion_demo_output.jpg"
cv2.imwrite(str(out_path), annot)

print(f"\nPv      = {Pv:.4f}")
print(f"Pa      = {Pa:.4f}")
print(f"P_fused = {P_fused:.4f}   (tau = {args.tau})")
print(f"DECISION: {decision}")
print(f"\nAnnotated image saved: {out_path}")
