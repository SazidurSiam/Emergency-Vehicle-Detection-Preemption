"""
evaluate.py - Stage A evaluation (proposal Section 6.6)
Outputs: mAP@0.5, precision, recall, F1 -> 04_Results/metrics/stageA_metrics.csv
Plus sample detection images in 04_Results/figures/
"""
from pathlib import Path
import csv
from ultralytics import YOLO

BASE    = Path(__file__).resolve().parents[2]
DATA    = BASE / "02_Data" / "splits" / "data.yaml"
WEIGHTS = BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt"
MET_DIR = BASE / "04_Results" / "metrics"
FIG_DIR = BASE / "04_Results" / "figures"

def main():
    MET_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(DATA), split="test", plots=True,
                  project=str(MET_DIR), name="val_test")
    p, r, map50 = float(m.box.mp), float(m.box.mr), float(m.box.map50)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    with open(MET_DIR / "stageA_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["precision", round(p, 4)])
        w.writerow(["recall", round(r, 4)])
        w.writerow(["F1", round(f1, 4)])
        w.writerow(["mAP50", round(map50, 4)])
    print(f"\nPrecision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}  mAP50={map50:.4f}")
    print(f"Saved: {MET_DIR / 'stageA_metrics.csv'}")
    test_imgs = list((BASE / "02_Data" / "splits" / "test" / "images").glob("*"))[:8]
    if test_imgs:
        model.predict(source=[str(x) for x in test_imgs], save=True,
                      project=str(FIG_DIR), name="sample_detections", exist_ok=True)
        print(f"Sample detections: {FIG_DIR / 'sample_detections'}")

if __name__ == "__main__":
    main()
