"""
train_yolo.py - Stage A baseline: YOLOv8-nano, CPU-only (proposal Section 6.3)
Run AFTER prep_data.py.

First test run : EPOCHS = 20  (confirms pipeline works, ~30-60 min on CPU)
Thesis run     : EPOCHS = 100
"""
from pathlib import Path
from ultralytics import YOLO

BASE   = Path(__file__).resolve().parents[2]
DATA   = BASE / "02_Data" / "splits" / "data.yaml"

EPOCHS = 20        # raise to 100 for the final Stage-A model
IMGSZ  = 640       # proposal Section 6.2
BATCH  = 16
DEVICE = "cpu"     # CPU-only is the whole point of your thesis

def main():
    if not DATA.exists():
        raise SystemExit("data.yaml not found - run prep_data.py first.")
    model = YOLO("yolov8n.pt")   # 3.2M params, pretrained on COCO
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=2,
        project=str(BASE / "04_Results"),
        name="stageA_yolov8n",
        exist_ok=True,
        pretrained=True,
        plots=True,
    )
    print("\nDone. Best weights:")
    print(BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt")

if __name__ == "__main__":
    main()
