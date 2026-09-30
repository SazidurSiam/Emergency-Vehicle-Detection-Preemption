"""
fine_tune_bd2.py - CONTROLLED Stage B fine-tune (attempt 2)
- freezes backbone (first 10 layers) -> keeps Stage-A COCO features
- low learning rate (1e-4) -> no catastrophic forgetting
Run:  python 03_Code/visual/fine_tune_bd2.py    (~4-6 hrs overnight)
"""
from pathlib import Path
from ultralytics import YOLO

BASE  = Path(__file__).resolve().parents[2]
DATA  = BASE / "02_Data" / "visual" / "local_bd" / "merged2" / "data.yaml"
START = BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt"

model = YOLO(str(START))
model.train(data=str(DATA), epochs=10, imgsz=640, batch=16, device="cpu",
            workers=2, lr0=1e-4, freeze=10,
            project=str(BASE / "04_Results"), name="stageB_bd2",
            exist_ok=True, pretrained=True, plots=True)
print("\nStage B (v2) model saved:")
print(BASE / "04_Results" / "stageB_bd2" / "weights" / "best.pt")
print("Morning: python 03_Code/visual/eval_stageb2.py")
