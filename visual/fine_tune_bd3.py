"""fine_tune_bd3.py - Plan B: BD-only fine-tune, lr fix applied (optimizer='Adam')
Run:  python 03_Code/visual/fine_tune_bd3.py   (~45-60 min)"""
from pathlib import Path
from ultralytics import YOLO
BASE  = Path(__file__).resolve().parents[2]
DATA  = BASE / "02_Data" / "visual" / "local_bd" / "merged3" / "data.yaml"
START = BASE / "04_Results" / "stageA_yolov8n" / "weights" / "best.pt"
model = YOLO(str(START))
model.train(data=str(DATA), epochs=15, imgsz=640, batch=16, device="cpu",
            workers=2, optimizer="Adam", lr0=1e-4, freeze=10,
            project=str(BASE / "04_Results"), name="stageB_bd3",
            exist_ok=True, pretrained=True, plots=True)
print("\nStage B (BD-only) saved:", BASE / "04_Results" / "stageB_bd3" / "weights" / "best.pt")
