"""eval_stageb3.py - Stage A vs StageB_bd3 on the SAME 34 BD-only test images"""
from pathlib import Path
import csv
from ultralytics import YOLO
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "02_Data" / "visual" / "local_bd" / "merged2" / "data.yaml"
MET  = BASE / "04_Results" / "metrics"; MET.mkdir(parents=True, exist_ok=True)
models = [("StageA (Western-only)", "stageA", BASE/"04_Results"/"stageA_yolov8n"/"weights"/"best.pt"),
          ("StageB-v3 (BD-only)", "stageBv3", BASE/"04_Results"/"stageB_bd3"/"weights"/"best.pt")]
rows = []
for name, slug, w in models:
    if not w.exists(): print("SKIP missing:", w); continue
    r = YOLO(str(w)).val(data=str(DATA), split="test", plots=True, project=str(MET), name=f"stageb3_val_{slug}")
    row = {"model": name, "precision": round(float(r.box.mp),4), "recall": round(float(r.box.mr),4), "mAP50": round(float(r.box.map50),4)}
    rows.append(row); print(f"\n>>> {name}: P={row['precision']} R={row['recall']} mAP50={row['mAP50']}")
if len(rows)==2: print(f"\nDELTA: mAP50 {rows[0]['mAP50']} -> {rows[1]['mAP50']}")
with open(MET/"stageb_comparison_v3.csv","w",newline="") as f:
    csv.DictWriter(f, fieldnames=["model","precision","recall","mAP50"]).writerows(rows)
