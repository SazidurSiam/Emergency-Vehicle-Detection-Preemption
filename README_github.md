# Multi-Modal Emergency Vehicle Detection & Traffic Signal Preemption (CPU-Only)

Lightweight, CPU-only multi-modal system for emergency vehicle detection and
intelligent traffic signal preemption, designed for South Asian (Bangladesh)
deployment conditions. Undergraduate honours thesis implementation.

**Pipeline:** YOLOv8-nano (visual) + MFCC-CNN siren classifier (audio) ->
weighted-voting decision fusion -> traffic signal preemption simulation.
**Everything runs on CPU** (developed and benchmarked on an Intel Core
i5-3230M, 8 GB RAM, no GPU).

## Results summary (measured)

| Module | Result |
|---|---|
| Visual detector (YOLOv8n, 3.0M params) | mAP@0.5 = 0.9831 (P=0.9854, R=0.9468) on Western test set |
| Siren classifier (SirenCNN, 23,650 params) | 92.57% clean / 75.43% @ -6 dB SNR (noise-augmented) |
| Fusion module | weighted voting, tunable flag rate 35.7%-96.9% |
| Preemption simulation | mean latency 1.992 s @ 59.8 FPS (CPU-only) |

### Stage B - Bangladesh regional adaptation (34 BD-only test images)

| Model | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| Stage A (Western training only) | 0.9033 | 0.8256 | 0.8604 |
| Fine-tune v1 (pilot, unfiltered labels) | 0.3444 | 0.4276 | 0.5631 |
| Fine-tune v2 (filtered, Indian+BD data) | 0.7134 | 0.7483 | 0.8142 |
| Fine-tune v3 (BD-only, frozen backbone) | 0.9798 | 0.7428 | 0.8453 |

Finding: a Western-trained detector already generalizes well to Bangladeshi
ambulances (small visual domain gap for emergency vans); careful regional
fine-tuning shifts the precision/recall profile (v3 achieves 0.98 precision).

## Repository structure

```
03_Code/
  visual/     YOLOv8n Stage-A training + Stage-B fine-tuning + evaluation
  audio/      MFCC extraction, SirenCNN training/evaluation
  fusion/     fusion demo + weight/threshold sweeps + BD video fusion
  simulation/ Pygame traffic signal preemption simulation
```

## Requirements

- Python 3.10 (Anaconda), `ultralytics`, `opencv-python`, `librosa`,
  `soundfile`, `torch`, `pandas`, `scikit-learn`, `pygame`, `yt-dlp`

## Datasets

- Roboflow Emergency Vehicle dataset (1,279 images, 2 classes) - Stage A
- UrbanSound8K (balanced 929 siren / 929 non-siren) - audio stage
- Regional BD set: bikroy.com listings, BD news media, YouTube frames +
  Indian-context public detection data (see thesis Section 4.6 for sources)

Dataset files are not redistributed; sources are described for reproducibility.

## Reproduce

```
# Stage A (visual) - ~8 hrs CPU
python 03_Code/visual/prep_data.py
python 03_Code/visual/train_yolo.py
python 03_Code/visual/evaluate.py

# Audio stage
python 03_Code/audio/prep_audio.py
python 03_Code/audio/train_audio_cnn.py
python 03_Code/audio/evaluate_audio_v2.py

# Fusion demo + simulation
python 03_Code/fusion/fusion_demo.py --image <img> --audio <wav>
python 03_Code/simulation/simulate_preemption.py

# Stage B (regional adaptation)
python 03_Code/visual/merge_bd.py
python 03_Code/visual/fine_tune_bd.py
python 03_Code/visual/eval_stageb.py
```

## License & citation

Research and educational use. If you use this work, please cite the thesis.
Images collected from public web sources are used for training/evaluation
only and are not redistributed.
