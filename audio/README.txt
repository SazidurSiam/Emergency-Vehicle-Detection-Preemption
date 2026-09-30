AUDIO STAGE A - ORDER OF OPERATIONS
===================================
Prereqs: UrbanSound8K extracted into 02_Data/audio/raw_urbansound8k/
         (must contain UrbanSound8K.csv under a metadata/ folder anywhere)
         thesis env active (set TEMP + activate.bat thesis)

TERMINAL (from E:\Thesis Paper\Thesis_EmergencyVehicle):
1) python 03_Code/audio/prep_audio.py        (~30-60 min: MFCC extraction)
2) python 03_Code/audio/train_audio_cnn.py   (~30-60 min: CNN training)
3) python 03_Code/audio/evaluate_audio.py    (~5-10 min incl. -6dB test)

Outputs:
  04_Results/audio_cnn/best.pt              (siren classifier weights)
  04_Results/metrics/audio_metrics.csv      (Acc, P, R, F1, FPR, -6dB acc)
  04_Results/figures/audio_confusion_matrix.png

Then: the FUSION module combines this with the YOLO detector
      (04_Results/stageA_yolov8n/weights/best.pt) -> P_fused = wv*Pv + wa*Pa
