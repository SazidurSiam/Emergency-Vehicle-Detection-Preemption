FUSION MODULE - QUICK START
===========================
Prereqs: both models trained (04_Results/stageA_yolov8n/weights/best.pt,
         04_Results/audio_cnn/best.pt)

1) DEMO on one image + one 4s wav:
   python 03_Code/fusion/fusion_demo.py --image 02_Data/splits/test/images/<pick one>.jpg
        --audio <path to any 4s wav>
   -> prints Pv, Pa, P_fused, decision + saves annotated image to 04_Results/figures/

2) WEIGHT + THRESHOLD sweep (mechanics validation):
   python 03_Code/fusion/fusion_weights.py
   -> saves fusion_weight_sweep.csv and fusion_threshold_sweep.csv to 04_Results/metrics/

Try --wv 0.4 --wa 0.6 too (proposal predicts audio slightly more reliable).
Stage B: run fusion on your BD street VIDEOS (frame + its own audio track =
synchronized pair) for the REAL fusion-gain experiment.
