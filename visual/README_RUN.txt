STAGE A - ORDER OF OPERATIONS (do exactly this)
===============================================
BEFORE anything: you still need the Roboflow dataset!
  -> Go to the Roboflow Universe link from chat, open "Emergency Vehicle
     Detection" dataset -> Download -> YOLOv8 format (free account)
  -> Extract the zip INTO: 02_Data/visual/raw_roboflow/
  (Kaggle images stay in raw_public/ - we use them later for the
   classification comparison. UrbanSound8K stays in audio/ - next phase.)

TERMINAL (every new window):
  set "TEMP=C:\Temp" && set "TMP=C:\Temp"
  call %USERPROFILE%naconda3\Scriptsctivate.bat thesis
  cd C:\path	o\Thesis_EmergencyVehicle

1) PREP  (1 min) : python 03_Code\prep_data.py
2) TRAIN (~1 hr) : python 03_Code	rain_yolo.py
3) EVAL  (5 min) : python 03_Code\evaluate.py

After training, your results live in:
  04_Results\stageA_yolov8n\           (curves, weights)
  04_Results\metrics\stageA_metrics.csv
  04_Resultsigures\sample_detections