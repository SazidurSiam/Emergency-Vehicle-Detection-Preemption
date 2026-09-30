SIGNAL PREEMPTION SIMULATION - QUICK START
==========================================
Prereqs: thesis env active (pygame is bundled with it? if not:
         pip install pygame)

Run:   python 03_Code/simulation/simulate_preemption.py
Play:  N / E / S / W = emergency vehicle approaching from that direction
       ESC = quit
Auto:  one random emergency every 25s (edit AUTO_EVENT_SECONDS at top,
       set 0 for keyboard-only)

Outputs:
  04_Results/metrics/latency_metrics.csv   (detection-to-green latency per event)
  04_Results/figures/sim/preemption_*.png  (screenshots for your thesis)

For your thesis: report mean/max latency + the fact that the full loop
(visual + audio + fusion) ran at X FPS on an Intel i5-3230M (CPU-only).
