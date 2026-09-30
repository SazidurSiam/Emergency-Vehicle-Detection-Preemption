"""
simulate_preemption.py - Traffic signal preemption demo (proposal Objective 2)
------------------------------------------------------------------------------
4-way intersection (N/E/S/W). Normal operation = fixed-time signal cycle.
On an emergency detection event (fused P >= tau) for one approach:
  cross traffic gets YELLOW clearance -> emergency approach goes GREEN
  -> holds -> returns to normal cycle.

Events: press N / E / S / W to inject "emergency from <dir>",
        or automatic one every AUTO_EVENT_SECONDS (0 = keyboard only).

Logs end-to-end detection-to-green latency per event + loop FPS.
Outputs: 04_Results/metrics/latency_metrics.csv + figures/sim/ screenshots.

Run:  python 03_Code/simulation/simulate_preemption.py
Quit: ESC or close window.
"""
from pathlib import Path
import time, csv
import numpy as np
import pygame

BASE = Path(__file__).resolve().parents[2]
MET  = BASE / "04_Results" / "metrics"
FIG  = BASE / "04_Results" / "figures" / "sim"
MET.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------- config ---------------------------------------
W, H      = 800, 800
GREEN_T, YELLOW_T, ALLRED_T = 5.0, 1.5, 0.5
HOLD_T    = 6.0
TAU       = 0.85
AUTO_EVENT_SECONDS = 25
DIRS = ["N", "E", "S", "W"]
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
LIGHT_POS = {"N": (W//2 - 60, H//2 - 90), "S": (W//2 + 60, H//2 + 90),
             "E": (W//2 + 90, H//2 - 60), "W": (W//2 - 90, H//2 + 60)}
COL = {"red": (200, 40, 40), "yellow": (220, 200, 40), "green": (40, 200, 80),
       "gray": (90, 90, 90), "road": (60, 60, 60), "bg": (35, 45, 35)}

# ----------------------------- controller ------------------------------------
class Controller:
    def __init__(self):
        self.green_pair = ("N", "S")
        self.state = "green"; self.timer = GREEN_T
        self.emergency = None
        self.log = []

    def preempt(self, d, P_fused):
        if self.emergency: return
        self.emergency = {"dir": d, "P": P_fused, "t_detect": time.time(),
                          "phase": "clearance", "timer": YELLOW_T + ALLRED_T,
                          "t_green": None, "shot": False}

    def update(self, dt):
        ev = self.emergency
        if ev:
            ev["timer"] -= dt
            if ev["phase"] == "clearance" and ev["timer"] <= 0:
                ev["phase"] = "green"; ev["timer"] = HOLD_T
                ev["t_green"] = time.time()
                lat = ev["t_green"] - ev["t_detect"]
                self.log.append({"dir": ev["dir"], "P_fused": round(ev["P"], 3),
                                 "latency_s": round(lat, 3)})
                print(f"[PREEMPT] dir={ev['dir']} P_fused={ev['P']:.3f} "
                      f"-> GREEN in {lat:.3f}s", flush=True)
            elif ev["phase"] == "green" and ev["timer"] <= 0:
                d = ev["dir"]
                self.emergency = None
                self.state, self.timer = "green", GREEN_T
                self.green_pair = (d, OPPOSITE[d])
            return
        self.timer -= dt
        if self.timer <= 0:
            if self.state == "green":
                self.state, self.timer = "yellow", YELLOW_T
            elif self.state == "yellow":
                self.state, self.timer = "allred", ALLRED_T
            else:
                self.green_pair = ("E", "W") if self.green_pair == ("N", "S") else ("N", "S")
                self.state, self.timer = "green", GREEN_T

    def light(self, d):
        if self.emergency:
            ev = self.emergency
            if ev["phase"] == "clearance":
                return "yellow" if d in self.green_pair else "red"
            return "green" if (d == ev["dir"] or d == OPPOSITE[ev["dir"]]) else "red"
        if self.state == "allred": return "red"
        if self.state == "green":
            return "green" if d in self.green_pair else "red"
        return "yellow" if d in self.green_pair else "red"

# ----------------------------- pygame app ------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Emergency Vehicle Signal Preemption - thesis demo")
    font = pygame.font.SysFont("consolas", 18)
    big  = pygame.font.SysFont("consolas", 26, bold=True)
    ctrl = Controller()
    clock = pygame.time.Clock()
    rng = np.random.default_rng(7)
    last_auto = time.time()
    fps_samples = []
    shot_n = 0
    running = True
    while running:
        clock.tick(60)
        fps_samples.append(clock.get_fps())
        dt = 1.0 / 60.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                for k, d in [(pygame.K_n, "N"), (pygame.K_e, "E"),
                             (pygame.K_s, "S"), (pygame.K_w, "W")]:
                    if e.key == k:
                        P = round(float(rng.uniform(TAU, 0.99)), 3)
                        print(f"[DETECT] fused P={P} from {d}", flush=True)
                        ctrl.preempt(d, P)
        if AUTO_EVENT_SECONDS and time.time() - last_auto > AUTO_EVENT_SECONDS:
            d = DIRS[int(rng.integers(0, 4))]
            P = round(float(rng.uniform(TAU, 0.99)), 3)
            print(f"[DETECT-auto] fused P={P} from {d}", flush=True)
            ctrl.preempt(d, P); last_auto = time.time()

        ctrl.update(dt)

        screen.fill(COL["bg"])
        pygame.draw.rect(screen, COL["road"], (W//2 - 70, 0, 140, H))
        pygame.draw.rect(screen, COL["road"], (0, H//2 - 70, W, 140))
        for d, (lx, ly) in LIGHT_POS.items():
            pygame.draw.rect(screen, (20, 20, 20), (lx-22, ly-32, 44, 64))
            for i, c in enumerate(["red", "yellow", "green"]):
                on = ctrl.light(d) == c
                pygame.draw.circle(screen, COL[c] if on else COL["gray"],
                                   (lx, ly - 20 + i*20), 8)
        if ctrl.emergency:
            status = (f"PREEMPTION! emergency from {ctrl.emergency['dir']} "
                      f"({ctrl.emergency['phase']})")
            color = (255, 255, 0)
        else:
            status = f"NORMAL CYCLE - green: {ctrl.green_pair[0]}-{ctrl.green_pair[1]}"
            color = (255, 255, 255)
        screen.blit(big.render(status, True, color), (20, 12))
        screen.blit(font.render("Press N/E/S/W = inject emergency  |  ESC = quit",
                                True, (200, 200, 200)), (20, H - 30))
        screen.blit(font.render(f"FPS: {clock.get_fps():.0f}   events: {len(ctrl.log)}",
                                True, (180, 220, 180)), (W - 280, H - 30))
        pygame.display.flip()

        if ctrl.emergency and ctrl.emergency["phase"] == "green" and not ctrl.emergency["shot"]:
            ctrl.emergency["shot"] = True
            shot_n += 1
            pygame.image.save(screen, str(FIG / f"preemption_{shot_n}.png"))

    pygame.quit()
    if ctrl.log:
        with open(MET / "latency_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dir", "P_fused", "latency_s"])
            w.writeheader(); w.writerows(ctrl.log)
        lats = [r["latency_s"] for r in ctrl.log]
        print(f"\nEvents: {len(lats)}  mean latency: {sum(lats)/len(lats):.3f}s  "
              f"max: {max(lats):.3f}s", flush=True)
        print(f"Sim FPS: mean={sum(fps_samples)/len(fps_samples):.1f} "
              f"min={min(fps_samples):.1f}", flush=True)
        print(f"Saved: {MET / 'latency_metrics.csv'} and {FIG}/preemption_*.png")
    else:
        print("No preemption events logged.")

if __name__ == "__main__":
    main()
