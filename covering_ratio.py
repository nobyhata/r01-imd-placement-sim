"""
Spread metric for the R01 (Approach): covering ratio = largest distance from any
lesion point to the nearest released device, divided by the lesion radius
(equal-volume sphere radius in 3D; disk radius in this 2D model).
Uses simulate() from mc_adaptive_microdevice.py (P2), which must return P.
Run from this folder:  python covering_ratio.py
Results (N=300 per arm, noise 0.4 mm), share of lesions with ratio <= 0.9:
  residual deflection 2.5 mm: 10 mm lesion static 8% / adaptive 70%; 14 mm static 39% / adaptive 91%
  ideal four-device pattern (0.55R square): 0.72; 3D, 6 mm devices: 0.72-0.75
"""
import numpy as np, importlib.util, pathlib
src = pathlib.Path(__file__).with_name("mc_adaptive_microdevice.py").read_text()
src = src.replace("    return rmse, inside\n", "    return rmse, inside, P\n")
ns = {"__name__": "mcsim"}; exec(compile(src, "mcsim", "exec"), ns); simulate = ns["simulate"]

def cover(P, R, n=81):
    g = np.linspace(-R, R, n); pts = np.array([[x, y] for x in g for y in g if x*x + y*y <= R*R])
    return np.min(np.linalg.norm(pts[:, None] - P[None], axis=2), axis=1).max() / R

if __name__ == "__main__":
    N = 300
    for R in (5, 7):
        for mag in (2.5, 3.5, 6, 9):
            row = []
            for ad in (False, True):
                c = np.array([cover(simulate(s, R, mag, 0.4, ad)[2], R) for s in range(N)])
                row.append(f"median {np.median(c):.2f}, <=0.9 {100*np.mean(c <= 0.9):.0f}%")
            print(f"{2*R} mm lesion, deflection {mag} mm: static {row[0]} | adaptive {row[1]}")
