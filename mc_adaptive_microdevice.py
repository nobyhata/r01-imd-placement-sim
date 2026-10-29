"""
Monte Carlo robustness test of the Bayesian adaptive (GP) microdevice planner
(Hata, "Adaptive Bayesian Game-Theoretic Planning for MRI-Guided Multi-Needle
Prostate Interventions", Appendix B). Reproduces Figure 3 (P8) of the R01.

Planner logic follows Appendix B (GP: SE kernel, l=10 mm, sf=2.5 mm; objective
= spread - 1.4*distance-to-free-ideal-slot - 4*outside; ideal slots = centered
square at 0.55 R). Per virtual patient we draw a random linear deflection field
d(x,y) = [a + b*y, c + d*x], |(a,c)| ~ U(0.5,1.0)*mag, b,d ~ U(-0.18, 0.18),
plus N(0, noise) per insertion. Static and adaptive arms share the same field
and noise (common random numbers). 2D didactic model, not a clinical tool.
"""
import itertools
import numpy as np
from scipy.stats import wilcoxon


def simulate(seed, les_r, mag, noise, adaptive, n_dev=4, grad=0.18):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi); m = rng.uniform(0.5, 1.0) * mag
    a, c = m * np.cos(ang), m * np.sin(ang); b, d = rng.uniform(-grad, grad, 2)
    nz = rng.normal(0, noise, (n_dev, 2))
    ideal = 0.55 * les_r * np.array([[np.cos(t), np.sin(t)] for t in np.deg2rad([45, 135, 225, 315])])
    gx = np.linspace(-les_r, les_r, 41)
    cand = np.array([[x, y] for x in gx for y in gx if np.hypot(x, y) <= les_r - 0.3])
    X = np.empty((0, 2)); Y = np.empty((0, 2)); l, sf, sn = 10.0, 2.5, max(noise, 0.1)
    kf = lambda A, B: sf**2 * np.exp(-0.5 * ((A[:, None] - B[None]) ** 2).sum(-1) / l**2)
    placed, used = [], set()
    for k in range(n_dev):
        mean = np.zeros_like(cand) if len(X) == 0 else kf(cand, X) @ np.linalg.solve(kf(X, X) + sn**2 * np.eye(len(X)), Y)
        pred = cand + mean
        free = [(i, s) for i, s in enumerate(ideal) if i not in used]; fs = np.array([s for _, s in free])
        dslot = np.min(np.linalg.norm(pred[:, None] - fs[None], axis=2), axis=1)
        spread = np.min(np.linalg.norm(pred[:, None] - np.array(placed)[None], axis=2), axis=1) if placed else 0
        out = np.maximum(0, np.linalg.norm(pred, axis=1) - les_r)
        j = int(np.argmax(spread - 1.4 * dslot - 4.0 * out))
        used.add(free[int(np.argmin(np.linalg.norm(fs - pred[j], axis=1)))][0])
        plan = cand[j]; act = plan + np.array([a + b * plan[1], c + d * plan[0]]) + nz[k]
        placed.append(act)
        if adaptive:
            X = np.vstack([X, plan]); Y = np.vstack([Y, act - plan])
    P = np.array(placed)
    rmse = min(np.sqrt(np.mean([np.sum((P[i] - ideal[pm[i]]) ** 2) for i in range(n_dev)]))
               for pm in itertools.permutations(range(n_dev)))
    inside = np.linalg.norm(P, axis=1) <= les_r
    return rmse, inside


if __name__ == "__main__":
    N = 300
    print("lesion  defl  noise | RMSE static/adapt | inside% static/adapt | adapt better% | p")
    for les_r in (5, 7):
        for mag in (3.5, 6, 9):
            for noise in (0.4, 1.5):
                S = [simulate(s, les_r, mag, noise, False) for s in range(N)]
                A = [simulate(s, les_r, mag, noise, True) for s in range(N)]
                rs, ra = np.array([x[0] for x in S]), np.array([x[0] for x in A])
                is_, ia = np.array([x[1] for x in S]), np.array([x[1] for x in A])
                print(f"{2*les_r:4d}mm {mag:5.1f} {noise:5.1f} | {rs.mean():5.2f} {ra.mean():5.2f} | "
                      f"{100*is_.mean():5.1f} {100*ia.mean():5.1f} | {100*np.mean(ra < rs):4.0f} | {wilcoxon(rs, ra).pvalue:.1e}")
    print("\nPer-device inside %, 10 mm lesion, mag 9, noise 0.4")
    for ad in (False, True):
        M = np.array([simulate(s, 5, 9, 0.4, ad)[1] for s in range(N)])
        print("adaptive" if ad else "static  ", np.round(100 * M.mean(0)))
    print("\nWith steering (residual deflection mag 2.5 ~ 2.2 mm), noise 0.4")
    for les_r in (5, 7):
        for ad in (False, True):
            M = np.array([simulate(s, les_r, 2.5, 0.4, ad)[1] for s in range(N)])
            print(2 * les_r, "adaptive" if ad else "static  ", round(100 * M.mean(), 1))
