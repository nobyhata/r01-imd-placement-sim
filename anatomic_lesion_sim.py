"""
P2 simulation on real lesion anatomy (PI-CAI expert csPCa delineations, CC BY-NC 4.0,
github.com/DIAGNijmegen/picai_labels, human_expert/resampled, 220 cases).

For each case, the largest csPCa lesion (connected component, ISUP>=2 label) is taken, and
its largest axial cross-section is used as the 2D target in the plane perpendicular to a
transperineal needle (needles run cranio-caudally, so deflection lies in the axial plane).
Planner, deflection model, noise, and static vs adaptive arms are identical to
mc_adaptive_microdevice.py, except that (i) the lesion is the real cross-section rather than
a circle, (ii) the four ideal slots are the centroids of a 4-region centroidal Voronoi
(k-means) partition of the lesion, and (iii) "inside" is tested against the lesion mask.
"""
import sys, glob, itertools, numpy as np, nibabel as nib
from scipy import ndimage
from sklearn.cluster import KMeans

def load_lesions(folder):
    out = []
    for f in sorted(glob.glob(folder + "/*.nii.gz")):
        img = nib.load(f); a = np.asarray(img.dataobj) > 0
        if not a.any(): continue
        lab, n = ndimage.label(a)
        sizes = ndimage.sum(a, lab, range(1, n + 1)); big = lab == (1 + int(np.argmax(sizes)))
        dx, dy = img.header.get_zooms()[:2]
        z = int(np.argmax(big.sum((0, 1)))); sl = big[:, :, z]
        ij = np.argwhere(sl); i0, j0 = ij.min(0) - 40; i1, j1 = ij.max(0) + 41
        i0, j0 = max(i0, 0), max(j0, 0); m = sl[i0:i1, j0:j1]
        ij = np.argwhere(m); ctr = ij.mean(0)
        pts = np.c_[(ij[:, 0] - ctr[0]) * dx, (ij[:, 1] - ctr[1]) * dy]
        dist_out = ndimage.distance_transform_edt(~m, sampling=(dx, dy))
        area = len(pts) * dx * dy
        out.append(dict(case=f.split('/')[-1][:-7], pts=pts, mask=m, dist=dist_out, ctr=ctr,
                        pix=(dx, dy), area=area, deq=2*np.sqrt(area/np.pi)))
    return out

def dist_outside(L, q):
    # distance (mm) from each point q to the lesion; 0 inside; large outside the cropped field
    q = np.atleast_2d(q); dx, dy = L['pix']; D = L['dist']
    i = np.rint(q[:, 0] / dx + L['ctr'][0]).astype(int); j = np.rint(q[:, 1] / dy + L['ctr'][1]).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < D.shape[0]) & (j < D.shape[1])
    d = np.full(len(q), 99.0); d[ok] = D[i[ok], j[ok]]
    return d

def simulate(L, seed, mag, noise, adaptive, n_dev=4, grad=0.18):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2*np.pi); m = rng.uniform(0.5, 1.0)*mag
    a, c = m*np.cos(ang), m*np.sin(ang); b, d = rng.uniform(-grad, grad, 2)
    nz = rng.normal(0, noise, (n_dev, 2))
    pts = L['pts']
    cpts = pts if len(pts) <= 1500 else pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    ideal = L['ideal']
    cand = pts if len(pts) <= 400 else pts[rng.choice(len(pts), 400, replace=False)]
    X = np.empty((0, 2)); Y = np.empty((0, 2)); l, sf, sn = 10.0, 2.5, max(noise, 0.1)
    kf = lambda A, B: sf**2*np.exp(-0.5*((A[:, None]-B[None])**2).sum(-1)/l**2)
    placed, used = [], set()
    for k in range(n_dev):
        mean = np.zeros_like(cand) if len(X) == 0 else kf(cand, X) @ np.linalg.solve(kf(X, X)+sn**2*np.eye(len(X)), Y)
        pred = cand + mean
        free = [(i, s) for i, s in enumerate(ideal) if i not in used]; fs = np.array([s for _, s in free])
        dslot = np.min(np.linalg.norm(pred[:, None]-fs[None], axis=2), axis=1)
        spread = np.min(np.linalg.norm(pred[:, None]-np.array(placed)[None], axis=2), axis=1) if placed else 0
        out = dist_outside(L, pred)
        j = int(np.argmax(spread - 1.4*dslot - 4.0*out))
        used.add(free[int(np.argmin(np.linalg.norm(fs-pred[j], axis=1)))][0])
        plan = cand[j]; act = plan + np.array([a + b*plan[1], c + d*plan[0]]) + nz[k]
        placed.append(act)
        if adaptive: X = np.vstack([X, plan]); Y = np.vstack([Y, act-plan])
    P = np.array(placed)
    perr = min(np.sqrt(np.mean([np.sum((P[i]-ideal[pm[i]])**2) for i in range(n_dev)]))
               for pm in itertools.permutations(range(n_dev)))
    inside = dist_outside(L, P) == 0
    cov = np.sqrt(((cpts[:, None]-P[None])**2).sum(-1)).min(1).max() / (L['deq']/2)
    return perr, inside, cov

if __name__ == "__main__":
    Ls = load_lesions(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    for L in Ls:
        L['ideal'] = KMeans(4, n_init=10, random_state=0).fit(L['pts']).cluster_centers_
    deq = np.array([L['deq'] for L in Ls])
    print(f"lesions {len(Ls)}; axial equivalent diameter median {np.median(deq):.1f} mm (IQR {np.percentile(deq,25):.1f}-{np.percentile(deq,75):.1f}), <=10 mm: {np.mean(deq<=10)*100:.0f}%")
    ideal_cov = []
    for L in Ls:
        ideal_cov.append(np.sqrt(((L['pts'][:, None]-L['ideal'][None])**2).sum(-1)).min(1).max()/(L['deq']/2))
    print(f"lesion pixel spacing: {sorted(set(L['pix'] for L in Ls))}")
    print(f"ideal 4-slot covering ratio median {np.median(ideal_cov):.2f} (IQR {np.percentile(ideal_cov,25):.2f}-{np.percentile(ideal_cov,75):.2f})")
    print("mag noise | inside% static adapt | (<=10mm) static adapt | pattern err static adapt | cov<=0.9 % static adapt | adapt better(pattern)%")
    for mag in (2.5, 3.5, 6, 9):
        for noise in (0.4, 1.5):
            R = {False: [], True: []}
            for li, L in enumerate(Ls):
                for s in range(N):
                    for ad in (False, True):
                        R[ad].append((li,) + simulate(L, 1000*li+s, mag, noise, ad))
            def agg(ad, sel=None):
                rows = [r for r in R[ad] if sel is None or sel(r[0])]
                ins = np.mean([r[2].mean() for r in rows])*100
                pe = np.mean([r[1] for r in rows]); cv = np.mean([r[3] <= 0.9 for r in rows])*100
                return ins, pe, cv, rows
            s_in, s_pe, s_cv, sr = agg(False); a_in, a_pe, a_cv, ar = agg(True)
            small = lambda i: deq[i] <= 10
            ss = agg(False, small)[0]; aa = agg(True, small)[0]
            better = np.mean([y[1] < x[1] for x, y in zip(sr, ar)])*100
            print(f"{mag:4.1f} {noise:4.1f} | {s_in:5.1f} {a_in:5.1f} | {ss:5.1f} {aa:5.1f} | {s_pe:4.1f} {a_pe:4.1f} | {s_cv:5.1f} {a_cv:5.1f} | {better:4.0f}", flush=True)
