"""
P2 simulation in 3D on real lesion anatomy, with each IMD modeled as a 6.5 mm device
(PI-CAI expert csPCa delineations, CC BY-NC 4.0, github.com/DIAGNijmegen/picai_labels).

Geometry. Transperineal needles run cranio-caudally, i.e., along the MRI slice axis (z).
Each IMD is a 6.5 mm x 0.75 mm cylinder released at the needle tip and lying along the
needle path (Rainer et al., J Magn Reson Imaging 2025), so it spans z_tip-6.5 .. z_tip.
Its 20 reservoirs sit at 7 levels along the axis (3 per level around the circumference);
a level counts as in target when its point lies inside the 3D lesion mask.

Primary outcome. Reservoir-in-target (RIT) fraction per device, as in Aims 1-2.
Secondary. Share of devices with any reservoir level inside the lesion (the pilot metric).

Devices per lesion follow the pilot recommendation to place fewer devices in small lesions:
largest lesion dimension <=10 mm: 2 devices; 10-15 mm: 3; >15 mm: 4.
Planned device centres are the centroids of a k-means partition of the lesion voxels.

Error model per simulated patient (identical in both arms): a shared lateral offset of
magnitude U(0.5,1)*mag in a random direction, a weak lateral gradient (+/-0.18 mm/mm),
a shared depth bias U(-1,1)*depth_bias, and per-insertion noise N(0, noise) laterally and
N(0, depth_noise) in depth.
Static arm: every device is aimed at its planned position.
Adaptive arm (LEAP): a Gaussian process (SE kernel, length 10 mm) learns the 3D error from
devices already placed and shifts each next aim point against the predicted error.
"""
import sys, glob, numpy as np, nibabel as nib
from scipy import ndimage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

DEV_LEN, N_LEVELS = 6.5, 7
LEVELS = np.linspace(0.45, DEV_LEN - 0.45, N_LEVELS)          # mm proximal to the tip

def load(folder):
    out = []
    for f in sorted(glob.glob(folder + "/*.nii.gz")):
        img = nib.load(f); a = np.asarray(img.dataobj) > 0
        if not a.any(): continue
        lab, n = ndimage.label(a); s = ndimage.sum(a, lab, range(1, n + 1))
        m = lab == 1 + int(np.argmax(s)); sp = np.array(img.header.get_zooms()[:3], float)
        ijk = np.argwhere(m); lo = ijk.min(0) - np.array([30, 30, 6]); hi = ijk.max(0) + np.array([31, 31, 7])
        lo = np.maximum(lo, 0); m = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ijk = np.argwhere(m); xyz = ijk * sp; ctr = xyz.mean(0); xyz = xyz - ctr
        cc = (ijk[:, 2].max() - ijk[:, 2].min() + 1) * sp[2]
        sl = m[:, :, int(np.argmax(m.sum((0, 1))))]; ij = np.argwhere(sl) * sp[:2]
        ax = pdist(ij).max() if len(ij) > 1 else sp[0]
        out.append(dict(case=f.split('/')[-1][:-7], mask=m, sp=sp, ctr=ctr, xyz=xyz,
                        D=max(ax, cc), ax=ax, cc=cc, vol=m.sum() * sp.prod()))
    return out

def inside(L, q):
    q = np.atleast_2d(q); idx = np.rint((q + L['ctr']) / L['sp']).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(L['mask'].shape)), axis=1)
    r = np.zeros(len(q), bool); r[ok] = L['mask'][idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return r

def n_devices(D):
    return 2 if D <= 10 else 3 if D <= 15 else 4

def simulate(L, seed, mag, noise, adaptive, depth_bias=2.0, depth_noise=1.0, grad=0.18):
    rng = np.random.default_rng(seed); n = L['n']
    ang = rng.uniform(0, 2 * np.pi); m = rng.uniform(0.5, 1.0) * mag
    a, c = m * np.cos(ang), m * np.sin(ang); b, d = rng.uniform(-grad, grad, 2)
    zb = rng.uniform(-1, 1) * depth_bias
    nz = np.c_[rng.normal(0, noise, (n, 2)), rng.normal(0, depth_noise, n)]
    l, sf, sn = 10.0, 2.5, max(noise, 0.1)
    kf = lambda A, B: sf**2 * np.exp(-0.5 * ((A[:, None] - B[None])**2).sum(-1) / l**2)
    X = np.empty((0, 3)); Y = np.empty((0, 3)); rit, any_in = [], []
    for k, slot in enumerate(L['slots']):
        tip_plan = slot + np.array([0, 0, DEV_LEN / 2])           # device centred on the slot
        aim = tip_plan.copy()
        if adaptive and len(X):
            aim = tip_plan - (kf(tip_plan[None], X) @ np.linalg.solve(kf(X, X) + sn**2 * np.eye(len(X)), Y))[0]
        err = np.array([a + b * aim[1], c + d * aim[0], zb]) + nz[k]
        tip = aim + err
        lv = tip[None] - np.c_[np.zeros((N_LEVELS, 2)), LEVELS]
        hit = inside(L, lv); rit.append(hit.mean()); any_in.append(hit.any())
        X = np.vstack([X, aim]); Y = np.vstack([Y, err])
    return np.array(rit), np.array(any_in)

if __name__ == "__main__":
    Ls = load(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    for L in Ls:
        L['n'] = n_devices(L['D'])
        L['slots'] = KMeans(L['n'], n_init=10, random_state=0).fit(L['xyz']).cluster_centers_
    D = np.array([L['D'] for L in Ls])
    print(f"lesions {len(Ls)}; largest dimension median {np.median(D):.1f} mm, range {D.min():.1f}-{D.max():.1f} mm; "
          f"cranio-caudal extent range {min(L['cc'] for L in Ls):.1f}-{max(L['cc'] for L in Ls):.1f} mm")
    bins = [(0, 10), (10, 15), (15, 20), (20, 27), (27, 999)]
    print("bins (largest dimension, mm):", [(lo, hi, int(((D > lo) & (D <= hi)).sum())) for lo, hi in bins])
    for mag in (2.5, 6, 9):
        for noise in (0.4, 1.5):
            res = {False: [], True: []}
            for li, L in enumerate(Ls):
                for s in range(N):
                    for ad in (False, True):
                        res[ad].append((li,) + simulate(L, 1000 * li + s, mag, noise, ad))
            line = f"offset {mag:3.1f} noise {noise:3.1f} |"
            for lo, hi in bins + [(0, 999)]:
                sel = lambda r: lo < D[r[0]] <= hi
                st = np.mean([r[1].mean() for r in res[False] if sel(r)]) * 100
                ad = np.mean([r[1].mean() for r in res[True] if sel(r)]) * 100
                sa = np.mean([r[2].mean() for r in res[False] if sel(r)]) * 100
                aa = np.mean([r[2].mean() for r in res[True] if sel(r)]) * 100
                line += f" {lo}-{hi}: RIT {st:4.1f}->{ad:4.1f} any {sa:4.1f}->{aa:4.1f} |"
            print(line, flush=True)
