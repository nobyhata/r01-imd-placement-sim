"""Derive reservoir-in-target rates for the Aim 2.5 swine study.
Device: 6 mm long, axis along insertion (z); 20 reservoirs at 5 levels along the axis (4 per level).
Lesion: sphere, diameter D. Device 'in target' if >= 2/3 of reservoirs lie inside the lesion.
Error: lateral (x,y) Gaussian with mean in-plane magnitude m_lat (Rayleigh); depth error sd_z.
"""
import numpy as np
rng=np.random.default_rng(1)
N=200000
levels=np.linspace(-2.5,2.5,5)          # reservoir levels along 6 mm device (reservoir span ~5 mm)
def rate(m_lat, sd_z, D):
    sig=m_lat/np.sqrt(np.pi/2)            # Rayleigh mean -> sigma
    x=rng.normal(0,sig,N); y=rng.normal(0,sig,N); z=rng.normal(0,sd_z,N)
    r=D/2
    inside=np.zeros(N)
    for l in levels:
        inside+= ((x**2+y**2+(z+l)**2)<=r**2)*4
    frac=inside/20
    return (frac>=2/3).mean(), (frac>0).mean(), frac.mean()
print("D=10 mm lesion, depth sd 2 mm")
for m in [2.2,3.0,3.1,4.0,4.4,5.0,6.4,8.3,9.1]:
    a,b,c=rate(m,2.0,10); print(f"in-plane mean {m:4.1f} mm: >=2/3 reservoirs in lesion {a:5.1%}; any reservoir {b:5.1%}; mean reservoir fraction {c:5.1%}")
print("D=13 mm (clinical median)")
for m in [6.4,8.3]:
    a,b,c=rate(m,2.0,13); print(f"in-plane mean {m:4.1f}: >=2/3 {a:5.1%}; any {b:5.1%}; mean frac {c:5.1%}")
