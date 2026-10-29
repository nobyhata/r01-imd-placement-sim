import numpy as np
from scipy import stats
def paired_power(d,n,alpha):
    df=n-1; nc=d*np.sqrt(n); t=stats.t.ppf(1-alpha/2,df)
    return 1-stats.nct.cdf(t,df,nc)+stats.nct.cdf(-t,df,nc)
for n in [24,28,30]: print("bench H2b n",n,round(paired_power(1/1.5,n,0.025),3))
print("H2a n=30",round(paired_power(1.0,30,0.025),3))
def prop_power(p0,p1,n,alpha=0.05):
    h=2*np.arcsin(np.sqrt(p1))-2*np.arcsin(np.sqrt(p0)); z=stats.norm.ppf(1-alpha/2)
    return stats.norm.cdf(abs(h)*np.sqrt(n/2)-z)
for p0,p1 in [(0.40,0.70),(0.30,0.70),(0.25,0.69)]:
    for DE in [1.4,2.0]:
        print(f"swine p0={p0} p1={p1} DE={DE} n_eff={60/DE:.0f} power={prop_power(p0,p1,60/DE):.3f}")
