"""Fixed summaries, conservative empirical thresholds and localisation metrics."""
import numpy as np

METHODS=['Gamma','iHS body','XP-EHH body','iHS flanks','XP-EHH flanks','iHS maximum','XP-EHH maximum']

def threshold(values,alpha=.05):
    # Missing scores count as no call, preserving the unconditional denominator.
    values=np.sort(np.where(np.isfinite(values),values,-np.inf))
    if not len(values):raise ValueError('Empty calibration')
    i=min(len(values)-1,int(np.ceil((len(values)+1)*(1-alpha)))-1)
    return float(values[i])

def detect(values,cutoff):
    values=np.asarray(values);return np.isfinite(values)&(values>cutoff)

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

def region_summary(pos,values,lo,hi,flanks=False,min_sites=1,kind='fraction'):
    pos=np.asarray(pos);values=np.asarray(values)
    inside=(pos>=lo)&(pos<hi)
    if flanks:inside=((pos>=lo-100000)&(pos<lo))|((pos>=hi)&(pos<hi+100000))
    v=values[inside&np.isfinite(values)]
    if len(v)<min_sites:return np.nan,len(v)
    if kind=='gamma':score=-float(np.mean(v))
    elif kind=='maximum':score=float(np.max(v))
    else:score=float(np.mean(v>2))
    return score,len(v)

def summaries(z,norm,center,width):
    lo=center-width/2;hi=center+width/2
    frequency=z['derived_frequency'];bins=np.minimum((frequency*20).astype(int),19)
    iz=(z['ihs']-np.asarray(norm['ihs_mean'])[bins])/np.asarray(norm['ihs_sd'])[bins]
    xz=(z['xp']-norm['xp_mean'])/norm['xp_sd'];iz=np.abs(iz)
    calls=[(z['gpos'],z['logtmrca'],False,2,'gamma'),
           (z['ipos'],iz,False,1,'fraction'),(z['xpos'],xz,False,1,'fraction'),
           (z['ipos'],iz,True,1,'fraction'),(z['xpos'],xz,True,1,'fraction'),
           (z['ipos'],iz,False,1,'maximum'),(z['xpos'],xz,False,1,'maximum')]
    return [region_summary(p,v,lo,hi,f,k,t) for p,v,f,k,t in calls]

def localisation(scores,centers,target):
    s=np.asarray(scores);c=np.asarray(centers);ok=np.isfinite(s)
    if not ok.any():return dict(scan_score=np.nan,error=np.nan,ties=0,tie_span=np.nan)
    best=np.max(s[ok]);ties=c[ok&(s==best)]
    return dict(scan_score=float(best),error=float(np.median(np.abs(ties-target))),
                ties=len(ties),tie_span=float(np.ptp(ties)))
