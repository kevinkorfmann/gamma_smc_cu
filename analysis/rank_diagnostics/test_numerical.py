"""Independent checks of signs, missingness and statistical definitions."""
import unittest
import numpy as np
from scipy.stats import rankdata,spearmanr
import allel
from compare import correlation

class NumericalChecks(unittest.TestCase):
    def test_spearman_and_partial_against_explicit_residuals(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(500,4));y=x[:,0]+rng.normal(size=500)
        s=.4*y+x[:,1]+rng.normal(size=500)
        self.assertAlmostEqual(correlation(y,s),spearmanr(y,s).statistic)
        design=np.column_stack([np.ones(500)]+[rankdata(c) for c in x.T])
        ry=rankdata(y);rs=rankdata(s)
        ey=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
        es=rs-design@np.linalg.lstsq(design,rs,rcond=None)[0]
        self.assertAlmostEqual(correlation(y,s,x),np.corrcoef(ey,es)[0,1])
    def test_xpehh_sign_and_fixed_focal_sites(self):
        rng=np.random.default_rng(31)
        h1=rng.integers(0,2,(500,40),dtype=np.int8)
        h2=rng.integers(0,2,(500,60),dtype=np.int8)
        h1[250]=0  # Completely fixed focal site, polymorphic in reference.
        pos=np.arange(500)*100+1
        a=allel.xpehh(h1,h2,pos,use_threads=False)
        b=allel.xpehh(h2,h1,pos,use_threads=False)
        np.testing.assert_allclose(a,-b,equal_nan=True,atol=1e-14)
        self.assertTrue(np.isfinite(a[250]))
        scaled=allel.xpehh(h1,h2,pos,map_pos=pos*1e-6,use_threads=False)
        np.testing.assert_allclose(a,scaled,equal_nan=True,atol=1e-12)

if __name__=='__main__':unittest.main()
