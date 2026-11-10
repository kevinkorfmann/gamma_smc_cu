import unittest
import numpy as np
import pandas as pd
from common import interval_summary,combine_summaries,overlap_mean,normalize_ihs

class CoordinateAndMissingnessTests(unittest.TestCase):
    def test_closed_boundaries_and_finite_denominator(self):
        got=interval_summary([9,10,11,12],[4,np.nan,-3,7],[10],[11])
        np.testing.assert_allclose(got[0],[2,1,3,1,3,9])
    def test_empty_missing_and_two_disjoint_flanks(self):
        a=interval_summary([3,8],[np.nan,4],[1],[3])
        b=interval_summary([3,8],[np.nan,4],[7],[9])
        z=combine_summaries(a,b)
        self.assertTrue(np.isnan(a[0,2]));self.assertEqual(z[0,1],1)
        self.assertEqual(z[0,2],4)
    def test_map_gap_not_filled_and_no_off_by_one(self):
        self.assertEqual(overlap_mean(np.array([0,20]),np.array([10,30]),np.array([.2,.8]),5,25),(.5,.5))
        val,cov=overlap_mean(np.array([0]),np.array([10]),np.array([.2]),10,11)
        self.assertTrue(np.isnan(val));self.assertEqual(cov,0)
    def test_normalization_insufficient_bins_stay_missing(self):
        d=pd.DataFrame({'freq':[.02]*60+[.3]*2,'ihs':list(range(60))+[1,2]})
        z=normalize_ihs(d)
        self.assertTrue(np.isnan(z[-2:]).all())
        self.assertAlmostEqual(z[:60].std(),1)

if __name__=='__main__':unittest.main()
