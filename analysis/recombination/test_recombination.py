"""Check scientific coordinate and aggregation invariants without GPU inference."""
import unittest
import numpy as np
from infer_maps import weighted_intervals

class AggregationTests(unittest.TestCase):
    def test_partial_coverage_and_boundaries(self):
        rates,covered=weighted_intervals([10,20],[20,40],[1.,3.],[0,10,15,40],[10,20,30,50])
        np.testing.assert_array_equal(covered,[0,10,15,0])
        np.testing.assert_allclose(rates[1:3],[1,7/3])
        self.assertTrue(np.isnan(rates[[0,3]]).all())

    def test_conserves_integrated_map(self):
        left=np.array([7,20,53]);right=np.array([20,53,101]);r=np.array([.2,2.,.7])
        starts=np.arange(0,110,10);ends=starts+10
        avg,cov=weighted_intervals(left,right,r,starts,ends)
        self.assertAlmostEqual(np.nansum(avg*cov),np.dot(right-left,r))
        self.assertEqual(cov.sum(),94)

if __name__=='__main__':unittest.main()
