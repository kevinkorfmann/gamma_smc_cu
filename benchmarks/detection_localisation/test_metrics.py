import unittest
import numpy as np
from metrics import threshold,detect,region_summary,localisation,wilson

class Checks(unittest.TestCase):
 def test_missing_counts_in_denominator(self):
  x=np.r_[np.full(50,np.nan),np.arange(50.)];t=threshold(x)
  self.assertEqual(t,45.);self.assertEqual(detect(x,t).sum(),4)
 def test_ties_are_conservative(self):
  x=np.r_[np.zeros(90),np.ones(10)];self.assertEqual(detect(x,threshold(x)).sum(),0)
 def test_half_open_and_disjoint_flanks(self):
  p=np.array([0,99,100,199,200,299]);v=np.arange(6.)
  self.assertEqual(region_summary(p,v,100,200,kind='maximum'),(3.,2))
  self.assertEqual(region_summary(p,v,100,200,flanks=True,kind='maximum'),(5.,4))
 def test_gamma_geometric_aggregation(self):
  self.assertAlmostEqual(region_summary([1,2],np.log([10,100]),0,3,min_sites=2,kind='gamma')[0],-np.log(np.sqrt(1000)))
  self.assertTrue(np.isnan(region_summary([1],np.log([10]),0,3,min_sites=2,kind='gamma')[0]))
 def test_tied_localisation_no_oracle_choice(self):
  r=localisation([1,3,3,np.nan],[0,10,20,30],11)
  self.assertEqual(r['error'],5);self.assertEqual(r['tie_span'],10);self.assertEqual(r['ties'],2)
 def test_no_score_no_position(self):self.assertTrue(np.isnan(localisation([np.nan],[0],0)['error']))
 def test_wilson_edges(self):
  lo,hi=wilson(0,100);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)
  lo,hi=wilson(100,100);self.assertLess(lo,1);self.assertAlmostEqual(hi,1)
if __name__=='__main__':unittest.main()
