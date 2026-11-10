import unittest
import numpy as np
from simulate import selected_intervals,validate_config
from metrics import loss,bin_piecewise,seed_for

class NegativeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.c=dict(model='negative',Ne=16000,s=-.001,h=.5,target='block',width=300000,
                    offset=-50000,selected_fraction=.7,Q=10,burn_Ne=10)
    def test_no_beneficial_or_lethal_scaled_coefficients(self):
        for s in [0,.001,-.1,-1]:
            with self.assertRaises(ValueError):validate_config(dict(self.c,s=s),206)
    def test_population_can_supply_diploid_sample(self):
        with self.assertRaises(ValueError):validate_config(dict(self.c,Ne=1000),314)
    def test_physical_block_coordinates(self):
        np.testing.assert_array_equal(selected_intervals(self.c,3000000,[]),[[1300000,1600000]])
    def test_block_stays_inside_context(self):
        with self.assertRaises(ValueError):selected_intervals(dict(self.c,width=3100000),3000000,[])
    def test_annotation_is_not_shifted_or_stretched(self):
        a=np.array([[100,300],[700,900]])
        np.testing.assert_array_equal(selected_intervals(dict(self.c,target='annotated'),1000,a),a)
    def test_physical_interval_averaging(self):
        np.testing.assert_allclose(bin_piecewise([0,10,30],[100,200],[0,20,30]),[150,200])
    def test_truth_not_rescaled_by_evaluator(self):
        y=np.array([1000.,10000.]);m=np.array([True,True])
        self.assertAlmostEqual(loss(y*10,y,m,m),np.log(10))
    def test_fresh_seeds_disjoint(self):
        a={seed_for('train',99,'locus',r) for r in range(3)}
        b={seed_for('validation',99,'locus',r) for r in range(24)}
        self.assertFalse(a&b)

if __name__=='__main__':unittest.main()
