import unittest
import numpy as np

from bprime_refit import BprimeFit


class RefitTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(81)
        self.model=BprimeFit(dict(coeff=rng.uniform(.01,.5,size=(100,3))/1e-8,
            theta=np.array([.001,2e-8,.2,.3,.5]),
            log10_mu_bounds=np.log10([5e-9,8e-8]),
            log10_pi0_bounds=np.log10([.0005,.008])))
        self.p=np.exp(self.model.X@self.model.published_x)
        self.Y=np.c_[1e7*(1-self.p),1e7*self.p]

    def test_raw_likelihood_matches_independent_binomial(self):
        actual=self.model.objective(self.model.published_x,self.Y,gradient=False)
        expected=-np.sum(self.Y[:,1]*np.log(self.p)+self.Y[:,0]*np.log1p(-self.p))
        self.assertAlmostEqual(actual/expected,1,places=13)

    def test_recovery_from_displaced_parameters(self):
        x=self.model.published_x.copy();x[0]+=.1;x[1:]=x[1:][::-1]
        fit=self.model.fit(self.Y,start=x)
        np.testing.assert_allclose(fit['x'],self.model.published_x,atol=2e-6)

    def test_data_changes_fit_but_not_fixed_prediction(self):
        fit=self.model.fit(self.Y)
        Y=self.Y.copy();Y[:40,1]*=.5;Y[:40,0]=1e7-Y[:40,1]
        changed=self.model.fit(Y)
        self.assertGreater(np.max(np.abs(changed['B']-fit['B'])),.01)
        np.testing.assert_array_equal(np.exp(self.model.X@self.model.published_x),self.p)

    def test_stable_objective_has_correct_gradient(self):
        x=self.model.published_x.copy();x[1]+=.05
        _,gradient=self.model.objective(x,self.Y)
        numeric=[]
        for j in range(len(x)):
            d=np.eye(len(x))[j]*1e-5
            numeric.append((self.model.objective(x+d,self.Y)[0]-self.model.objective(x-d,self.Y)[0])/2e-5)
        np.testing.assert_allclose(gradient,numeric,atol=1e-10,rtol=1e-6)


if __name__=='__main__':unittest.main()
