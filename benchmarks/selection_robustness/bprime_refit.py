"""Refit the published single-feature Buffalo--Kern likelihood, with audits.

For one annotation, beta_j = mu*W_j is an exact reparameterization. The
binomial log likelihood is concave in log(pi0), beta. Bounds on mu become
bounds on sum(beta), allowing a transparent constrained convex fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from common import digest, write_json


class BprimeFit:
    def __init__(self, data):
        self.data=data
        self.C=np.asarray(data['coeff'])*1e-8
        self.X=np.column_stack([np.ones(len(self.C)),-self.C])
        theta=data['theta']
        self.published_x=np.r_[np.log(theta[0]),theta[1]*theta[2:]/1e-8]
        self.lo_mu,self.hi_mu=10**data['log10_mu_bounds']/1e-8
        self.lo_pi,self.hi_pi=np.log(10**data['log10_pi0_bounds'])
        self.bounds=Bounds(np.r_[self.lo_pi,np.zeros(self.C.shape[1])],np.r_[self.hi_pi,np.full(self.C.shape[1],self.hi_mu)])
        self.constraint=LinearConstraint(np.r_[0,np.ones(self.C.shape[1])][None,:],self.lo_mu,self.hi_mu)
        self.anchor_eta=self.X@self.published_x
        self.anchor_p=np.exp(self.anchor_eta)

    def objective(self,x,Y,gradient=True):
        eta=self.X@x;p=np.exp(eta);same,diff=Y.T
        scale=Y[:,1].sum()
        if not gradient:return float(-(diff@eta+same@np.log1p(-p)))
        # Remove an x-independent constant stably: local perturbations are tiny
        # relative to the genome-wide likelihood. This preserves the gradient.
        delta=eta-self.anchor_eta
        loss=-(diff@delta+same@np.log1p(-self.anchor_p*np.expm1(delta)/(1-self.anchor_p)))/scale
        grad=-(self.X.T@(diff-same*p/(1-p)))/scale
        return loss,grad

    def fit(self,Y,start=None):
        result=minimize(self.objective,self.published_x if start is None else start,args=(Y,),
                        jac=True,method='SLSQP',bounds=self.bounds,constraints=self.constraint,
                        options={'ftol':1e-15,'maxiter':2000})
        if not result.success:
            raise RuntimeError(f'Optimizer failed: {result.message}')
        x=result.x;B=np.exp(-self.C@x[1:])
        if x[1:].min() < -1e-10 or not self.lo_mu-1e-7<=x[1:].sum()<=self.hi_mu+1e-7:
            raise ValueError('Parameter constraints violated')
        return dict(x=x,B=B,pi0=float(np.exp(x[0])),mu=float(x[1:].sum()*1e-8),
                    nll=self.objective(x,Y,gradient=False),iterations=int(result.nit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--published',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    z=np.load(a.published);model=BprimeFit(z)
    B=np.exp(-model.C@model.published_x[1:])
    np.testing.assert_allclose(B,z['Bprime'],rtol=1e-13)
    nll=model.objective(model.published_x,z['Y'],gradient=False)
    provenance=json.loads(a.published.with_name('provenance.json').read_text())
    np.testing.assert_allclose(nll,provenance['published_nll'],rtol=1e-12)
    result=model.fit(z['Y'])
    # Objective and gradient checked against independent finite differences.
    x=model.published_x.copy();loss,grad=model.objective(x,z['Y']);numeric=[]
    for j in range(len(x)):
        step=1e-5;plus=x.copy();minus=x.copy();plus[j]+=step;minus[j]-=step
        numeric.append((model.objective(plus,z['Y'])[0]-model.objective(minus,z['Y'])[0])/(2*step))
    np.testing.assert_allclose(grad,numeric,rtol=2e-5,atol=1e-8)
    assert result['nll'] <= nll+1
    # Recover a separately generated expectation under known parameters.
    total=z['Y'].sum(axis=1);synthetic_pi=np.exp(model.X@model.published_x)
    expected=np.c_[total*(1-synthetic_pi),total*synthetic_pi]
    displaced=model.published_x.copy();displaced[0]+=np.log(1.1)
    displaced[1:]=displaced[1:][::-1]
    recovery=model.fit(expected,start=displaced)
    np.testing.assert_allclose(recovery['B'],B,rtol=2e-4,atol=1e-6)
    np.testing.assert_allclose(np.exp(model.X@recovery['x']),synthetic_pi,rtol=2e-4,atol=1e-9)
    a.out.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(a.out/'refit.npz',**{k:v for k,v in result.items() if k in ['x','B']})
    write_json(a.out/'validation.json',dict(published_prediction_parity=True,analytic_gradient_check=True,
        known_parameter_expected_data_recovery=True,published_nll_parity=True,
        recovery_max_B_error=float(np.max(np.abs(recovery['B']-B))),published_nll_recomputed=nll,
        refitted_nll=result['nll'],relative_nll_improvement=(nll-result['nll'])/nll,
        maximum_B_change=float(np.max(np.abs(result['B']-B))),
        refit_pi0=result['pi0'],refit_mu=result['mu'],source_sha256=digest(__file__),
        input_sha256=digest(a.published),note='Exact single-feature likelihood and bounds; optimizer differs from author implementation'))
    print(json.dumps(json.loads((a.out/'validation.json').read_text())))


if __name__=='__main__':main()
