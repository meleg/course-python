#import sys
#import os.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import nep_classes
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math
import nep_solvers

n=40
A0=np.random.random((n,n))
A1=np.random.random((n,n))

def Meval_dep(l):
    return -l*np.eye(n)+A0+A1*np.exp(-l)
def Mdd_dep(i):
    if i==0:
        return A0+A1
    elif i==1:
        return -np.eye(n)-A1
    else:
        return ((-1)**i)*A1
M_dep=nep_classes.nep(Meval_dep, Mdd_dep)

# check that the iar and the companion method give at least
# one consistent approximation of one eigenvalue
def test_dep_iar_companion_comparison():
    ll, _=nep_solvers.iar(M_dep,50)
    P=nep_solvers.generate_pep_approximation(M_dep,10)
    l,_=nep_solvers.companinon_solver(P)
    diff_iar_comp=abs(l[np.argmin(abs(l))]-ll[np.argmin(abs(ll))])
    assert diff_iar_comp<1e-5, \
    "iar and companion based linearization give different approximations"


# check that the residual inverse iteration improve the approximation of the
# worse eigenpair-approximation computed by iar
def test_res_inv():
    ll,vv=nep_solvers.iar(M_dep,50)
    # extract the worse eigenpair approximation
    k=ll.shape[0]-1;    vv=vv[:,k:k+1]
    print("size",vv.shape)

    ll=ll[k];
    err_iar=npla.norm(M_dep.Meval(ll).dot(vv))
    l,v=nep_solvers.res_inv(M_dep,sigma=ll,n_iter=10)
    err_res_inv=npla.norm(M_dep.Meval(l).dot(v))

    print("error res_inv",err_res_inv)
    print("error iar",err_iar)
    assert err_res_inv<err_iar
