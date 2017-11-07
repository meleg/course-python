import nep_classes
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math
import nep_solvers

# GENERATE A POLYNOMIAL EIGENVALUE PROBLEM
# generate coefficients matrix
d=10; n=20; coeff=np.zeros((n,n,d))
for i in range(10):
    coeff[:,:,i]=np.random.random((n,n));
# define the polynomial eigenvalue problem
P=nep_classes.pep(coeff)

# test that P(1) is the sum of the coefficients
def test_pep_sum_coefficients():
    P1_sum=np.zeros((n,n));
    for i in range(10):
        P1_sum+=coeff[:,:,i];
    Q=nep_solvers.pep2nep(P)
    assert npla.norm(Q.Meval(1)-P1_sum)<1e-6

# GENERATE A DELAY EIGEVALUE PROBLEM
# generate coefficients matrix
A0=np.random.random((n,n)); A1=np.random.random((n,n))

# function evaluation
def Meval_dep(l):
    return -l*np.eye(n)+A0+A1*np.exp(-l)
# first derivative
def Mpeval_dep(l):
    return -np.eye(n)-A1*np.exp(-l)
# high order derivatives in zero
def Md_dep(i):
    if i==0:
        return A0+A1
    elif i==1:
        return -np.eye(n)-A1
    else:
        return ((-1)**i)*A1

# create the nonlinear eigenvalue problem
M_dep=nep_classes.nep(Meval_dep, Md_dep, Mpeval_dep)

# check several known equalities for the dep
def test_dep_approx_pep():
    assert npla.norm(M_dep.Meval(0)-A0-A1)<1e-5
    assert npla.norm(M_dep.Meval(1)+np.eye(n)-A0-A1*math.exp(-1))<1e-5
    assert npla.norm(M_dep.Md(1)-M_dep.Mpeval(0))<1e-5

# convert the polynomial eigenvalue problem to a nep
P_nep=nep_solvers.pep2nep(M_dep)

# check several known equalities (similar to the previous tests)
def test_pep2nep_conv():
    assert npla.norm(P_nep.Meval(0)-A0-A1)<1e-5
    assert npla.norm(P_nep.Meval(1)+np.eye(n)-A0-A1*math.exp(-1))<1e-5
    assert npla.norm(P_nep.Md(1)-M_dep.Mpeval(0))<1e-5
