import nep_classes
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math
import nep_solvers

# GENERATE A DELAY EIGEVALUE PROBLEM
# generate coefficients matrix
n=10
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

# approximate the delay eigenvalue problem with a
# polynomial eigenvalue problem and test
# known equalities
P_dep=nep_solvers.generate_pep_approximation(M_dep,10)
def test_poly_approx():
    assert npla.norm(P_dep.coeff[:,:,0]-A0-A1)<1e-5
    assert npla.norm(P_dep.coeff[:,:,1]+np.eye(n)+A1)<1e-5
    assert npla.norm(P_dep.coeff[:,:,2]-A1/2)<1e-5
    assert npla.norm(P_dep.coeff[:,:,3]+A1/6)<1e-5
    assert npla.norm(P_dep.coeff[:,:,4]-A1/24)<1e-5

# convert the polynomial eigenvalue problem to a nep
P_dep2pep2nep=nep_solvers.pep2nep(P_dep)

# check several known equalities (similar to the previous tests)
def test_pep2nep_conv():
    assert npla.norm(P_dep2pep2nep.Meval(0)-A0-A1)<1e-5
    assert npla.norm(P_dep2pep2nep.Meval(1)+np.eye(n)-A0-A1*math.exp(-1))<1e-5
    assert npla.norm(P_dep2pep2nep.Md(1)-M_dep.Mpeval(0))<1e-5
