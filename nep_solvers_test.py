# import nep_classes
# import numpy as np
# import scipy.linalg as la
# import numpy.linalg as npla
# import matplotlib.pyplot as plt
# import math
# import nep_solvers
#
# n=50;
# A0=np.random.random((n,n))
# A1=np.random.random((n,n))
#
# def Meval_dep(l):
#     return -l*np.eye(n)+A0+A1*np.exp(-l)
# def Mdd_dep(i):
#     if i==0:
#         return A0+A1
#     elif i==1:
#         return -np.eye(n)-A1
#     else:
#         return ((-1)**i)*A1
#
# M_dep=nep_classes.nep(Meval_dep, Mdd_dep)

# def test_dep_iar_companion_comparison():
#     ll, _=nep_solvers.iar(M_dep,50)
#     ll=np.asarray(ll)
#     P=nep_solvers.generate_pep_approximation(M_dep,10)
#     l,_=nep_solvers.companinon_solver(P)
#     assert abs(l[np.argmin(abs(l))]-ll[np.argmin(abs(ll))])<1e-5, \
#     "iar and companion based linearization give different approximations"

def test_1():
    assert True
