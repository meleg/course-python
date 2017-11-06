"""
module for defining the nep classes
"""
# clear & clear & python -m pytest test/nep_classes_test.py
# testing mode

import numpy as np
import types

matrix_type=type(np.zeros((1,1)))

# exeptions
class NotMatrix(Exception):
    pass

class NotFunction(Exception):
    pass

class NotTensor(Exception):
    pass

class nep:
    """
    nep is the main class for nonlinear eigenvalue problems with the following inputs:
    -Meval is the function evaluation for Meval(l)=M(l)
    -Md is the function that computes the derivatives in zeros Md(j)=M^(j)(0)
    -(optional) Mpeval is the function evaluation for the derivative Mpeval(l)=M'(l)
    """
    def __init__(self, Meval, Md, Mpeval=None):
        if type(Md)!=types.FunctionType:
            raise NotFunction
        if type(Meval)!=types.FunctionType:
            raise NotFunction
        if type(Meval(0))!=matrix_type:
            raise NotMatrix
        if type(Md(0))!=matrix_type:
            raise NotMatrix
        if Mpeval!=None:
            if type(Mpeval)!=types.FunctionType:
                raise NotFunction
        self.Meval=Meval
        self.Md=Md
        self.Mpeval=Mpeval
        self.n,self.n=Md(0).shape

class pep(nep):
    """
    pep is a subclass of nep with as
    input a three dimensional array
    containing the coefficients.

    coeff:  is a three dimensional array containing
            the coefficients defining the polynomial
            eigenvalue problem

    companion:  is a function that provides as output
                the standard companion linearization
                of the polynomial eigenvalue problem
    """
    def __init__(self, coeff):
        if not(type(coeff)==matrix_type):
            raise NotMatrix
        if not(coeff.ndim==3):
            raise NotTensor

        n,n,d=coeff.shape
        def companion():
            B=np.zeros(((d-1)*n,(d-1)*n))
            for j in range(1,d):
                B[0:n,(j-1)*n:j*n]=coeff[:,:,j]
            for j in range(1,d-1):
                B[j*n:(j+1)*n,(j-1)*n:j*n]=np.eye(n)
            A=np.eye((d-1)*n)
            A[0:n,0:n]=-coeff[:,:,0]
            return A,B

        self.coeff=coeff
        self.d=d
        self.n=n
        self.companion=companion
