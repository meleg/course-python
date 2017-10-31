"""
module for defining the nep classes
"""

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
    pep is a subclass of nep with as input a three dimensional array containing the coefficients
    """
    def __init__(self, coeff):
        if not(type(coeff)==matrix_type):
            raise NotMatrix
        if not(coeff.ndim==3):
            raise NotTensor
        self.coeff=coeff
        n,n,d=coeff.shape
        self.d=d
        self.n=n

if __name__ == "__main__":
    n=2;
    A=np.random.random((n,n))
    B=np.random.random((n,n))
    C=np.random.random((n,n))

    def myM(l):
        return A*l*l+B*l+C

    def myMd(l):
        return A

    Mynep=nep(myM, myMd)
    print(Mynep.Meval(2))
    print(Mynep.Md(1))

    d=3;
    coeff=np.random.random((n,n,d))
    myP=pep(coeff)
    print(myP.coeff[:,:,0])
    print(myP.coeff[:,:,1])
    print(myP.coeff[:,:,2])
