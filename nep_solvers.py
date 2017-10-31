# module for solving nonlinear eigenvalue problems
# testing mode
# clear & clear & python -m pytest test_dep.py

import nep_classes
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla
import matplotlib.pyplot as plt
import math

# exeptions
class NotNep(Exception):
    pass

class NotPep(Exception):
    pass

def companinon_solver(P):
    """
    Computes eigenvalues of a pep using the standard companion linearization
    """

    if not(issubclass(type(P),nep_classes.nep)):
        raise NotNep

    if type(P)!=nep_classes.pep:
        print("Convert the nep to a pep with generate_pep_approximation before using this function")
        raise NotPep

    n=P.n; d=P.d
    B=np.zeros(((d-1)*n,(d-1)*n))
    for j in range(1,d):
        B[0:n,(j-1)*n:j*n]=P.coeff[:,:,j]
    for j in range(1,d-1):
        B[j*n:(j+1)*n,(j-1)*n:j*n]=np.eye(n)

    A=np.eye((d-1)*n)
    A[0:n,0:n]=-P.coeff[:,:,0]

    return la.eig(A,B)

def res_inv(MM,sigma=0,n_iter=10,verbose=0):
    """
    Compute the eigenvalues closest to sigma of a nep using residual inverse iteration (res_inv) with n_iter iterations
    """

    if not(issubclass(type(MM),nep_classes.nep)):
        raise NotNep

    # if the input is a pep, convert it to a nep
    if type(MM)==nep_classes.pep:
        MM=pep2nep(MM)

    # if the derivative is not provided, estimate it
    if MM.Mpeval==None:
        delta=1e-12
        def Mpeval(l):
            return (MM.Meval(l+delta)-MM.Meval(l))/delta
        MM.Mpeval=Mpeval

    # initialization
    l=sigma
    v=np.random.random((MM.n,1))
    v=v/npla.norm(v)

    Msigma=MM.Meval(sigma)
    for j in range(n_iter):
        # compute the eigenvector iteration
        v=v-npla.solve(Msigma,MM.Meval(l).dot(v))
        v=v/npla.norm(v)
        # compute the eigenvalue iteration (one step of Newton)
        l=l-v.transpose().dot(MM.Meval(l).dot(v))/v.transpose().dot(MM.Mpeval(l).dot(v))
        err=npla.norm(MM.Meval(l).dot(v))
        if verbose==1:
            print("Residual at iteration ",j,"is ",err)
        if err<1e-15:
            break
    return l,v

def iar(MM,m,verbose=0):
    """
    Compute the eigenvalues (closest to zero) of a nep using the Infinite Arnoldi method (iar)
    """

    if not(issubclass(type(MM),nep_classes.nep)):
        raise NotNep

    # if the input is a pep, convert it to a nep
    if type(MM)==nep_classes.pep:
        MM=pep2nep(MM)

    n=MM.n

    # initialization
    V=np.zeros((n*(m+1),m+1))
    V[:n,:1]=np.random.random((n,1))
    V[:n,:1]=V[:n,:1]/npla.norm(V[:n,:1])
    H=np.zeros((m+1,m))

    # run the Arnoldi iterations
    for k in range(m):
        y=np.zeros((n,k+2))
        w=V[:(k+1)*n,k].reshape((k+1,n)).transpose()

        for j in range(k+1):
            y[:,j+1]=w[:,j]/(j+1)

        for s in range(1,k+2):
            y[:,0]+=MM.Md(s).dot(y[:,s])
        y[:,0]=npla.solve(MM.Md(0),-y[:,0])
        y=y.transpose().reshape(((k+2)*n,1))

        # orthogonalization
        H[:k+1,k:k+1]=V[:(k+2)*n,:k+1].transpose().dot(y);
        w=V[:(k+2)*n,:k+1].dot(H[:k+1,k:k+1]); w.shape=((k+2)*n,1)
        y=y-w;

        # normalization
        H[k+1,k]=npla.norm(y);
        V[:(k+2)*n,k+1:k+2]=y/H[k+1,k];

    # orthogonality test
    assert npla.norm( V.T.dot(V)-np.eye(m+1) ) < 1e-6, "loss of orthogonality"

    # compute the Ritz pairs and the convergence history
    err=np.zeros((m,m))
    for k in range(m):
        d,Z=npla.eig(H[:k+1,:k+1])
        W=V[:,:k+1].dot(Z)
        for j in range(k):
            err[k,j]=npla.norm(MM.Meval(1/d[j]).dot(W[:n,j:j+1]))
        err[k,:k]=np.sort(err[k,:k])
    if verbose==1:
        # plot the convergence history
        for i in range(m):
            plt.semilogy(np.arange(i,m), err[i:m,i])

        plt.ylim([1e-16,1e1])
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.show()

    # extract the converged Ritz pairs
    tol=1e-6; eig_val=[]; eig_vec=[];
    for i in range(m):
        if npla.norm(MM.Meval(1/d[i]).dot(W[:n,i:i+1]))<tol:
            eig_val.append(1/d[i])
            eig_vec.append(W[:n,i:i+1])

    eig_val=np.asarray(eig_val)
    #eig_vec=np.asarray(eig_vec)
    return eig_val,eig_vec

def generate_pep_approximation(nep,d):
    """
    Approximate a nep with a pep by truncating a Taylor expansion with d terms
    """
    n=nep.n; coeff=np.zeros((n,n,d))
    for j in range(d):
        coeff[:,:,j]=nep.Md(j)/math.factorial(j)
    return nep_classes.pep(coeff)

def pep2nep(P):
    """
    Convert a pep to a nep
    """
    n,n,d=P.coeff.shape
    def pepMd(j):
        if j<d:
            return P.coeff[:,:,j]*math.factorial(j)
        else:
            return np.zeros((n,n))
    def pepMeval(l):
        MM=np.zeros((n,n))
        for j in range(d):
            MM=MM+(l**j)*P.coeff[:,:,j]
        return MM
    return nep_classes.nep(pepMeval, pepMd)
