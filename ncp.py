"""
# ncp: nonnegative tensor decomposition (CANDECOMP/PARAFAC) by block-coordinate update
#  min 0.5*||M - A_1\circ...\circ A_N||_F^2  
#  subject to A_1>=0, ..., A_N>=0
#
# input: 
#       M: input nonnegative tensor
#       r: estimated rank (each A_i has r columns); require exact or moderate overestimates
#       opts.
#           tol: tolerance for relative change of function value, default: 1e-4
#           maxit: max number of iterations, default: 500
#           maxT: max running time, default: 1e3
#           rw: control the extrapolation weight, default: 1
#           A0: initial point in cell struct, default: Gaussian random
#           matrices
# output:
#       A: nonnegative ktensor
#       Out.
#           iter: number of iterations
#           hist_obj: history of objective values
#           hist_rel: history of relative objective changes (row 1) and relative residuals (row 2)
#
# require MATLAB Tensor Toolbox from
# http://www.sandia.gov/~tgkolda/TensorToolbox/
#
# More information can be found at:
# http://www.caam.rice.edu/~optimization/bcu/
"""

from copy import deepcopy
import time
import warnings
import numpy as np
from tensorly.cp_tensor import unfolding_dot_khatri_rao


def ncp(M, r, tol=1e-4, maxit=500, maxT=1e+6, rw=1, verbose=False):

    # Data preprocessing and initialization

    N       = M.ndim  # M is an N-way tensor
    M_shape = M.shape  # dimensions of M
    M_norm  = np.linalg.norm(M)  # norm of M
    obj0    = .5 * M_norm ** 2  # initial objective value

    if verbose == True:
        print('N=', N)
        print('M_shape=', M_shape)
        print('M_norm=', M_norm)
        print('obj0=', obj0)
        print()

    # initial tensor factors
    A0, Asq = [], []

    for m in M_shape:
        # randomly generate each factor
        A0m = np.maximum(np.zeros((m, r)), np.random.rand(m, r))
        A0.append(A0m)
        # normalize A0 and cache its square
        # fro: Frobenius norm
        A0m /= np.linalg.norm(A0m, ord='fro') * M_norm ** (1 / N)
        Asqm = A0m.T @ A0m
        Asq.append(Asqm)

    Am = deepcopy(A0)
    A  = deepcopy(A0)

    nstall = 0  # of stalled iterations
    t0 = 1  # used for extrapolation weight update
    wA = np.ones((N, 1))  # extrapolation weight aaray
    L0 = np.ones((N, 1))  # Lipschitz constant array
    L  = np.ones((N, 1))

    # Store data?

    ### Iterations of block-coordinate update

    # iteratively updated variables
    # =============================
    #   Gn: gradients with respect to A[n]
    #   A: new updates
    #   A0: old updates
    #   Am: extrapolations of A
    #   L, L0: current and previous Lipshitz bounds
    #   obj, obj0: current and previous objective values

    start_time = time.process_time()

    for k in range(maxit):

        for n in range(N):
            Bsq = np.ones(r)
            for i in range(N):
                if not i == n:
                    Bsq = Bsq * Asq[i]  # element-wise product
            
            L0[n] = L[n]  # caution!!
            L[n]  = np.linalg.norm(Bsq)  # gradient Lipschitz constant

            # Here, not using stored data in the original code
            MB = unfolding_dot_khatri_rao(M, (None, A), n)

            # compute the gradient
            Gn = Am[n] @ Bsq - MB
            A[n] = np.maximum(np.zeros(Am[n].shape), Am[n] - Gn / L[n])
            Asq[n] = A[n].T @ A[n]

        obj = .5 * (
            np.sum(np.sum(Asq[-1] * Bsq))
            - 2 * np.sum(np.sum(A[-1] * MB))
            + M_norm ** 2
        )

        relerr1 = np.abs(obj - obj0) / (obj0 + 1)
        relerr2 = (2 * obj) ** .5 / M_norm

        # check stopping criterion
        if relerr1 < tol:
            break
        if nstall >= 3 or relerr2 < tol:
            break
        if time.process_time() - start_time > maxT:
            warnings.warn("Time over")
            break

        # correction and extrapolation
        t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2

        if obj >= obj0:
            Am = A0

        else:
            # apply extrapolation
            w = (t0 - 1) / t
            for n in range(N):
                wA[n] = np.minimum(w, rw * L0[n] / L[n])
                Am[n] = A[n] + wA[n] * (A[n] - A0[n])
            
            A0 = A
            t0 = t
            obj0 = obj

    return A  # nonnegative k-tensor
