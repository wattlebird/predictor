from numpy.linalg import solve
import numpy as np

def update_U(r, u, V, bu, bm, avg, C):
    """
    r: records of user i, sparse array
    u: U[i,:]
    bu: scalar, bu[i]
    """
    if r.nonzero()[1].shape[0]==0:
        return u

    A = C*np.eye(V.shape[1], dtype=np.float)
    b = np.zeros(V.shape[1], dtype=np.float)
    for j in r.nonzero()[1]:
        A += np.outer(V[j,:], V[j,:])
        b += (r[0, j] - avg - bu - bm[j])*V[j,:]
    return solve(A, b)

def update_V(r, U, v, bu, bm, avg, C):
    """
    j: current index to be updated
    r: records of item j, sparse array
    v: V[j,:]
    bm: bm[j]
    """
    if r.nonzero()[0].shape[0]==0:
        return v

    A = C*np.eye(U.shape[1], dtype=np.float)
    b = np.zeros(U.shape[1], dtype=np.float)
    for i in r.nonzero()[0]:
        A += np.outer(U[i,:], U[i,:])
        b += (r[i, 0] - avg - bu[i] - bm)*U[i,:]
    return solve(A, b)

def update_bu(r, u, V, bu, bm, avg, C):
    """
    r: records of user i, sparse array
    u: U[i,:]
    bu: bu[i]
    """
    if r.nonzero()[1].shape[0]==0:
        return bu

    a = C+r.getnnz()
    b = 0
    for j in r.nonzero()[1]:
        b += (r[0, j] - avg - bm[j] - np.dot(u, V[j, :]))
    return np.divide(b, a)

def update_bm(r, U, v, bu, bm, avg, C):
    """
    r: records of item j, sparse array
    v: V[j,:]
    bm: bm[j]
    """
    if r.nonzero()[0].shape[0]==0:
        return bm

    a = C+r.getnnz()
    b = 0
    for i in r.nonzero()[0]:
        b += (r[i, 0] - avg - bu[i] - np.dot(U[i, :], v))
    return np.divide(b, a)
