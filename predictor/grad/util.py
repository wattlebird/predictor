import numpy as np

def gred_U(r, u, V, bu, bm, avg, C):
    """
    r: a sparse row of user i
    u: row i
    """
    du = 2*C*u
    if r.getnnz()==0:
        return du
    d = 0
    for j in r.nonzero()[1]:
        d += r[0, j]-avg-bu-bm[j]-np.dot(u, V[j,:])
    du -= 2*d*V[j,:]
    return du

def gred_V(r, U, v, bu, bm, avg, C):
    """
    r: a sparse col of movie j
    v: movie j
    """
    dv = 2*C*v
    if r.getnnz()==0:
        return dv
    d = 0
    for i in r.nonzero()[0]:
        d += r[i, 0]-avg-bu[i]-bm-np.dot(U[i,:], v)
    dv -= 2*d*U[i,:]
    return dv

def gred_bu(r, u, V, bu, bm, avg, C):
    """
    r: a sparse row of user i
    u: row i
    """
    dbu = 2*C*bu
    if r.getnnz()==0:
        return dbu
    d = 0
    for j in r.nonzero()[1]:
        d += r[0, j]-avg-bu-bm[j]-np.dot(u, V[j,:])
    dbu -= 2*d
    return dbu

def gred_bm(r, U, v, bu, bm, avg, C):
    dbm = 2*C*bm
    if r.getnnz()==0:
        return dbm
    d = 0
    for i in r.nonzero()[0]:
        d += r[i, 0]-avg-bu[i]-bm-np.dot(U[i,:], v)
    dbm -= 2*d
    return dbm
