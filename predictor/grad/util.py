import numpy as np

def gred_u(r, u, v, bu, bm, avg, C):
    """
    r: data of user i, movie j
    u: row i
    v: col j
    """
    du = 2*C*u
    d = r-avg-bu-bm-np.dot(u, v)
    du -= 2*d*v
    return du

def gred_v(r, u, v, bu, bm, avg, C):
    dv = 2*C*v
    d = r-avg-bu-bm-np.dot(u, v)
    dv -= 2*d*u
    return dv

def gred_bu(r, u, v, bu, bm, avg, C):
    dbu = 2*C*bu
    d = r-avg-bu-bm-np.dot(u, v)
    dbu -= 2*d
    return dbu

def gred_bm(r, u, v, bu, bm, avg, C):
    dbm = 2*C*bm
    d = r-avg-bu-bm-np.dot(u, v)
    dbm -= 2*d
    return dbm

def gred(r, u, v, bu, bm, avg, C):
    return (gred_u(r, u, v, bu, bm, avg, C),
            gred_v(r, u, v, bu, bm, avg, C),
            gred_bu(r, u, v, bu, bm, avg, C),
            gred_bm(r, u, v, bu, bm, avg, C))