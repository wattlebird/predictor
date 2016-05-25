from sklearn.externals.joblib import Parallel, delayed
import numpy as np

L=0.0001 # with laplace smoothing


def _bias_movie(c, bu, u):
    b = 0
    for i in c.nonzero()[0]:
        b += c[i, 0] - u - bu[i]
    return b/(c.getnnz()+L)

def _bias_user(c, bm, u):
    b = 0
    for j in c.nonzero()[1]:
        b += c[0, j] - u - bm[j]
    return b/(c.getnnz()+L)

def bias_user(R, avg, bm=None):
    Rc = R.tocsr()
    if bm is None:
        bu = np.zeros(Rc.shape[0])
        for i in xrange(Rc.shape[0]):
            r = Rc.getrow(i)
            bu[i] = r.sum()/(r.getnnz()+L) - avg*r.getnnz()/(r.getnnz()+L)
    else:
        bu = np.hstack(Parallel(n_jobs=-1)(delayed(_bias_user)(Rc.getrow(i), bm, avg) for i in xrange(R.shape[0])))
    return bu

def bias_movie(R, avg, bu=None):
    Rc = R.tocsc()
    if bu is None:
        bm = np.zeros(Rc.shape[1])
        for i in xrange(Rc.shape[1]):
            c = Rc.getcol(i)
            bm[i] = c.sum()/(c.getnnz()+L) - avg*c.getnnz()/(c.getnnz()+L)
    else:
        bm = np.hstack(Parallel(n_jobs=-1)(delayed(_bias_movie)(Rc.getcol(j), bu, avg) for j in xrange(R.shape[1])))
    return bm
