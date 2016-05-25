from numpy.linalg import norm
from scipy.sparse import issparse
from sklearn.externals.joblib import Parallel, delayed
import numpy as np

def _row_loss_dense(r, u, V, bu, bm, avg):
    sqerr=0
    for j in xrange(V.shape[0]):
        if r[j]==0:continue;
        sqerr += (r[j] - np.dot(u, V[j,:]) - bu - bm[j] - avg)**2
    return sqerr

def _row_loss_sparse(r, u, V, bu, bm, avg):
    sqerr=0
    for j in r.nonzero()[1]:
        sqerr += (r[0, j] - np.dot(u, V[j,:]) - bu - bm[j] - avg)**2
    return sqerr

def rmse(R, U, V, bu, bm, u):
    """
    R: nu x nm, sparse matrix
    U: nu x k
    V: nm x k
    bu: nu x 1
    bm: nm x 1
    u: scalar
    """
    if issparse(R):
        Rc = R.tocsr()
        sqerr = sum(Parallel(n_jobs=-1)(delayed(_row_loss_sparse)(Rc.getrow(i), U[i,:], V, bu[i], bm, u) for i in xrange(Rc.shape[0])))
        sqerr /= Rc.getnnz()
    else:
        sqerr = sum(Parallel(n_jobs=-1)(delayed(_row_loss_dense)(R[i,:], U[i,:], V, bu[i], bm, u) for i in xrange(R.shape[0])))
        sqerr /= np.count_nonzero(R)

    return np.sqrt(sqerr)

def loss(R, U, V, bu, bm, u, C):
    """
    R: nu x nm, sparse matrix
    U: nu x k
    V: nm x k
    bu: nu x 1
    bm: nm x 1
    u: scalar
    C: scalar
    """
    loss = rmse(R, U, V, bu, bm, u)
    if issparse(R):
        loss = loss**2 * R.getnnz()
    else:
        loss = loss**2 * np.count_nonzero(R)
    loss += C*norm(bu)
    loss += C*norm(bm)
    loss += C*norm(U, ord='fro')
    loss += C*norm(V, ord='fro')
    return loss
