from scipy.sparse import coo_matrix
from sklearn.cross_validation import StratifiedKFold
from predictor.bias import *
from scipy.sparse.linalg import svds
from predictor.grad.util import *
from sklearn.externals.joblib import delayed, Parallel
import numpy as np
from random import shuffle
from predictor.eva import *

def batch_update(parallel, data, row, col):
    U_ = U[row, :]
    V_ = V[col, :]
    bu_ = bu[row]
    bm_ = bm[col]


    du = parallel(delayed(gred_u)(data[i], U_[i,:], V_[i,:], bu_[i], bm_[i], avg, C) for i in xrange(len(data)))
    dv = parallel(delayed(gred_v)(data[i], U_[i,:], V_[i,:], bu_[i], bm_[i], avg, C) for i in xrange(len(data)))
    dbu = parallel(delayed(gred_bu)(data[i], U_[i,:], V_[i,:], bu_[i], bm_[i], avg, C) for i in xrange(len(data)))
    dbm = parallel(delayed(gred_bm)(data[i], U_[i,:], V_[i,:], bu_[i], bm_[i], avg, C) for i in xrange(len(data)))
    if method=='sgd':
        for i in xrange(len(data)):
            U_[i,:] -= eta*du[i]
            V_[i,:] -= eta*dv[i]
            bu_[i] -= eta*dbu[i]
            bm_[i] -= eta*dbm[i]
        for c, i in enumerate(row):
            U[i,:]=U_[c,:]
            bu[i]=bu_[c]
        for c, j in enumerate(col):
            V[j,:]=V_[c,:]
            bm[j]=bm_[c]
    elif method=='adagrad':
        for c, i in enumerate(row):
            gdu[i] += np.dot(du[c], du[c])
            gdbu[i] += np.dot(dbu[c], dbu[c])
            U[i,:]-=eta*du[c]/sqrt(gdu[i]+epislon)
            bu[i]-=eta*dbu[c]/sqrt(gdbu[i]+epislon)
        for c, j in enumerate(col):
            gdv[j] += np.dot(dv[c], dv[c])
            gdbm[j] += np.dot(dbm[c], dbm[c])
            V[j,:]-=eta*dv[c]/sqrt(gdv[i]+epislon)
            bm[j]-=eta*dbm[c]/sqrt(gdbm[i]+epislon)

nu=6040
nm=3952

data = [0]*1000209
row = [0]*1000209
col = [0]*1000209
with open("ml-1m/ratings.dat", "r") as fr:
    recs = fr.readlines()
    for i, rec in enumerate(recs):
        rec = rec.strip()
        rec = [int(item) for item in rec.split("::")]
        data[i] = rec[2]
        row[i] = rec[0]-1
        col[i] = rec[1]-1

kf = StratifiedKFold(row, shuffle=True, n_folds=5)
for train, test in kf:
    pass


R_train = coo_matrix((np.asarray(data)[train],
                     (np.asarray(row)[train], np.asarray(col)[train])),
                     dtype=np.float, shape=(nu, nm))
R_test = coo_matrix((np.asarray(data)[test],
                     (np.asarray(row)[test], np.asarray(col)[test])),
                     dtype=np.float, shape=(nu, nm))

avg = np.sum(R_train.data)/R_train.nnz
bu = bias_user(R_train, avg)
bm = bias_movie(R_train, avg, bu)

R_init = R_train.copy()
for e, (i, j) in enumerate(zip(R_train.row, R_train.col)):
    R_init.data[e]-=bu[i]+bm[j]+avg


u, s, vt = svds(R_init, k=50)
U = u.copy()
V = vt.T

method = 'sgd'
eta = 0.002
C=0.01

total=6
batchsize=1024
idx=range(R_train.getnnz())

for t in xrange(1, total+1):
    shuffle(idx)
    b=0
    with Parallel(n_jobs=8) as parallel:
        while b<R_train.getnnz():
            e=min(R_train.getnnz(), b+batchsize)
            batch_update(parallel, R_train.data[idx[b:e]], R_train.row[idx[b:e]], R_train.col[idx[b:e]])
            b+=batchsize
    print "In iteration {0}, we have rmse {1} on training set, rmse {2} on test set.".format(
        t, rmse(R_train, U, V, bu, bm, avg), rmse(R_test, U, V, bu, bm, avg))