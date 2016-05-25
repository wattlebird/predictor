from scipy.sparse import coo_matrix
from sklearn.cross_validation import StratifiedKFold
from predictor.bias import *
from scipy.sparse.linalg import svds
from predictor.generator import riter
from predictor.eva import rmse
from predictor.altmin.util import *
import numpy as np
import argparse

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

def run(Cf, Cbu, Cbm, T, k):
    train_avg_rmse=0
    test_avg_rmse=0
    kf = StratifiedKFold(row, shuffle=True, n_folds=5)
    for fold, (train, test) in enumerate(kf):
        print "Fold {0} now start. Preparing for training data...".format(fold+1)
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
        u, s, vt = svds(R_init, k=k)
        U = u.copy()
        V = vt.T

        total = T*2+1
        Rv = []
        for i, Ra in enumerate(riter(R_train, 0.5, total)):
            Rv.append(Ra)

        print "Now begin training."

        for t in xrange(1,T+1):
            rc = Rv[2*t-1]
            Rc = rc.tocsc()
            # get updated V
            #print "Before update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))
            V = np.vstack(Parallel(n_jobs=-1)(delayed(update_V)(Rc.getcol(j), U, V[j,:], bu, bm[j], avg, Cf) for j in xrange(Rc.shape[1])))
            #print "After update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))
            bm = np.hstack(Parallel(n_jobs=-1)(delayed(update_bm)(Rc.getcol(j), U, V[j,:], bu, bm[j], avg, Cbm) for j in xrange(Rc.shape[1])))
            #print "After update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))

            rc=Rv[2*t]
            Rc = rc.tocsr()
            # get updated U

            #print "Before update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))
            U = np.vstack(Parallel(n_jobs=-1)(delayed(update_U)(Rc.getrow(i), U[i,:], V, bu[i], bm, avg, Cf) for i in xrange(Rc.shape[0])))
            #print "After update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))
            bu = np.hstack(Parallel(n_jobs=-1)(delayed(update_bu)(Rc.getrow(i), U[i,:], V, bu[i], bm, avg, Cbu) for i in xrange(Rc.shape[0])))
            #print "After update loss:{0}".format(loss(Rc, U, V, bu, bm, avg, C))
            #print loss
            print "In iteration {0}, we have rmse={1} on training set, rmse={2} on test set.".format(
                t, rmse(R_train, U, V, bu, bm, avg), rmse(R_test, U, V, bu, bm, avg))
        train_avg_rmse += rmse(R_train, U, V, bu, bm, avg)
        test_avg_rmse += rmse(R_test, U, V, bu, bm, avg)
    print "Overall rmse on training set: {0}; on test set: {1}".format(train_avg_rmse/5, test_avg_rmse/5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train predicting model using AltMinComplete.')
    parser.add_argument("--turns", type=int, default=12, dest="T")
    parser.add_argument("--cf", type=float, default=10, dest="Cf")
    parser.add_argument("--cbu", type=float, default=1, dest="Cbu")
    parser.add_argument("--cbm", type=float, default=1, dest="Cbm")
    parser.add_argument("--k", type=int, default=50)

    args = parser.parse_args()
    run(args.Cf, args.Cbu, args.Cbm, args.T, args.k)
