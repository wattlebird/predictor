from random import shuffle
from scipy.sparse import coo_matrix
import numpy as np

def riter(M, p, n):

    data = M.data
    row = M.row
    col = M.col

    cutoff = int(data.shape[0]*p)
    idx = range(data.shape[0])
    for i in xrange(n):
        shuffle(idx)
        Mc = coo_matrix((np.asarray(data)[idx[:cutoff]],
                        (np.asarray(row)[idx[:cutoff]], np.asarray(col)[idx[:cutoff]])),
                            dtype=np.float, shape=M.shape)
        yield Mc
