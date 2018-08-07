import numpy as np
import copy
from scipy.optimize import brute # grid search
from sklearn.base import BaseEstimator

from computational_utilities import reshape

def fk(Ql, zl, a):
    rv = 0
    N = len(Ql)
    for n in xrange(N):
        rv += np.linalg.norm(np.dot(Ql[n], zl[n]) - a)
    return rv
    
def cobe(Yl, eps = 1e-8, gamma = 1e-8, maxitnum = 20, verbose=False, maxInnerItNum = 500, inform=False, fast=True):
    # QR factorization
    N = len(Yl)
    Ql = []
    zl = []
    for n in xrange(N):
        Qn, Rn = np.linalg.qr(Yl[n])
        Ql.append(Qn)
        zl.append(np.random.normal(size = [Qn.shape[1], 1]))
    Al = []
    k = 1
    inner_list = []
    fval_list = []
    for itnum in xrange(maxitnum):
        a = np.random.normal(size = [Ql[0].shape[0], 1])
        inner_fvl = []
        for innerItNum in xrange(maxInnerItNum):
            # a update
            anew = np.zeros([Ql[0].shape[0], 1])
            for n in xrange(N):
                anew += np.dot(Ql[n], zl[n])
            anew /= np.linalg.norm(anew)
            # z update
            for n in xrange(N):
                zl[n] = reshape(np.dot(Ql[n].T, anew), [-1,1])
            inner_fvl.append(np.linalg.norm(a - anew))
            a = anew.copy()
            if inner_fvl[-1] < gamma:
                break
        inner_list.append(inner_fvl)
        fval_list.append(fk(Ql, zl, a))
        Al.append(a.copy())
        k += 1
        for n in xrange(N):
            Pmat = np.eye(Ql[n].shape[1]) - np.dot(zl[n], zl[n].T)
            Ql[n] = np.dot(Ql[n], Pmat)
        if verbose:
            print "Itnum: %d/%d\t fval: %.5e\t inner_itnum: %d/%d\t inner_diff: %.5e" % (
                itnum+1, maxitnum, fval_list[-1], innerItNum+1, maxInnerItNum, inner_fvl[-1]
            )
        if fval_list[-1] >= eps:
            break
    support = {
        'fval': fval_list,
        'inner_it': inner_list
    }
    if inform:
        return np.hstack(Al), support
    return np.hstack(Al)

# COBEC
def fast_svd(A, eps=1e-8, dotaxis=1):
    if dotaxis == 1:
        ata = np.dot(A.T, A)
        v, s, _ = np.linalg.svd(ata)
    elif dotaxis == 0:
        aat =  np.dot(A, A.T)
        u, s, _ =  np.linalg.svd(aat)
    else:
        raise ValueError
    s = s**0.5
    cum = np.cumsum(s[::-1])
    I = (cum > eps).sum()
    if dotaxis == 1:
        u = np.dot(A, v[:, :I])
        u /= s[:I]
    else:
        v = np.dot(u[:, :I].T, A).T
        v /= s[:I]
    return u, s, v

def cobec(Yl, C, eps = 1e-8, maxitnum=30, verbose=False, inform=False, fast=False):
    # QR factorization
    N = len(Yl)
    Ql = []
    Zl = []
    fval_list = []
    for n in xrange(N):
        if not fast:
            Qn, Rn = np.linalg.qr(Yl[n])
        else:
            Qn, _, _ = fast_svd(Yl[n], eps=eps)
        Ql.append(Qn)
        Zl.append(np.random.normal(size = [Ql[n].shape[1], C]))
    A = np.random.uniform(-1, 1, [Ql[0].shape[0], C])
    for itnum in xrange(maxitnum):
        P = np.zeros([Ql[0].shape[0], Zl[0].shape[1]])
        for n in xrange(N):
            P += np.dot(Ql[n], Zl[n])
        if not fast:
            U, S, Vt = np.linalg.svd(P)
        else:
            U, S, Vt = fast_svd(P, eps=eps)
            Vt = Vt.T
        # truncated SVD
        Anew = np.dot(U[:, :C], Vt[:C, :])
        for n in xrange(N):
            Zl[n] = np.dot(Ql[n].T, Anew)
        fval_list.append(np.linalg.norm(A - Anew))
        A = Anew.copy()
        if verbose:
            print "Itnum: %d/%d\t fval: %.5e" % (itnum+1, maxitnum, fval_list[-1])
        if fval_list[-1] < eps:
            break
    if inform:
        support = {'fval': fval_list}
        return A, support
    return A

# SORTE
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.305.1290&rep=rep1&type=pdf
def sorte_fun(lamb, p):
    m = lamb.size
    assert (p >= 0) and ( p < m-2)
    dlamb = -np.diff(lamb)
    var2 = np.var(dlamb[p:])
    if var2 == 0:
        return np.inf
    return np.var(dlamb[p+1:]) / var2

def sorte(lamb):
    m = lamb.size
    p = brute(my_func, range(m-2), full_output = False)
    return p

if __name__=='__main__':
    pass
