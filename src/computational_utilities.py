import numpy as np

import collections

def reshape(a, shape):
    return np.reshape(a, shape, order='F')

def vec(a, which=None):
    if which is None:
        return a.flatten(order='F')
    if which == 'column':
        return reshape(a, [-1, 1])
    if which == 'row':
        return reshape(a, [1, -1])
    raise NotImplementedError
    
def prodTenMat(T, M, mode_t, mode_m=1):
    assert M.ndim == 2, "Second operand must be a matrix"
    #result = _np.swapaxes(T, 0, mode_t)
    #result = _np.tensordot(M, result, axes=[(mode_m), (0)])
    #result = _np.swapaxes(result, 0, mode_t)
    subT = range(T.ndim)
    subR = range(T.ndim)
    subR[mode_t] = T.ndim
    subM = [T.ndim, T.ndim]
    subM[mode_m] = subT[mode_t]
    result = np.einsum(T, subT, M, subM, subR)
    return result
    
def sub2ind(indices, shape):
    index = np.ravel_multi_index(indices, shape, order='F')
    return index
    
def ind2sub(index, shape):
    indices = np.unravel_index(index, shape, order='F')
    return indices

#https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
def unsortedSet(s):
    tmp = collections.OrderedDict.fromkeys(s)
    return tmp.keys()

if __name__ == '__main__':
    pass
