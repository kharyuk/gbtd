import numpy as np

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

def cca(X, Y, N=1, fast=True, mode='uv'):
    shapeX = X.shape
    shapeY = Y.shape
    
    assert shapeX[0] == shapeY[0]
    
    p = min(shapeX[1], shapeY[1])
    assert (N > 0) and (N <= p)
    
    if not fast:
        u1, s1, v1 = np.linalg.svd(X)
        u2, s2, v2 = np.linalg.svd(Y)
        v1 = v1[:s1.size].T
        v2 = v2[:s2.size].T
        u1 = u1[:, :s1.size]
        u2 = u2[:, :s2.size]
    else:
        u1, s1, v1 = fast_svd(X)
        u2, s2, v2 = fast_svd(Y) 
        
    U, S, Vt = np.linalg.svd(np.dot(u1.T, u2))
    if mode == 'ab':
        a = np.dot(v1 / s1, U[:, :N])
        b = np.dot(v2 / s2, Vt[:N, :].T)
        return a, b
    elif mode == 'uv':
        uloc = np.dot(u1, U[:, :N])
        vloc = np.dot(u2, Vt[:N, :].T)
        return uloc, vloc
    else:
        raise NotImplementedError
    return

def corr(x, y):
    return np.corrcoef(x, y, rowvar=0)

def principal_angle(X, Y):
    u, v = cca(X, Y)
    C = corr(u, v)
    return C[0, 1]

if __name__ == '__main__':
    u = np.random.rand(8, 4)
    v = np.random.rand(8, 3)
    a, b = cca(u, v, N=1)
    print corr(a, b)
