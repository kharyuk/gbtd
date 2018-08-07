from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, FastICA
import time
import copy
import numpy as np

from .. import gtcd
from .. import group
from .. import cca

from ..computational_utilities import reshape


# global variables
_maxInnerIt = 15
_tolRes = 1e-8
_tolGrad = 1e-8
_tolSwamp = 1e-8


class GLROClassifier(BaseEstimator):
    '''
    Group (Lr, 1) (GLRO) based classifier.
    
    Attributes:
        shapeObject
        commonRank
        sources
        sourcesType
        sourceModes
        maxitnum
        epsilon
        classes
        nShortModes [except group mode]
        
    Methods:
        fit()
        transformSources()
        predict()
        saveParameters()
        loadParameters()
        
        
    '''
        
    def __init__(self, individualRank=1, commonRank=1, shapeObject=None, sourceModes=None,
                 method='als', nShortModes=None, constraintMethod='projected',
                 fullModesConstraint=None, maxitnum=100, epsilon=1e-5):
        super(GLROClassifier, self).__init__()
        self.shapeObject = list(shapeObject)
        self.individualRank = individualRank
        self.commonRank = commonRank
        self.sources = None
        self.sourcesType = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.classes = None
        self.sourceModes = list(sourceModes)
        if len(self.sourceModes) > 1:
            raise NotImplementedError('Multi source modes are not supported at the moment')
        self.fullModesConstraint = fullModesConstraint
        self.nShortModes = 1
        if nShortModes is not None:
            assert isinstance(nShortModes, int) and (0 < nShortModes < len(self.shapeObject))-1
            self.nShortModes += nShortModes
        # method
        current_method = method.lower()
        if current_method == 'als':
            self.method = alsM
        elif current_method == 'gd':
            self.method = gdM
        elif current_method == 'cg-fr':
            self.method = cgfrM
        elif current_method == 'cg-pr':
            self.method = cgprM
        elif current_method == 'cg-hs':
            self.method = cghsM
        elif current_method == 'cg-dy':
            self.method = cgdyM
        elif current_method == 'gn':
            self.method = gnM
        elif current_method == 'lm-n':
            self.method = lmnM
        elif current_method == 'lm-q':
            self.method = lmqM
        elif current_method == 'dogleg':
            self.method = doglegM
        elif current_method == 'scg-qn':
            self.method = scg_qnM
        elif current_method == 'scg-fn':
            self.method = scg_fnM
        else:
            raise NotImplementedError
            
        # constraints
        current_constraintMethod = constraintMethod.lower()
        assert current_constraintMethod in ('projected', 'lm')
        self.groupConstraintMethod = current_constraintMethod

    def transformSources(self, constraint, random_state=None, maxitnum=500):
        constraint_local = constraint.lower()
        sources = {i: [] for i in xrange(len(self.shapeObject))}
        if constraint_local == 'ica':
            ica = FastICA(
                n_components=self.commonRank, algorithm='parallel', whiten=1,
                fun='logcosh', fun_args=None, max_iter=maxitnum, tol=1e-5,
                w_init=None, random_state=random_state
            )
            for k in xrange(len(self.shapeObject)):
                for i in xrange(len(self.sources[k])):
                    tic = time.clock()
                    S = ica.fit_transform(self.sources[k][i])
                    toc = time.clock()
                    self.timesEstimate[i] += toc-tic ######
                    sources[k].append(S.copy())
            del self.sources
            self.sources = sources
            self.sourcesType = 'ica'
        elif constraint_local == 'nmf':
            nmf = NMF(
                n_components=self.commonRank, init=None, solver='mu',
                beta_loss='frobenius', tol=1e-5, max_iter=maxitnum,
                random_state=random_state, alpha=0., l1_ratio=0., verbose=0,
                shuffle=0
            )
            for k in xrange(len(self.sourceModes)):
                sourceMode = self.sourceModes[k]
                for i in xrange(len(self.sources[sourceMode])):
                    tic = time.clock()
                    S = nmf.fit_transform(np.maximum(self.sources[sourceMode][i], 0.))
                    #S, _ = cnfe(self.sources[i], self.mixing[i], self.commonRank)
                    toc = time.clock()
                    self.timesEstimate[i] += toc-tic #####
                    sourcess[sourceMode].append(S.copy())
            del self.sources
            self.sources = sources
            self.sourcesType = 'nmf'
        else:
            raise NotImplementedError

    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        Y = reshape(X, shape)
        transposition = range(1, len(self.shapeObject)+1) + [0]
        Y = np.transpose(Y, transposition)
        return Y
        
    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            classes=self.classes, timesEstimate=self.timesEstimate
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType
        self.sources = df['sources'].item()
        self.sourcesType = df['sourcesType']
        self.classes = df['classes']
        self.timesEstimate = df['timesEstimate']
               
    def fit(self, X, y_true, verbose=True, maxitnum=None):
        if maxitnum is None:
            maxitnum = self.maxitnum
        self.classes = list(set(y_true))
        #self.sources = {self.sourceModes[i]: [] for i in xrange(len(self.sourceModes))}
        self.sources = {i: [] for i in xrange(len(self.shapeObject))}
        self.timesEstimate = []
        for i in xrange(len(self.classes)):
            current_class = self.classes[i]
            ind = np.where(np.array(y_true) == current_class)[0]
            clX = X[ind].copy()
            clX = self._reshape(clX)
            clX = clX.astype('d')
            n = clX.shape
            groupConstraint = group.group_constraint(
                n, self.sourceModes, self.groupConstraintMethod
            )
            znorm= np.linalg.norm(clX)
            clX /= znorm
            cdN = None
            P = len(clX.shape)-self.nShortModes
            ldN = {
                'L': [self.individualRank]*n[-1] + [self.commonRank],
                'P': P,
                'fullModesConstraint': self.fullModesConstraint
            }
            tdN = None
            x0 = None
            tic = time.clock()
            cdN, ldN, tdN, info = self.method(
                clX, x0, cdN, ldN, tdN, maxitnum, groupConstraint, verbose
            )
            toc = time.clock()
            self.timesEstimate.append(toc-tic)
            indCommon = self.commonRank
            fmc = self.fullModesConstraint
            for k in xrange(len(self.shapeObject)):
                #modeNumber = self.sourceModes[k]
                if k < P:
                    if (fmc is not None) and (fmc[k] is not None):
                        self.sources[k].append(
                            ldN['B'][k][2][:, -indCommon:]
                        )
                    else:
                        self.sources[k].append(
                            ldN['B'][k][:, -indCommon:]
                        )
                else:
                    self.sources[k].append(
                            ldN['B'][k][:, -1:]
                        )
            #if len(self.sourceModes) > 0:
                
        return
 

    def predict(self, X, metric='principal_angle', save_filename=None, reproject=False):
        Nobj = X.shape[0]
        Nclasses = len(self.classes)
        if callable(metric):
            selected_metric = metric
        else:
            selected_metric_name = metric.lower()
            if selected_metric_name == 'principal_angle':
                selected_metric = cca.principal_angle
            else:
                raise NotImplementedError
        Y = self._reshape(X)
        shapeY = Y.shape
        answer = []
        if save_filename is not None:
            results = np.zeros([Nobj, Nclasses])
        # TODO: multimodal sources
        sourceMode = self.sourceModes[0]
        
        for k in xrange(Nobj):
            #Tens = Y[k] #.T # Npix x Nframes
            measured = []
            for i in xrange(len(self.classes)):
                S = self.sources[sourceMode][i]
                if reproject:
                    Z = Y.copy()
                    for k in xrange(len(self.shapeObject)):
                        if k in self.sourceModes:
                            continue
                        Z = gtcd.prodTenMat(Z, np.linalg.pinv(self.sources[k][i]), k)
                    '''
                    I = np.zeros([self.commonRank]*len(self.shapeObject))
                    np.fill_diagonal(I, 1.)
                    axes = set(range(len(self.shapeObject))).difference(set(self.sourceModes))
                    axes = np.array(list(axes))
                    axes0 = np.arange(len(self.shapeObject)+1)
                    axes1 = np.arange(len(self.shapeObject))+len(self.shapeObject)+1
                    axes1[axes] = axes
                    axes0[-1] = 2*(len(self.shapeObject)+1)
                    Z = np.einsum(Z, axes0, I, axes1)
                    '''
                else:
                    Z = Y.copy()
                Z = reshape((Z.T[k]).T, [Z.shape[0], -1])
                distance = selected_metric(Z, S)
                measured.append(distance)
            measured = np.array(measured)
            answer.append(self.classes[measured.argmax()])
            if save_filename is not None:
                results[k, :] = measured
                np.savez_compressed(
                    save_filename, results=results, k=k, classes=self.classes
                )
        return answer
    
    

class GTLDClassifier(BaseEstimator):
    '''
    Group Tucker-(Lr, 1) decomposition (GTLD) based classifier.
    
    
    Attributes:
        shapeObject
        commonRank
        sources
        sourcesType
        sourceModes
        maxitnum
        epsilon
        classes
        nShortModes [except group mode]
        
    Methods:
        fit()
        transformSources()
        predict()
        saveParameters()
        loadParameters()
        
        
    '''
        
    def __init__(self, individualRank=1, commonRank=1, shapeObject=None, sourceModes=None,
                 method='als', nShortModes=None, constraintMethod='projected',
                 fullModesConstraint=None, maxitnum=100, epsilon=1e-5):
        super(GTLDClassifier, self).__init__()
        self.shapeObject = list(shapeObject)
        self.individualRank = individualRank
        self.commonRank = commonRank
        self.sources = None
        self.sourcesType = None
        self.cores = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.classes = None
        self.sourceModes = list(sourceModes)
        if len(self.sourceModes) > 1:
            raise NotImplementedError('Multi source modes are not supported at the moment')
        self.fullModesConstraint = fullModesConstraint
        self.nShortModes = 1
        if nShortModes is not None:
            assert isinstance(nShortModes, int) and (0 < nShortModes < len(self.shapeObject))-1
            self.nShortModes += nShortModes
        # method
        current_method = method.lower()
        if current_method == 'als':
            self.method = alsM
        elif current_method == 'gd':
            self.method = gdM
        elif current_method == 'cg-fr':
            self.method = cgfrM
        elif current_method == 'cg-pr':
            self.method = cgprM
        elif current_method == 'cg-hs':
            self.method = cghsM
        elif current_method == 'cg-dy':
            self.method = cgdyM
        elif current_method == 'gn':
            self.method = gnM
        elif current_method == 'lm-n':
            self.method = lmnM
        elif current_method == 'lm-q':
            self.method = lmqM
        elif current_method == 'dogleg':
            self.method = doglegM
        elif current_method == 'scg-qn':
            self.method = scg_qnM
        elif current_method == 'scg-fn':
            self.method = scg_fnM
        else:
            raise NotImplementedError
            
        # constraints
        current_constraintMethod = constraintMethod.lower()
        assert current_constraintMethod in ('projected', 'lm')
        self.groupConstraintMethod = current_constraintMethod

    def transformSources(self, constraint, random_state=None, maxitnum=500):
        constraint_local = constraint.lower()
        sources = {i: [] for i in xrange(len(self.shapeObject))}
        if constraint_local == 'ica':
            ica = FastICA(
                n_components=self.commonRank, algorithm='parallel', whiten=1,
                fun='logcosh', fun_args=None, max_iter=maxitnum, tol=1e-5,
                w_init=None, random_state=random_state
            )
            for k in xrange(len(self.shapeObject)):
                for i in xrange(len(self.sources[k])):
                    tic = time.clock()
                    S = ica.fit_transform(self.sources[k][i])
                    toc = time.clock()
                    self.timesEstimate[i] += toc-tic ######
                    sources[k].append(S.copy())
            del self.sources
            self.sources = sources
            self.sourcesType = 'ica'
        elif constraint_local == 'nmf':
            nmf = NMF(
                n_components=self.commonRank, init=None, solver='mu',
                beta_loss='frobenius', tol=1e-5, max_iter=maxitnum,
                random_state=random_state, alpha=0., l1_ratio=0., verbose=0,
                shuffle=0
            )
            for k in xrange(len(self.sourceModes)):
                sourceMode = self.sourceModes[k]
                for i in xrange(len(self.sources[sourceMode])):
                    tic = time.clock()
                    S = nmf.fit_transform(np.maximum(self.sources[sourceMode][i], 0.))
                    #S, _ = cnfe(self.sources[i], self.mixing[i], self.commonRank)
                    toc = time.clock()
                    self.timesEstimate[i] += toc-tic #####
                    sourcess[sourceMode].append(S.copy())
            del self.sources
            self.sources = sources
            self.sourcesType = 'nmf'
        else:
            raise NotImplementedError

    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        Y = reshape(X, shape)
        transposition = range(1, len(self.shapeObject)+1) + [0]
        Y = np.transpose(Y, transposition)
        return Y
        
    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            classes=self.classes, timesEstimate=self.timesEstimate,
            cores=self.cores
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType
        self.sources = df['sources'].item()
        self.sourcesType = df['sourcesType']
        self.classes = df['classes']
        self.timesEstimate = df['timesEstimate']
        self.cores = df['cores']
               
    def fit(self, X, y_true, verbose=True, maxitnum=None, modeSizeFirstPriority=True):
        if maxitnum is None:
            maxitnum = self.maxitnum
        self.classes = list(set(y_true))
        #self.sources = {self.sourceModes[i]: [] for i in xrange(len(self.sourceModes))}
        self.sources = {i: [] for i in xrange(len(self.shapeObject))}
        self.cores = []
        self.timesEstimate = []
        for i in xrange(len(self.classes)):
            current_class = self.classes[i]
            ind = np.where(np.array(y_true) == current_class)[0]
            clX = X[ind].copy()
            clX = self._reshape(clX)
            clX = clX.astype('d')
            n = clX.shape
            groupConstraint = group.group_constraint(
                n, self.sourceModes, self.groupConstraintMethod
            )
            znorm= np.linalg.norm(clX)
            clX /= znorm
            cdN = None
            P = len(clX.shape)-self.nShortModes
            ldN = {
                'L': [self.individualRank]*n[-1],
                'P': P,
                'fullModesConstraint': self.fullModesConstraint
            }
            r = np.zeros([1, len(clX.shape)])
            r[0, :-1] = self.commonRank
            if modeSizeFirstPriority:
                r[0, :-1] = np.minimum(r[0, :-1], n[:-1]) 
            # last mode - group axis
            r[:, -1] = n[-1]
            tdN = {
                'r': r.astype('i')
            }
            x0 = None
            tic = time.clock()
            cdN, ldN, tdN, info = self.method(
                clX, x0, cdN, ldN, tdN, maxitnum, groupConstraint, verbose
            )
            toc = time.clock()
            self.timesEstimate.append(toc-tic)
            indCommon = self.commonRank
            fmc = self.fullModesConstraint
            for k in xrange(len(self.shapeObject)):
                self.sources[k].append(
                    tdN['A'][k]
                )
            self.cores.append(
                tdN['G']
            )
            #if len(self.sourceModes) > 0:
                
        return
 

    def predict(self, X, metric='principal_angle', save_filename=None, reproject=False,
    use_core=False):
        Nobj = X.shape[0]
        Nclasses = len(self.classes)
        if callable(metric):
            selected_metric = metric
        else:
            selected_metric_name = metric.lower()
            if selected_metric_name == 'principal_angle':
                selected_metric = cca.principal_angle
            else:
                raise NotImplementedError
        Y = self._reshape(X)
        shapeY = Y.shape
        answer = []
        if save_filename is not None:
            results = np.zeros([Nobj, Nclasses])
        # TODO: multimodal sources
        sourceMode = self.sourceModes[0]
        
        for k in xrange(Nobj):
            #Tens = Y[k] #.T # Npix x Nframes
            measured = []
            for i in xrange(len(self.classes)):
                S = self.sources[sourceMode][i]
                if reproject:
                    Z = Y.copy()
                    for k in xrange(len(self.shapeObject)):
                        if k in self.sourceModes:
                            continue
                        Z = gtcd.prodTenMat(Z, np.linalg.pinv(self.sources[k][i]), k)
                    if use_core:
                        axes = set(range(len(self.shapeObject))).difference(set(self.sourceModes))
                        axes = list(axes)
                        transposition = self.sourceModes + axes
                        G = np.transpose(self.cores[i], transposition)
                        shapeG = np.array(G.shape)
                        G = reshape(G, [int(np.prod(shapeG[np.array(self.sourceModes)])), -1])
                        G = np.linalg.pinv(G)
                        G = reshape(G, shapeG[::-1])
                        G = np.transpose(G, np.argsort(transposition))
                        axes0 = np.arange(len(self.shapeObject)+1)
                        axes1 = np.arange(len(self.shapeObject))+len(self.shapeObject)+1
                        axes1[axes] = axes
                        axes0[-1] = 2*(len(self.shapeObject)+1)
                        Z = np.einsum(Z, axes0, G, axes1)
                    '''
                    I = np.zeros([self.commonRank]*len(self.shapeObject))
                    np.fill_diagonal(I, 1.)
                    axes = set(range(len(self.shapeObject))).difference(set(self.sourceModes))
                    axes = np.array(list(axes))
                    
                    '''
                else:
                    Z = Y.copy()
                Z = reshape((Z.T[k]).T, [Z.shape[0], -1])
                distance = selected_metric(Z, S)
                measured.append(distance)
            measured = np.array(measured)
            answer.append(self.classes[measured.argmax()])
            if save_filename is not None:
                results[k, :] = measured
                np.savez_compressed(
                    save_filename, results=results, k=k, classes=self.classes
                )
        return answer





# set up different algorithms
def alsM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='als', verbose=verbose, 
        regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gdM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gdrtM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True,
        verbose=verbose, regTGD=1e-3, regPGD=None, doSA=0, constraints=constraints
    )
def gdrpM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=1e-3, doSA=0, constraints=constraints
    )
def cgfrM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='fr', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def cgprM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='pr', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def cghsM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='hs',
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )

def cgdyM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='dy', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes,tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gn', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def lmqM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='lm', epsilonLM=1e-8,
        lmSetup='Quadratic', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
        doSA=0, constraints=constraints
    )
def lmnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='lm', epsilonLM=1e-8,
        lmSetup='Nielsen', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
        doSA=0, constraints=constraints
    )
def doglegM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, trStep='dogleg',
        trDelta0=1.2,trEta=0.23
    )
def scg_qnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, curvature=0, trStep='scg',
        trDelta0=1.2, trEta=0.23
    )
def scg_fnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, curvature=1, trStep='scg',
        trDelta0=1.2, trEta=0.23
    )
        
        
if __name__ == '__main__':
    pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
