from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, FastICA

import numpy as np
import time
from .. import cca
from .. import cobe

from ..computational_utilities import reshape


class COBEClassifier(BaseEstimator):
    '''
    Common orthogonal basis extraction (COBE) based classifier.
    
    Attributes:
        shapeObject
        commonRank
        sources
        sourcesType
        maxitnum
        epsilon
        classes
        #mixing
        
    Methods:
        fit()
        transformSources()
        predict()
        saveParameters()
        loadParameters()
        
        
    '''
        
    def __init__(self, commonRank=1, shapeObject=None, maxitnum=100, epsilon=1e-5):
        super(COBEClassifier, self).__init__()
        self.shapeObject = list(shapeObject) # (Nsamples, Nfeatures)
        self.commonRank = commonRank
        self.sources = None
        #self.mixing = None
        self.sourcesType = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.classes = None
        
    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        return reshape(X, shape)
        
    def fit(self, X, y_true):
        self.classes = np.unique(y_true)
        self.sources = []
        self.mixing = []
        self.timesEstimate = []
        for i in xrange(len(self.classes)):
            current_class = self.classes[i]
            ind = np.where(np.array(y_true) == current_class)[0]
            clX = X[ind].copy()
            if clX.ndim == 2:
                clX = self._reshape(clX)
            elif clX.ndim != 3:
                raise ValueError
            # fast: use AtA to estimate left orthogonal matrix
            # it is assumed that commonRank is small
            tic = time.clock()
            S = cobe.cobec(
                clX, self.commonRank, eps=self.epsilon, maxitnum=self.maxitnum,
                verbose=0, inform=0, fast=1
            )
            toc = time.clock()
            #B = []
            #for j in xrange(clX.shape[0]):
            #    B.append( np.dot(S.T, clX[j]) )
            self.timesEstimate.append(toc-tic)
            self.sources.append(S.copy())
            #self.mixing.append(B)
        self.sourcesType = 'computed'
        return
        
    def transformSources(self, constraint, random_state=None, maxitnum=500):
        constraint_local = constraint.lower()
        if constraint_local == 'ica':
            ica = FastICA(
                n_components=self.commonRank, algorithm='parallel', whiten=1,
                fun='logcosh', fun_args=None, max_iter=maxitnum, tol=1e-5,
                w_init=None, random_state=random_state
            )
            sources = []
            for i in xrange(len(self.sources)):
                tic = time.clock()
                S = ica.fit_transform(self.sources[i])
                toc = time.clock()
                self.timesEstimate[i] += toc-tic
                sources.append(S.copy())
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
            sources = []
            for i in xrange(len(self.sources)):
                tic = time.clock()
                S = nmf.fit_transform(np.maximum(self.sources[i], 0.))
                #S, _ = cnfe(self.sources[i], self.mixing[i], self.commonRank)
                toc = time.clock()
                self.timesEstimate[i] += toc-tic
                sources.append(S.copy())
            del self.sources
            self.sources = sources
            self.sourcesType = 'nmf'
        else:
            raise NotImplementedError

    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            classes=self.classes, timesEstimate=self.timesEstimate
            #mixing=self.mixing
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType#, self.mixing
        self.sources = df['sources']
        self.sourcesType = df['sourcesType']
        self.classes = df['classes']
        self.timesEstimate = df['timesEstimate']
        #self.mixing = df['mixing']

    def predict(self, X, metric='principal_angle', save_filename=None):
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
        Y = X.copy()
        if Y.ndim == 2:
            Y = self._reshape(Y)
        elif Y.ndim != 3:
            raise ValueError
        shapeY = Y.shape
        answer = []
        if save_filename is not None:
            results = np.zeros([Nobj, Nclasses])
        for k in xrange(Nobj):
            #Tens = Y[k] #.T # Npix x Nframes
            measured = []
            for i in xrange(len(self.classes)):
                S = self.sources[i]
                distance = selected_metric(Y[k], S)
                measured.append(distance)
            measured = np.array(measured)
            answer.append(self.classes[measured.argmax()])
            if save_filename is not None:
                results[k, :] = measured
                np.savez_compressed(
                    save_filename, results=results, k=k, classes=self.classes
                )
        return answer
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    pass
