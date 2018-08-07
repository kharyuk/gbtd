from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, FastICA
import numpy as np
import time
import copy
from .. import cca
from .. import gica

from ..computational_utilities import reshape

class GICAClassifier(BaseEstimator):
    '''
    Group independent component analysis (GICA) based classifier.
    
    Attributes:
        shapeClass
        individualRankPCA
        groupRankPCA
        commonRank
        sources
        maxitnum
        epsilon
        classes
        
    Methods:
        fit()
        transformSources()
        predict()
        saveParameters()
        loadParameters()
        
        
    '''
        
    def __init__(self, individualRankPCA=1, groupRankPCA=1, commonRank=1, shapeObject=None,
                 maxitnum=100, epsilon=1e-5):
        super(GICAClassifier, self).__init__()
        self.individualRankPCA = individualRankPCA
        self.groupRankPCA = groupRankPCA
        self.commonRank = commonRank
        self.timesEstimate = None
        self.shapeObject = list(shapeObject) # (Nobjects, Nsamples, Nfeatures)
        self.sources = None
        self.sourcesType = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.classes = None
        
    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        return reshape(X, shape)
        
    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            classes=self.classes, timesEstimate=self.timesEstimate
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType
        self.sources = df['sources']
        self.sourcesType = df['sourcesType']
        self.classes = df['classes']
        self.timesEstimate = df['timesEstimate']
        
    def fit(self, X, y_true):
        self.classes = np.unique(y_true)
        self.sources = []
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
            result = gica.gica(
                clX, self.individualRankPCA, self.groupRankPCA, self.commonRank,
                feature_extractor='fast_ica', maxitnum=self.maxitnum,
                random_state=None
            )
            toc = time.clock()
            S = result['ica']['sources']
            self.sources.append(S)
            self.timesEstimate.append(toc-tic)
        return

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
