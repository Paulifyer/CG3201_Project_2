import Bayes
import numpy as np
import numpy.typing as npt

class GaussSimple:
    def __init__(self):
        self.priorSpam = None
        self.logProbPixFace = None
        self.logProbPixBody = None
    
    def fit(self, Xtrain, ytrain):
        faceMask = (ytrain == 1)
        bodyMask = ~faceMask
        
        self.priorSpam = np.mean(ytrain)

        facePixCount = Xtrain[faceMask].sum(axis=0)
        bodyPixCount = Xtrain[bodyMask].sum(axis=0)
        faceCount = faceMask.sum()
        bodyCount = bodyMask.sum()

        self.logProbPixFace = np.log2((facePixCount + 1) / (faceCount + Xtrain.shape[1]))
        self.logProbPixBody = np.log2((bodyPixCount + 1) / (bodyCount + Xtrain.shape[1]))

    def computeScores(self, Xtest):
        faceScores = Xtest @ self.logProbPixFace
        bodyScores = Xtest @ self.logProbPixBody

        return faceScores - bodyScores
    
    def predict(self, Xtest):
        scores = self.computeScores(Xtest)
        return (scores > 0).astype(int)
    


