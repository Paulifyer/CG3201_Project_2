import numpy as np
import numpy.typing as npt

class GaussSimple:
    def __init__(self):
        self.priors = np.array([])
        self.classes = np.array([])
        self.mu = np.array([])
        self.sigma2 = np.array([])
    
    def fit(self, Xtrain, ytrain):
        self.classes = np.unique(ytrain)
        nClasses = len(self.classes)
        nFeatures = Xtrain.shape[1]

        self.mu = np.zeros((nClasses, nFeatures))
        self.sigma2 = np.zeros((nClasses, nFeatures))
        self.priors = np.zeros(nClasses)

        for i, c in enumerate(self.classes):
            X_c = Xtrain[ytrain == c]
            self.mu[i] = np.mean(X_c, axis=0)
            self.sigma2[i] = np.var(X_c, axis=0)
            self.priors[i] = len(X_c) / len(ytrain)       
    
    def predict(self, Xtest):
        nSamples = Xtest.shape[0]
        nClasses = len(self.classes)

        logPosteriors = np.zeros((nSamples, nClasses))

        for i in range(nClasses):
            logPriors = np.log(self.priors[i])
            logProbsP1 = -0.5 * np.log(2 * np.pi * self.sigma2[i])
            logProbsP2 = -0.5 * ((Xtest - self.mu[i]) ** 2) / self.sigma2[i]
            logProbs = logProbsP1 + logProbsP2
            logLikelihood = np.sum(logProbs, axis=1)
            logPosteriors[:, i] = logPriors + logLikelihood
        
        predictions = self.classes[np.argmax(logPosteriors, axis=1)]

        return predictions
    
class LaplacePredictor:
    def __init__(self):
        self.priors = np.array([])
        self.classes = np.array([])
        self.mu = np.array([])
        self.b = np.array([])

    def fit(self, Xtrain, ytrain):
        self.classes = np.unique(ytrain)
        nClasses = len(self.classes)
        nFeatures = Xtrain.shape[1]

        self.mu = np.zeros((nClasses, nFeatures))
        self.b = np.zeros((nClasses, nFeatures))
        self.priors = np.zeros(nClasses)

        for i, c in enumerate(self.classes):
            X_c = Xtrain[ytrain == c]
            self.mu[i] = np.median(X_c, axis=0)
            self.b[i] = np.mean(np.abs(X_c - self.mu[i]), axis=0)
            self.priors[i] = len(X_c) / len(ytrain)    

    def predict(self, Xtest):
        nSamples = Xtest.shape[0]
        nClasses = len(self.classes)

        logPosteriors = np.zeros((nSamples, nClasses))

        for i in range(nClasses):
            logPriors = np.log(self.priors[i])
            logProbsP1 = -np.log(2 * self.b[i])
            logProbsP2 = -np.abs(Xtest - self.mu[i]) / self.b[i]
            logProbs = logProbsP1 + logProbsP2
            logLikelihood = np.sum(logProbs, axis=1)
            logPosteriors[:, i] = logPriors + logLikelihood
        
        predictions = self.classes[np.argmax(logPosteriors, axis=1)]

        return predictions

    


