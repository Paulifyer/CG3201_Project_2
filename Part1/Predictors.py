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


class GaussWithMap(GaussSimple):
    def __init__(self, mu0, sigma0):
        super().__init__()
        self.mu0 = mu0
        self.sigma0 = sigma0
    
    def fit(self, Xtrain, ytrain):
        self.classes = np.unique(ytrain)
        nClasses = len(self.classes)
        nFeatures = Xtrain.shape[1]

        self.mu = np.zeros((nClasses, nFeatures))
        self.sigma2 = np.zeros((nClasses, nFeatures))
        self.priors = np.zeros(nClasses)
        var = np.var(Xtrain, axis=0)
        for i, c in enumerate(self.classes):
            X_c = Xtrain[ytrain == c]
            mean = np.mean(X_c, axis=0)
            # var = np.var(X_c, axis=0)
            n = len(X_c)
            self.mu[i] = (var * self.mu0 + n * self.sigma0 * mean) / (var + n * self.sigma0)
            self.sigma2[i] = (var * self.sigma0) / (var + n * self.sigma0)
            self.priors[i] = n / len(ytrain)
        with open("fit.txt", "w") as text_file:
            text_file.write("\n=== MODEL PARAMETERS ===\n")
            for i, c in enumerate(self.classes):
                text_file.write(f"\nClass {c}:\n")
                text_file.write(f"  Prior: {self.priors[i]:.6f}\n")
                text_file.write(f"  log(prior): {np.log(self.priors[i]):.6f}\n")
                text_file.write(f"  mu mean: {self.mu[i].mean():.6f}\n")
                text_file.write(f"  mu std: {self.mu[i].std():.6f}\n")
                text_file.write(f"  sigma2 mean: {self.sigma2[i].mean():.10f}\n")
                text_file.write(f"  sigma2 min: {self.sigma2[i].min():.10f}\n")
                text_file.write(f"  sigma2 max: {self.sigma2[i].max():.10f}\n")
            
            text_file.write("\n=== PRIOR PARAMETERS ===\n")
            text_file.write(f"mu0 mean: {self.mu0.mean():.6f}\n")
            text_file.write(f"mu0 std: {self.mu0.std():.6f}\n")
            text_file.write(f"sigma0 mean: {self.sigma0.mean():.10f}\n")
            text_file.write(f"sigma0 min: {self.sigma0.min():.10f}\n")
            text_file.write(f"sigma0 max: {self.sigma0.max():.10f}\n")

    def predict(self, Xtest):
        nSamples = Xtest.shape[0]
        nClasses = len(self.classes)

        logPosteriors = np.zeros((nSamples, nClasses))

        eps = 1e-6 # eps is for stability to prevent any index of sigma2 from being 0

        with open("test.txt", "w") as text_file:
            text_file.write("\n=== FIRST TEST SAMPLE BREAKDOWN ===\n")
            for i in range(nClasses):
                logPriors = np.log(self.priors[i])

                sigma2_stable = self.sigma2[i] + eps

                logProbsP1 = -0.5 * np.log(2 * np.pi * sigma2_stable)
                logProbsP2 = -0.5 * ((Xtest - self.mu[i]) ** 2) / sigma2_stable
                logProbs = logProbsP1 + logProbsP2
                logLikelihood = np.sum(logProbs, axis=1)
                logPosteriors[:, i] = logPriors + logLikelihood
                text_file.write(f"\nClass {self.classes[i]}:\n")
                text_file.write(f"  log prior: {logPriors:.6f}\n")
                text_file.write(f"  sum(log normalization): {logProbsP1.sum():.6f}\n")
                text_file.write(f"  sum(log distance penalty): {logProbsP2.sum():.6f}\n")
                text_file.write(f"  total log likelihood: {(logProbsP1.sum() + logProbsP2.sum()):.6f}\n")
                text_file.write(f"  log posterior: {logPosteriors[0, i]:.6f}\n")
        
        predictions = self.classes[np.argmax(logPosteriors, axis=1)]

        return predictions


