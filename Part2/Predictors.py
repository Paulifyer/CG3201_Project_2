import numpy as np
import numpy.linalg as npl
import numpy.typing as npt


class ScratchLinReg:
    def __init__(self):
        self.theta = np.array([])

    def fit(self, Xtrain : npt.NDArray, ytrain : npt.NDArray):
        XtX = Xtrain.T @ Xtrain
        XtXInv = npl.inv(XtX)
        XtY = Xtrain.T @ ytrain
        self.theta = XtXInv @ XtY
        

    def predict(self, Xtest):
        return Xtest @ self.theta
    
class LinearRegressionMAP(ScratchLinReg):
    def __init__(self):
        super().__init__()
        self.l = (0 + 2 + 7 + 3 + 1 + 2 + 5) / 100.0 # Lambda Calculation

    def fit(self, Xtrain : npt.NDArray, ytrain : npt.NDArray):
        XtX = Xtrain.T @ Xtrain
        XtXInv = npl.inv(XtX + self.l * np.identity(XtX.shape[0]))
        XtY = Xtrain.T @ ytrain
        self.theta = XtXInv @ XtY

