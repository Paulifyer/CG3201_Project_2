import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 

from Predictors import ScratchLinReg, LinearRegressionMAP

def buildData() -> tuple[npt.ArrayLike, npt.ArrayLike]:
    filePath = os.path.join('computer+hardware', 'machine.data')
    df = pd.read_csv(filePath, header=None, names=[
        'vendor', 'model', 'MYCT', 'MMIN', 'MMAX',
        'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
    ])

    print("First few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())

    X = df[['MYCT', 'MMIN', 'MMAX','CACH', 'CHMIN', 'CHMAX', 'PRP']].values
    Y = np.array(df['ERP'].values)
    Y = Y.reshape(-1, 1)

    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    return X, Y


def plotActualVsPredicted(Xtrain: npt.NDArray, ytrain: npt.NDArray, Xtest: npt.NDArray, ytest: npt.NDArray):
    models = {
        "MLE" : ScratchLinReg(),
        "MAP" : LinearRegressionMAP(),
        "Sklearn LinearRegression": LinearRegression(),
        "Ridge Regression": Ridge()
    }

    fig, axes = plt.subplots(1, len(models), figsize=(18, 5))

    for i, (key, model) in enumerate(models.items()):
        model.fit(Xtrain, ytrain)
        yPred = model.predict(Xtest)
        mse = mean_squared_error(ytest, yPred)
        print(f"MSE of {key}: {mse}")
        axes[i].scatter(ytest, yPred, color='green', label="Actual Data")
        axes[i].set_title(f"Actual vs Predicted ({key})")
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red', linestyle='--')
    plt.show()

def plotWeights(Xtrain: npt.NDArray, ytrain: npt.NDArray, Xtest: npt.NDArray, ytest: npt.NDArray):
    models = {
        "MLE" : ScratchLinReg(),
        "MAP" : LinearRegressionMAP(),
    }

    features = ['MYCT', 'MMIN', 'MMAX','CACH', 'CHMIN', 'CHMAX', 'PRP']

    x = np.arange(len(features))
    width = 0.35
    models['MLE'].fit(Xtrain, ytrain)
    models['MAP'].fit(Xtrain, ytrain)
    plt.bar(x - width/2, models['MLE'].theta.ravel(), width=width, label=f"MLE")
    plt.bar(x + width/2, models['MAP'].theta.ravel(), width=width, label=f"MAP")
    print(models['MLE'].theta.ravel())
    print(models['MAP'].theta.ravel())
    plt.xticks(x, features)
    plt.legend()
    plt.title("Feature Weights by Model")
    plt.xlabel('Features')
    plt.ylabel('Value')        
    plt.show()


def __main__():
    X, Y = buildData()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    plotWeights(Xtrain, ytrain, Xtest, ytest)

    
    


__main__()
