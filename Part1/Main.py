import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from Predictors import GaussSimple, LaplacePredictor, GaussWithMap

np.random.seed(0)



def preprocess_image(path, size=(64, 64)):
    img = cv2.imread(path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img.flatten()

def build_data(dataset_dir):
    labels = []
    rows = []

    for label, folder in [(1, "cougar_face"), (-1, "cougar_body")]:
        folder_path = os.path.join(dataset_dir, folder)
        
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".jpg"):
                continue
            path = os.path.join(folder_path, fname)
            row = np.hstack([
                label
            ])
            row2 = preprocess_image(path)
            labels.append(row)
            rows.append(row2)
    labels = np.array(labels)
    rows = np.array(rows)
    return rows, labels

def calcMuAndSigma(S: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    mean = S.mean() / 7
    mu0 = np.full(4096, mean * 255)
    sigma0 = np.arange(4096)
    sigma0 = (S.var() * (1 + (sigma0 % S.max()) / 10))
    # print(f"mu0: {mu0}")
    # print(f"sigma0: {sigma0}")
    return sigma0, mu0


def PlotConfusionMatrix(
        Xtrain : npt.ArrayLike, ytrain : npt.ArrayLike,
        Xtest : npt.ArrayLike, ytest : npt.ArrayLike,
        mu0: npt.ArrayLike, sigma0: npt.ArrayLike) -> None:
    classifiers = {
        "Gaussian": GaussSimple(),
        "Gaussian w/ MAP": GaussWithMap(mu0, sigma0),
        "Laplace": LaplacePredictor()
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for i, (key, classifier) in enumerate(classifiers.items()):
        classifier.fit(Xtrain, ytrain)
        yPred = classifier.predict(Xtest)
        print("accuracy: ", np.mean(yPred==ytest))
        ConfusionMatrixDisplay.from_predictions(ytest, yPred, ax=axes[i], colorbar=False)
        axes[i].set_title(key)


    plt.tight_layout()
    plt.show()


def __main__():
    S = np.array([0, 2, 7, 3, 1, 2, 5])
    sigma0, mu0 = calcMuAndSigma(S)

    X, y = build_data("caltech_cougar")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, shuffle=True)
    print(Xtrain)
    ytrain = ytrain.ravel()

    PlotConfusionMatrix(Xtrain, ytrain, Xtest, ytest, mu0, sigma0)
    


__main__()

