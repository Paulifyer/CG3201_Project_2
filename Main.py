import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from Predictors import GaussSimple, LaplacePredictor

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


def __main__():
    S = np.array([0, 2, 7, 3, 1, 2, 5])
    mean = S.mean()
    S = np.full(64*64, mean / 7 * 255)
    print(S)
    classifiers = {
        "Gaussian": GaussSimple(),
        "Laplace": LaplacePredictor()
    }
    X, y = build_data("caltech_cougar")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, shuffle=True)
    ytrain = ytrain.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (key, classifier) in enumerate(classifiers.items()):
        classifier.fit(Xtrain, ytrain)
        yPred = classifier.predict(Xtest)
        print("accuracy: ", np.mean(yPred==ytest))
        ConfusionMatrixDisplay.from_predictions(ytest, yPred, ax=axes[i], colorbar=False)
        axes[i].set_title(key)


    plt.tight_layout()
    plt.show()


__main__()

