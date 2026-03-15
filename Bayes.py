import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

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
    X, y = build_data("caltech_cougar")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.75, shuffle=True)
    print(ytrain.shape)
    print(Xtrain.shape)

__main__()

