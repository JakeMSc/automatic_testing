from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import os

X, y = make_classification(10000, n_features = 10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

if not os.path.isdir("data/"):
    os.mkdir("data")

np.savetxt("data/X_train.csv", X_train, delimiter=",")
np.savetxt("data/X_test.csv", X_test, delimiter=",")
np.savetxt("data/y_train.csv", y_train, delimiter=",")
np.savetxt("data/y_test.csv", y_test, delimiter=",")