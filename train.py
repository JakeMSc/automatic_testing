from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import os

X = np.genfromtxt("data/X_train.csv", delimiter=",")
y = np.genfromtxt("data/y_train.csv", delimiter=",")

# Train a model
clf = LogisticRegression().fit(X, y)

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.joblib'
joblib.dump(clf, open(filename, 'wb'))
