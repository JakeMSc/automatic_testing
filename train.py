from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

X = np.genfromtxt("data/X_train.csv", delimiter=",")
y = np.genfromtxt("data/y_train.csv", delimiter=",")

# Train a model
clf = RandomForestClassifier().fit(X, y)

# Write the model to a file
if not os.path.isdir("models/"):
    os.mkdir("models")

filename = 'models/model.joblib'
joblib.dump(clf, open(filename, 'wb'))
