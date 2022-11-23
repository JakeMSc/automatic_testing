import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import json

model = joblib.load(open("models/model.joblib", "rb"))

X_test = np.genfromtxt("data/X_test.csv", delimiter=",")
y_test = np.genfromtxt("data/y_test.csv", delimiter=",")

# Test on the model
y_hat = model.predict(X_test)

acc = np.mean(y_hat==y_test)
cm = confusion_matrix(y_test, y_hat, labels=model.classes_)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('confusion.png')