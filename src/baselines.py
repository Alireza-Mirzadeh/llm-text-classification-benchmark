# Baseline model that predicts the majority class for all samples.

import numpy as np

class MajorityClassBaseline:

    def fit(self, x, y):
        # Store the majority class label
        self.majority_class = np.bincount(y).argmax()

    def predict(self, x):
        # Predict the majority class for all the samples
        return np.full(len(x), self.majority_class)
