"""TF-IDF with Linear SVM Classifier model implementation."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Inherit from BaseEstimator and ClassifierMixin to create a custom classifier
class TFIDFLinearSVM(BaseEstimator, ClassifierMixin):
    """TF-IDF with Linear SVM Classifier"""

    def __init__(
        self,
        max_features=10000,
        ngram_range=(1, 1),
        C=1.0,
        random_state=42,
    ):

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(max_features=max_features, ngram_range=ngram_range),
                ),
                (
                    "svm",
                     LinearSVC(C=C, random_state=random_state)
                ),
            ]
        )

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X (array-like): Training data.
            y (array-like): Target labels.

        Returns:
            self: Fitted model.
        """

        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the labels for the given data.

        Args:
            X (array-like): Data to predict.

        Returns:
            array-like: Predicted labels.
        """

        return self.pipeline.predict(X)
