"""
Bag of Words Classifier Module.
This module implements a Bag of Words classifier using scikit-learn's CountVectorizer and Logistic Regression. It provides methods to fit the model to training data, predict labels for new data, and predict probabilities for each class.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Class definition for the Bag of Words Classifier
class BagOfWordsClassifier:
    """
    A Bag of Words classifier using scikit-learn's CountVectorizer and Logistic Regression.
    """

    def __init__(
            self,
            max_features: int = 10000,
            ngram_range: tuple = (1, 1),
            random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

        self.classifier = LogisticRegression(random_state=self.random_state, max_iter=1000)

    def fit(self, X, y):
        """ 
        Fit the Bag of Words model to the training data.

        Args:
            X (list): List of text documents.
            y (list): List of labels corresponding to the text documents.

        Returns:
            self: Fitted BagOfWordsClassifier instance.
        """

        X_vectorized = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vectorized, y)
        return self

    def predict(self, X):
        """
        Predict the labels for the given text documents.

        Args:
            X (list): List of text documents.

        Returns:
            list: List of predicted labels.
        """

        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict(X_vectorized)

    def predict_proba(self, X):
        """
        Predict the probabilities for the given text documents.

        Args:
            X (list): List of text documents.
        
        Returns:
            list: List of predicted probabilities for each class.
        """

        X_vectorized = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vectorized)

    