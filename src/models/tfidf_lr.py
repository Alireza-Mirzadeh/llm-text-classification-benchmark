""" 
Tf-idf classifier that uses a logistic regression model for text classification. 
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# TfidfClassifier class definition
class TfidfClassifier(BaseEstimator, ClassifierMixin):
    """ 
    Tf-idf classifier that uses a logistic regression model for text classification.
    """

    def __init__(
            self,
            max_features: int = 10000,
            ngram_range: tuple = (1, 1),
            max_iter: int = 1000,
            random_state: int = 42
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        # Create a pipeline with TfidfVectorizer and LogisticRegression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)),
            ('clf', LogisticRegression(max_iter=self.max_iter, random_state=self.random_state))
        ])

    def fit(self, X, y):
        """ Fit the model to the training data. """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """ Predict the labels for the given data. """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """ Predict the class probabilities for the given data. """
        return self.pipeline.predict_proba(X)
    
