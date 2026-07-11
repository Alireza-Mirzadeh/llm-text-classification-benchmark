""" Hugging Face Pipeline Classifier """

import logging
import numpy as np
import torch
from transformers import pipeline
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin

class HuggingFacePipelineClassifier(BaseEstimator, ClassifierMixin):
    """ Hugging Face Pipeline Classifier """

    def __init__(self, model_name: str, batch_size: int = 32, device = "auto"):
        """ 
        Initialize the Hugging Face Pipeline Classifier.

        Args:
            model_name (str): The name of the pre-trained model to use.
            batch_size (int): The number of samples to process in each batch.
            device (str or int): The device to run the model on. Use "auto" for automatic selection.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize the Hugging Face pipeline to None; it will be loaded during fitting.
        self.pipe = None

    def _get_device(self):
        """ Determine the device to use for inference."""

        # If the device is set to "auto", check if CUDA is available and return the appropriate device index.
        if self.device == "auto":
             if torch.cuda.is_available():
                return 0 
            
            # If CUDA is not available, return -1 to indicate CPU usage.
             return -1 

        # If the device is specified as an integer, return it directly.
        return self.device

    def fit (self, X=None, y=None):
        """ Load  the Hugging Face pipeline for the specified model. No training is performed as the model is pre-trained. """

        self.logger.info(f"Loading Hugging Face pipeline for model: {self.model_name}")

        self.pipe = pipeline(
            task="text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            return_all_scores=True,
            device=self._get_device()
        )

        return self

    def predict(self, X):
        """ Predict the class labels for the input data X using the Hugging Face pipeline. """

        predictions = []

        # Use the pipeline to get predictions for the input data X in batches
        outputs = self.pipe(X, batch_size=self.batch_size, truncation=True)

        # Iterate through the outputs and extract the predicted class labels based on the scores
        for output in outputs:

            negative_score = output[0]['score']
            positive_score = output[1]['score']

            # Append the predicted class label (0 for negative, 1 for positive) based on the scores
            predictions.append(np.argmax([negative_score, positive_score]))

        # Return the predictions as a NumPy array
        return np.array(predictions)
