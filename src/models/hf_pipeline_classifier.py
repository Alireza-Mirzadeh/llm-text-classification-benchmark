""" Hugging Face Pipeline Classifier for Taslk-Specific Transformer Sentiment Model """

import logging
import numpy as np
import torch
from transformers import pipeline
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.base import BaseEstimator, ClassifierMixin

class HuggingFacePipelineClassifier(BaseEstimator, ClassifierMixin):
    """ Hugging Face Pipeline Classifier for Task-Specific Transformer Sentiment Model
     This class wraps a Hugging Face pipeline for text classification, allowing it to be used as a scikit-learn compatible classifier.
     
     Args:
        BaseEstimator: Base class for all estimators in scikit-learn.
        ClassifierMixin: Mixin class for all classifiers in scikit-learn.
    
    Returns:
        A scikit-learn compatible classifier that uses a Hugging Face pipeline for text classification.
    """

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
        """ Determine the device to use for inference. 
        Returns:
            int: The device index to use for the Hugging Face pipeline. Returns 0 for GPU if available, -1 for CPU, or the specified device index.
        """

        # If the device is set to "auto", check if CUDA is available and return the appropriate device index.
        if self.device == "auto":
             if torch.cuda.is_available():
                return 0 
            
            # If CUDA is not available, return -1 to indicate CPU usage.
             return -1 

        # If the device is specified as an integer, return it directly.
        return self.device

    def fit (self, X=None, y=None):
        """ Load  the Hugging Face pipeline for the specified model. No training is performed as the model is pre-trained. 
         Args:
            X: Ignored. Present for API consistency by convention.
            y: Ignored. Present for API consistency by convention.
            
        Returns:
            self: Returns the instance itself.
        """

        self.logger.info(f"Loading Hugging Face pipeline for model: {self.model_name}")

        # Initialize the Hugging Face pipeline for text classification with the specified model and tokenizer. The device is determined based on the availability of CUDA or the specified device index.
        self.pipe = pipeline(
            task="text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,
            device=self._get_device()
        )

        return self

    # Predict method to classify the input dataset using the Hugging Face pipeline.
    def predict(self, dataset, verbose=False):
        """ Predict the class labels for the input data X using the Hugging Face pipeline. """

        predictions = []

        # Process the input data in batches and obtain predictions from the pipeline.
        for output in tqdm(self.pipe(KeyDataset(dataset, "text"), batch_size=self.batch_size), desc="Predicting", total=len(dataset)):
            if verbose:
                print(f"Output: {output}")
            scores = {item["label"].lower(): item["score"] for item in output}
            if verbose:
                print(f"Scores: {scores}")
            # Determine the predicted class based on the scores.
            prediction = int(scores["positive"] > scores["negative"])
            if verbose:
                print(f"Prediction: {prediction} \n")
            predictions.append(prediction)

            
        # Return the predictions as a NumPy array
        return np.array(predictions)
