""" 
Evaluation module for the project. This module provides functions to evaluate the performance of classification models using various metrics and visualizations.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt



# Evaluation function to assess the performance of a classifier.
def evaluate_classifier(y_true, y_pred):

    """ 
    Evaluate the performance of a classifier and return metrics for benchmarking along with a detailed classification report. 

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    
    Returns:
        metrics (dict): Dictionary containing accuracy, precision, recall, and F1 score.
        report (dict): Detailed classification report.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "positive"], 
        output_dict=True,
        zero_division=0
    )

    return metrics, report

# Function to plot a confusion matrix for the given model and predictions.
def plot_confusion_matrix(model_name, y_true, y_pred):

    """
    Plot a confusion matrix for the given model and predictions.

    Args:
        model_name (str): Name of the model.
        y_true (list): True labels.
        y_pred (list): Predicted labels.

    Returns:
        fig (matplotlib.figure.Figure): The confusion matrix figure.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    return fig

