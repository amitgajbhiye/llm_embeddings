import json
import logging
import numpy as np
from scipy.optimize import brute
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def f1(th, y_true, y_score):
    # th: threshold
    # y_true: true labels
    # y_pred: predicted labels
    y_pred = (y_score >= th) * 1
    return -f1_score(y_true, y_pred)


def optimal_threshold(y_true, y_score):
    # y_true: true labels
    # y_pred: predicted labels
    bounds = [(np.min(y_score), np.max(y_score))]
    result = brute(f1, args=(y_true, y_score), ranges=bounds, full_output=True, Ns=200)
    return result[0][0], -f1(result[0][0], y_true, y_score)


def pre_rec_f1(y_true, y_pred):
    # y_true: true labels
    # y_pred: predicted labels
    return (
        round(precision_score(y_true, y_pred), 4),
        round(recall_score(y_true, y_pred), 4),
        round(f1_score(y_true, y_pred), 4),
    )


def read_config(config_file):
    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            config_dict = json.load(json_file)
            return config_dict
    else:
        return config_file


def f1(th, y_true, y_score):
    # th: threshold
    # y_true: true labels
    # y_pred: predicted labels
    y_pred = (y_score >= th) * 1
    return -f1_score(y_true, y_pred)


# find the best threshold for classification
def optimal_threshold(y_true, y_score):
    # y_true: true labels
    # y_pred: predicted labels
    bounds = [(np.min(y_score), np.max(y_score))]
    result = brute(f1, args=(y_true, y_score), ranges=bounds, full_output=True, Ns=200)
    return result[0][0], -f1(result[0][0], y_true, y_score)


def compute_scores(labels, preds):
    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

    scores = {
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores
