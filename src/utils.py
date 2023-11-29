import json
import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def read_config(config_file):
    if isinstance(config_file, str):
        with open(config_file, "r") as json_file:
            config_dict = json.load(json_file)
            return config_dict
    else:
        return config_file


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
