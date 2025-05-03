from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    print(logits, labels)
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    # recall = recall_score(y_true=labels, y_pred=predictions)
    # precision = precision_score(y_true=labels, y_pred=predictions)
    # f1 = f1_score(y_true=labels, y_pred=predictions)

    return {
        "accuracy": accuracy,
        # "recall": recall,
        # "precision": precision,
        # "f1": f1
    }
