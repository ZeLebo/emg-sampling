"""LDA model helpers for baseline experiments."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score


def train_lda(X_train: np.ndarray, y_train: np.ndarray) -> LinearDiscriminantAnalysis:
    """Fits LDA classifier on feature matrix."""
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    clf: LinearDiscriminantAnalysis,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Returns accuracy and macro_f1 for the classifier.

    Macro-F1 is used because classes should contribute equally to final score.
    """
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
