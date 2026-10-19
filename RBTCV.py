import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)


def rbtc_cv(
    X,
    y,
    estimator,
    classes,
    cost_matrix,
    n_splits=3,
    gap=2,
    test_size=None,
    initial_train_fraction=0.5
):
    """
    Risk-Aware Blocked Temporal Cross-Validation (RBTCV).

    Training observations always precede the test block.
    An optional gap separates training and testing.

    Parameters
    ----------
    X : array-like
        Features ordered chronologically.

    y : array-like
        Target labels ordered chronologically.

    estimator : sklearn estimator
        Model implementing fit() and predict().

    classes : list
        Complete list of class labels, in cost-matrix order.

    cost_matrix : array-like
        Rows = true classes.
        Columns = predicted classes.

    n_splits : int
        Number of future test blocks.

    gap : int
        Number of observations excluded immediately
        before each test block.

    test_size : int or None
        Number of observations per test block.
        If None, it is selected automatically.

    initial_train_fraction : float
        Fraction of the data used to determine the
        initial training-window size.

    Returns
    -------
    Dictionary containing fold results, pooled
    predictions, and aggregate evaluation metrics.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = list(classes)
    cost_matrix = np.asarray(cost_matrix, dtype=float)

    n = len(y)

    if len(X) != n:
        raise ValueError("X and y must have equal lengths.")

    if n_splits < 1 or gap < 0:
        raise ValueError("Invalid n_splits or gap.")

    if not 0 < initial_train_fraction < 1:
        raise ValueError(
            "initial_train_fraction must be between 0 and 1."
        )

    if len(classes) != cost_matrix.shape[0]:
        raise ValueError("Cost-matrix row count is incorrect.")

    if cost_matrix.shape != (len(classes), len(classes)):
        raise ValueError("Cost matrix must be square.")

    class_to_index = {
        label: i for i, label in enumerate(classes)
    }

    unknown = set(y) - set(classes)

    if unknown:
        raise ValueError(
            f"Labels missing from classes: {unknown}"
        )

    # Determine initial training size.
    initial_train_size = max(
        2,
        int(n * initial_train_fraction)
    )

    # Choose test-block size if not supplied.
    if test_size is None:
        test_size = (
            n - initial_train_size - gap
        ) // n_splits

    if test_size < 1:
        raise ValueError(
            "Insufficient data for the requested windows."
        )

    required = (
        initial_train_size
        + gap
        + n_splits * test_size
    )

    if required > n:
        raise ValueError(
            "Not enough observations. Reduce n_splits, "
            "test_size, gap, or initial_train_fraction."
        )

    fold_results = []
    all_true = []
    all_pred = []
    all_losses = []

    # ------------------------------------------------
    # Expanding-window chronological evaluation
    # ------------------------------------------------

    for fold in range(n_splits):

        test_start = (
            initial_train_size
            + gap
            + fold * test_size
        )

        train_end = test_start - gap
        test_end = test_start + test_size

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        # Fit a fresh copy of the estimator.
        model = clone(estimator)
        model.fit(X_train, y_train)

        # Predict the future block.
        y_pred = model.predict(X_test)

        # Ordinary classification error.
        ordinary_error = 1.0 - accuracy_score(
            y_test, y_pred
        )

        # Cost-sensitive operational error.
        losses = np.array([
            cost_matrix[
                class_to_index[true_label],
                class_to_index[pred_label]
            ]
            for true_label, pred_label
            in zip(y_test, y_pred)
        ])

        risk_error = float(np.mean(losses))

        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0
        )

        fold_results.append({
            "fold": fold + 1,
            "train_size": len(train_indices),
            "gap_size": gap,
            "test_size": len(test_indices),
            "ordinary_error": ordinary_error,
            "risk_error": risk_error,
            "macro_f1": macro_f1,
            "train_end_index": train_end - 1,
            "test_start_index": test_start,
            "test_end_index": test_end - 1
        })

        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_losses.extend(losses)

    # ------------------------------------------------
    # Aggregate across all evaluated observations
    # ------------------------------------------------

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)
    all_losses = np.asarray(all_losses)

    pooled_recall = recall_score(
        all_true,
        all_pred,
        labels=classes,
        average=None,
        zero_division=0
    )

    return {
        "fold_results": pd.DataFrame(fold_results),

        "ordinary_error": float(
            1.0 - accuracy_score(all_true, all_pred)
        ),

        "risk_error": float(
            np.mean(all_losses)
        ),

        "macro_f1": float(
            f1_score(
                all_true,
                all_pred,
                labels=classes,
                average="macro",
                zero_division=0
            )
        ),

        "class_recall": dict(
            zip(classes, pooled_recall)
        ),

        "confusion_matrix": confusion_matrix(
            all_true,
            all_pred,
            labels=classes
        ),

        "y_true": all_true,
        "y_pred": all_pred
    }
