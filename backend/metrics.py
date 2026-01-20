
from __future__ import annotations

import math
import numpy as np
from typing import Callable, Tuple


# --------------------------
# Fonctions de prétraitement
# --------------------------

def train_test_split_simple(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Coupe aléatoirement X et y en train/test sans stratification.
    Utilisé pour la régression où l'équilibre des classes importe moins.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Découpe en train/test en conservant les proportions de classes.
    Adapté aux tâches de classification binaire.
    """
    rng = np.random.default_rng(seed)
    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]
    rng.shuffle(class0)
    rng.shuffle(class1)

    def _split(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        split = int(len(indices) * (1 - test_size))
        return indices[:split], indices[split:]

    c0_train, c0_test = _split(class0)
    c1_train, c1_test = _split(class1)

    train_idx = np.concatenate([c0_train, c1_train])
    test_idx = np.concatenate([c0_test, c1_test])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centre et réduit X, retourne X_norm, mean, std pour réutilisation sur le test.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # évite la division par zéro
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applique le centrage-réduction avec les paramètres appris sur le train."""
    return (X - mean) / (std + 1e-8)


# --------------
# Métriques base
# --------------

def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp + 1e-12))


def recall_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn + 1e-12))


def f1_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision_score_np(y_true, y_pred)
    r = recall_score_np(y_true, y_pred)
    return float(2 * p * r / (p + r + 1e-12))


def log_loss_np(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Log loss binaire. y_proba est la proba de la classe 1.
    """
    y_proba = np.clip(y_proba, 1e-12, 1 - 1e-12)
    return float(-np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)))


def roc_auc_score_np(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    AUC binaire via la méthode du trapèze sur la courbe ROC.
    """
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]
    y_proba_sorted = y_proba[order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / (tps[-1] + 1e-12)
    fpr = fps / (fps[-1] + 1e-12)

    # Ajoute le point (0,0) pour fermer la courbe
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    return float(np.trapz(tpr, fpr))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Retourne la matrice 2x2 [[tn, fp], [fn, tp]]."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


# -----------------
# Métriques régress.
# -----------------

def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def explained_variance_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.var(y_true - y_pred)
    den = np.var(y_true) + 1e-12
    return float(1 - num / den)


# -------------------------
# Validation croisée simple
# -------------------------

def kfold_indices(n_samples: int, k: int = 5, seed: int = 42):
    """Génère des couples (train_idx, test_idx) pour une K-fold simple."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def cross_val_score_simple(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    k: int = 5,
) -> Tuple[float, float]:
    """
    Validation croisée générique : crée un modèle neuf à chaque fold via model_factory.
    Retourne (moyenne, écart-type) du score.
    """
    scores = []
    for train_idx, test_idx in kfold_indices(len(X), k=k):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(metric_fn(y[test_idx], preds))
    return float(np.mean(scores)), float(np.std(scores))
