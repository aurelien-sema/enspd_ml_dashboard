
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from data_preparation import DataPreparation
from metrics import (
    accuracy_score_np,
    precision_score_np,
    recall_score_np,
    f1_score_np,
    roc_auc_score_np,
    log_loss_np,
    cross_val_score_simple,
)


class LinearSVM:
    """SVM linéaire optimisé avec hinge loss + régularisation L2."""

    def __init__(self, lr: float = 0.01, c: float = 1.0, n_iter: int = 1500):
        self.lr = lr  # taux d'apprentissage
        self.c = c  # coefficient de régularisation C (pénalise les erreurs)
        self.n_iter = n_iter  # nombre d'itérations
        self.weights = None  # poids linéaires
        self.bias = 0.0
        self.train_cost_history = []  # historique du coût d'entraînement (hinge loss + L2)
        self.val_cost_history = []  # historique du coût de validation

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Descente de gradient sur la hinge loss avec régularisation L2
        et early stopping sur une petite validation interne.
        """
        # Convertit y en {-1, 1} pour la hinge loss
        y_signed_full = np.where(y == 1, 1, -1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Split interne train/validation
        rng = np.random.default_rng(42)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        split = max(1, int(0.8 * n_samples))
        train_idx, val_idx = indices[:split], indices[split:]
        X_train, y_train_signed = X[train_idx], y_signed_full[train_idx]
        X_val, y_val_signed = (X[val_idx], y_signed_full[val_idx]) if len(val_idx) > 0 else (None, None)

        best_val_loss = float("inf")
        best_w = None
        best_b = None
        patience = 50
        no_improve = 0
        tol = 1e-4
        n_train = len(train_idx)
        
        # Réinitialiser les historiques
        self.train_cost_history = []
        self.val_cost_history = []

        for iter_num in range(self.n_iter):
            margins = y_train_signed * (X_train @ self.weights + self.bias)
            mask = margins < 1  # erreurs ou sur le bord

            grad_w = self.weights - self.c * np.sum((mask * y_train_signed)[:, None] * X_train, axis=0) / n_train
            grad_b = -self.c * np.sum(mask * y_train_signed) / n_train

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            # Enregistrer le coût d'entraînement (hinge loss + L2)
            hinge_losses_train = np.maximum(0.0, 1 - margins)
            train_loss = float(0.5 * np.sum(self.weights ** 2) + self.c * np.mean(hinge_losses_train))
            self.train_cost_history.append(train_loss)

            # Early stopping : hinge loss + pénalité L2 sur la validation
            if X_val is not None and len(val_idx) > 0:
                margins_val = y_val_signed * (X_val @ self.weights + self.bias)
                hinge_losses = np.maximum(0.0, 1 - margins_val)
                val_loss = float(0.5 * np.sum(self.weights ** 2) + self.c * np.mean(hinge_losses))
                self.val_cost_history.append(val_loss)
                if val_loss + tol < best_val_loss:
                    best_val_loss = val_loss
                    best_w = self.weights.copy()
                    best_b = float(self.bias)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            else:
                self.val_cost_history.append(None)

        # Recharge les meilleurs poids
        if best_w is not None:
            self.weights = best_w
            self.bias = best_b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Projection sigmoïde pour obtenir une pseudo-proba
        logits = self.decision_function(X)
        proba = 1 / (1 + np.exp(-logits))
        return proba


class SVMModel:
    def __init__(self):
        self.model = LinearSVM()
        self.predictions = None
        self.probabilities = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)

        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)

        train_acc = accuracy_score_np(y_train, self.model.predict(X_train))
        test_acc = accuracy_score_np(y_test, self.predictions)
        precision = precision_score_np(y_test, self.predictions)
        recall = recall_score_np(y_test, self.predictions)
        f1 = f1_score_np(y_test, self.predictions)
        auc_roc = roc_auc_score_np(y_test, self.probabilities)
        train_ll = float(np.sum(np.log(np.clip(self.model.predict_proba(X_train), 1e-9, 1))))
        test_ll = float(np.sum(np.log(np.clip(self.probabilities, 1e-9, 1))))
        train_cost = log_loss_np(y_train, self.model.predict_proba(X_train))
        test_cost = log_loss_np(y_test, self.probabilities)

        cv_mean, cv_std = cross_val_score_simple(
            lambda: LinearSVM(self.model.lr, self.model.c, self.model.n_iter),
            X_train,
            y_train,
            accuracy_score_np,
            k=3,
        )

        print("\n MÉTRIQUES :")
        print(f"   Accuracy Train/Test: {train_acc:.4f} / {test_acc:.4f}")
        print(f"   Precision/Recall/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        print(f"   Log-vraisemblance Train/Test: {train_ll:.4f} / {test_ll:.4f}")
        print(f"   Log Loss Train/Test: {train_cost:.4f} / {test_cost:.4f}")
        print(f"   CV Accuracy (3 folds): {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"   Nombre de vecteurs de support (approx via marge<1): {int(np.sum(self.model.decision_function(X_train) * (2 * y_train - 1) < 1))}")

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "train_log_likelihood": train_ll,
            "test_log_likelihood": test_ll,
            "train_cost": train_cost,
            "test_cost": test_cost,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "train_cost_history": self.model.train_cost_history,
            "val_cost_history": self.model.val_cost_history,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    model = SVMModel()
    results = model.train_and_evaluate(
        data_prep.X_train, data_prep.X_test, data_prep.y_train_class, data_prep.y_test_class
    )
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1)

    # Courbe ROC
    sorted_idx = np.argsort(-model.probabilities)
    y_sorted = data_prep.y_test_class[sorted_idx]
    proba_sorted = model.probabilities[sorted_idx]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = tps / (tps[-1] + 1e-12)
    fpr = fps / (fps[-1] + 1e-12)
    ax1.plot(fpr, tpr, label=f"AUC = {results['auc_roc']:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_title("Courbe ROC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    # Barres de métriques
    metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
    metrics_values = [
        results["test_accuracy"],
        results["precision"],
        results["recall"],
        results["f1_score"],
    ]
    ax2.bar(metrics_names, metrics_values, color=["steelblue", "seagreen", "darkorange", "purple"])
    ax2.set_ylim(0, 1)
    for i, v in enumerate(metrics_values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    ax2.set_title("Métriques principales (test)")
    ax2.grid(True, alpha=0.3, axis="y")


    ax3 = plt.subplot(2, 2, 3)
    # Évolution du coût (Hinge Loss + L2)
    if model.model.train_cost_history:
        iterations = range(len(model.model.train_cost_history))
        ax3.plot(iterations, model.model.train_cost_history, label="Train Loss", color="steelblue", linewidth=2)
        val_losses = [v for v in model.model.val_cost_history if v is not None]
        if val_losses:
            val_iterations = [i for i, v in enumerate(model.model.val_cost_history) if v is not None]
            ax3.plot(val_iterations, val_losses, label="Validation Loss", color="indianred", linewidth=2)
        ax3.set_xlabel("Itération")
        ax3.set_ylabel("Coût (Hinge Loss + L2)")
        ax3.set_title("Évolution du coût")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    # Graphe d'importance des features basé sur les poids
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    if model.model.weights is not None:
        feature_importance = np.abs(model.model.weights)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-12)
    else:
        feature_importance = np.zeros(len(feature_names))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(feature_names)))
    ax4.barh(feature_names, feature_importance, color=colors)
    ax4.set_xlabel("Importance (magnitude des poids)")
    ax4.set_title("Importance des Features - SVM Linéaire")
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("svm_analyse.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    main()
