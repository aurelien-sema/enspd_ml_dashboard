
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
    confusion_matrix_np,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionModel:
    """Régression logistique binaire sans sklearn."""

    def __init__(self, lr: float = 0.05, n_iter: int = 2000, reg_lambda: float = 0.0):
        self.lr = lr  # taux d'apprentissage
        self.n_iter = n_iter  # nombre d'itérations d'entraînement
        self.reg_lambda = reg_lambda  # régularisation L2
        self.weights = None  # vecteur de poids
        self.bias = 0.0  # biais
        self.predictions = None  # dernières prédictions de classes
        self.probabilities = None  # dernières probabilités
        self.train_cost_history = []  # historique du coût d'entraînement
        self.val_cost_history = []  # historique du coût de validation

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraînement par descente de gradient avec régularisation L2
        et early stopping basé sur une petite validation interne (pour limiter l'overfitting).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Split interne train/validation (80/20) pour early stopping
        rng = np.random.default_rng(42)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        split = max(1, int(0.8 * n_samples))
        train_idx, val_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx] if len(val_idx) > 0 else (None, None)

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
            linear = X_train @ self.weights + self.bias
            y_proba = sigmoid(linear)

            error = y_proba - y_train
            grad_w = (1 / n_train) * (X_train.T @ error) + self.reg_lambda * self.weights
            grad_b = (1 / n_train) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            # Enregistrer le coût d'entraînement
            train_loss = log_loss_np(y_train, y_proba)
            self.train_cost_history.append(train_loss)

            # Early stopping si on a un set de validation
            if X_val is not None and len(val_idx) > 0:
                val_proba = sigmoid(X_val @ self.weights + self.bias)
                # log_loss binaire sur la validation
                val_loss = log_loss_np(y_val, val_proba)
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

        # Recharge les meilleurs poids trouvés sur la validation
        if best_w is not None:
            self.weights = best_w
            self.bias = best_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def compute_likelihood(self, X: np.ndarray, y_true: np.ndarray) -> float:
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-12, 1 - 1e-12)
        return float(np.sum(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))

    def compute_cost(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return log_loss_np(y_true, self.predict_proba(X))

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.fit(X_train, y_train)

        self.predictions = self.predict(X_test)
        self.probabilities = self.predict_proba(X_test)

        train_acc = accuracy_score_np(y_train, self.predict(X_train))
        test_acc = accuracy_score_np(y_test, self.predictions)
        precision = precision_score_np(y_test, self.predictions)
        recall = recall_score_np(y_test, self.predictions)
        f1 = f1_score_np(y_test, self.predictions)
        auc_roc = roc_auc_score_np(y_test, self.probabilities)

        train_ll = self.compute_likelihood(X_train, y_train)
        test_ll = self.compute_likelihood(X_test, y_test)
        train_cost = self.compute_cost(X_train, y_train)
        test_cost = self.compute_cost(X_test, y_test)

        cv_mean, cv_std = cross_val_score_simple(
            lambda: LogisticRegressionModel(self.lr, self.n_iter, self.reg_lambda),
            X_train,
            y_train,
            accuracy_score_np,
            k=5,
        )

        cm = confusion_matrix_np(y_test, self.predictions)

        print("\n MÉTRIQUES :")
        print(f"   Accuracy Train/Test: {train_acc:.4f} / {test_acc:.4f}")
        print(f"   Precision/Recall/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        print(f"   Log-vraisemblance Train/Test: {train_ll:.4f} / {test_ll:.4f}")
        print(f"   Log Loss Train/Test: {train_cost:.4f} / {test_cost:.4f}")
        print(f"   CV Accuracy (5 folds): {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"   Matrice de confusion: \n{cm}")

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
            "confusion_matrix": cm,
            "train_cost_history": self.train_cost_history,
            "val_cost_history": self.val_cost_history,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    model = LogisticRegressionModel(lr=0.05, n_iter=2000, reg_lambda=0.01)
    results = model.train_and_evaluate(
        data_prep.X_train, data_prep.X_test, data_prep.y_train_class, data_prep.y_test_class
    )

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1)

    # Courbe ROC approximée
    sorted_idx = np.argsort(-model.probabilities)
    y_sorted = data_prep.y_test_class[sorted_idx]
    proba_sorted = model.probabilities[sorted_idx]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = tps / (tps[-1] + 1e-12)
    fpr = fps / (fps[-1] + 1e-12)
    ax1.plot(fpr, tpr, label=f"AUC = {results['auc_roc']:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("Faux positifs")
    ax1.set_ylabel("Vrais positifs")
    ax1.set_title("Courbe ROC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    # Barres des métriques
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
    # Évolution du coût (Log Loss)
    if model.train_cost_history:
        iterations = range(len(model.train_cost_history))
        ax3.plot(iterations, model.train_cost_history, label="Train Loss", color="steelblue", linewidth=2)
        val_losses = [v for v in model.val_cost_history if v is not None]
        if val_losses:
            val_iterations = [i for i, v in enumerate(model.val_cost_history) if v is not None]
            ax3.plot(val_iterations, val_losses, label="Validation Loss", color="indianred", linewidth=2)
        ax3.set_xlabel("Itération")
        ax3.set_ylabel("Log Loss")
        ax3.set_title("Évolution du Log Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    # Graphe d'importance des features basé sur les poids
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    if model.weights is not None:
        feature_importance = np.abs(model.weights)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-12)
    else:
        feature_importance = np.zeros(len(feature_names))
    colors = plt.cm.twilight(np.linspace(0, 1, len(feature_names)))
    ax4.barh(feature_names, feature_importance, color=colors)
    ax4.set_xlabel("Importance (magnitude des poids)")
    ax4.set_title("Importance des Features - Régression Logistique")
    ax4.grid(True, alpha=0.3, axis="x")
    

    plt.tight_layout()
    plt.savefig("regression_logistique_analyse.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    main()
