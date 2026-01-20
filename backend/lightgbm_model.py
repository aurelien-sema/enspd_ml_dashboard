from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from data_preparation import DataPreparation
from regression_logistique import LogisticRegressionModel
from metrics import (
    accuracy_score_np,
    precision_score_np,
    recall_score_np,
    f1_score_np,
    roc_auc_score_np,
    log_loss_np,
    cross_val_score_simple,
)


class LightGBMModel:
    """
    Version « sans lightgbm » du modèle, utilisant une régression logistique maison.
    L'interface (attributs, méthodes) reste la même que dans lightgbm_model.py.
    """

    def __init__(self):
        self.model = LogisticRegressionModel(
            lr=0.05,
            n_iter=2000,
            reg_lambda=0.01,
        )
        self.predictions = None
        self.probabilities = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        y_test = np.asarray(y_test, dtype=int)

        # Entraînement
        self.model.fit(X_train, y_train)

        # Prédictions
        self.predictions = self.model.predict(X_test)
        self.probabilities = self.model.predict_proba(X_test)

        # Métriques
        train_preds = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)

        train_acc = accuracy_score_np(y_train, train_preds)
        test_acc = accuracy_score_np(y_test, self.predictions)
        precision = precision_score_np(y_test, self.predictions)
        recall = recall_score_np(y_test, self.predictions)
        f1 = f1_score_np(y_test, self.predictions)
        auc_roc = roc_auc_score_np(y_test, self.probabilities)

        train_ll = float(np.sum(np.log(np.clip(train_proba, 1e-9, 1))))
        test_ll = float(np.sum(np.log(np.clip(self.probabilities, 1e-9, 1))))
        train_cost = log_loss_np(y_train, train_proba)
        test_cost = log_loss_np(y_test, self.probabilities)

        # Validation croisée simple (sur le train uniquement)
        def factory():
            return LogisticRegressionModel(
                lr=self.model.lr,
                n_iter=self.model.n_iter,
                reg_lambda=self.model.reg_lambda,
            )

        cv_mean, cv_std = cross_val_score_simple(
            factory,
            X_train,
            y_train,
            accuracy_score_np,
            k=3,
        )

        # Importance des features : on prend la magnitude des poids
        if self.model.weights is not None:
            w = np.abs(self.model.weights)
            feature_importance = w / (w.sum() + 1e-12)
        else:
            feature_importance = np.zeros(X_train.shape[1], dtype=float)

        print("\n MÉTRIQUES (modèle logistique):")
        print(f"   Accuracy Train/Test: {train_acc:.4f} / {test_acc:.4f}")
        print(f"   Precision/Recall/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        print(f"   Log-vraisemblance Train/Test: {train_ll:.4f} / {test_ll:.4f}")
        print(f"   Log Loss Train/Test: {train_cost:.4f} / {test_cost:.4f}")
        print(f"   CV Accuracy (3 folds): {cv_mean:.4f} (+/- {cv_std:.4f})")

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
            "feature_importance": feature_importance,
            "train_cost_history": self.model.train_cost_history,
            "val_cost_history": self.model.val_cost_history,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    model = LightGBMModel()
    results = model.train_and_evaluate(
        data_prep.X_train, data_prep.X_test, data_prep.y_train_class, data_prep.y_test_class
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Courbe ROC
    sorted_idx = np.argsort(-model.probabilities)
    y_sorted = data_prep.y_test_class[sorted_idx]
    proba_sorted = model.probabilities[sorted_idx]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = tps / (tps[-1] + 1e-12)
    fpr = fps / (fps[-1] + 1e-12)
    axes[0].plot(fpr, tpr, label=f"AUC = {results['auc_roc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("Courbe ROC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

     # Graphe d'importance des features en grand
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    colors = plt.cm.Spectral(np.linspace(0, 1, len(feature_names)))
    axes[1].barh(feature_names, results["feature_importance"], color=colors)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Importance des Features - LightGBM (Logistique)")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Évolution du coût (Log Loss)
    if model.model.train_cost_history:
        iterations = range(len(model.model.train_cost_history))
        axes[2].plot(iterations, model.model.train_cost_history, label="Train Loss", color="steelblue", linewidth=2)
        val_losses = [v for v in model.model.val_cost_history if v is not None]
        if val_losses:
            val_iterations = [i for i, v in enumerate(model.model.val_cost_history) if v is not None]
            axes[2].plot(val_iterations, val_losses, label="Validation Loss", color="indianred", linewidth=2)
        axes[2].set_xlabel("Itération")
        axes[2].set_ylabel("Log Loss")
        axes[2].set_title("Évolution du Log Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("lightgbm_analyse.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
