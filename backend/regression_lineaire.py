
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from data_preparation import DataPreparation
from metrics import (
    mean_absolute_error_np,
    mean_squared_error_np,
    r2_score_np,
    explained_variance_score_np,
    cross_val_score_simple,
)


class LinearRegressionModel:
    """Modèle linéaire entraîné par descente de gradient (avec L1 ou L2)."""

    def __init__(self, model_type: str = "linear", lr: float = 0.01, n_iter: int = 2000, alpha: float = 1.0):
        self.model_type = model_type  # linear | ridge | lasso
        self.lr = lr  # taux d'apprentissage pour la descente de gradient
        self.n_iter = n_iter  # nombre d'itérations d'optimisation
        self.alpha = alpha  # coefficient de régularisation
        self.weights = None  # vecteur de coefficients appris
        self.bias = 0.0  # ordonnée à l'origine
        self.train_cost_history = []  # historique du coût d'entraînement (MSE)
        self.val_cost_history = []  # historique du coût de validation (MSE)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Apprend les poids via descente de gradient pleine, avec régularisation
        (ridge/lasso) et early stopping sur une validation interne (80/20).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Split interne train/validation pour limiter l'overfitting
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
        tol = 1e-5
        n_train = len(train_idx)
        
        # Réinitialiser les historiques
        self.train_cost_history = []
        self.val_cost_history = []

        for iter_num in range(self.n_iter):
            y_pred = X_train @ self.weights + self.bias
            error = y_pred - y_train

            # Gradient MSE
            grad_w = (2 / n_train) * (X_train.T @ error)
            grad_b = (2 / n_train) * np.sum(error)

            if self.model_type == "ridge":
                grad_w += 2 * self.alpha * self.weights
            elif self.model_type == "lasso":
                grad_w += self.alpha * np.sign(self.weights)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            # Enregistrer le coût d'entraînement (MSE)
            train_loss = float(np.mean(error ** 2))
            self.train_cost_history.append(train_loss)

            # Early stopping sur la MSE de validation
            if X_val is not None and len(val_idx) > 0:
                val_pred = X_val @ self.weights + self.bias
                val_loss = float(np.mean((y_val - val_pred) ** 2))
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

        # Recharge les meilleurs poids sur la validation
        if best_w is not None:
            self.weights = best_w
            self.bias = best_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def compute_likelihood(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Log-vraisemblance sous hypothèse d'erreur gaussienne."""
        y_pred = self.predict(X)
        residuals = y_true - y_pred
        sigma = np.std(residuals) + 1e-8
        n = len(y_true)
        return float(-n / 2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2))

    def compute_cost(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return mean_squared_error_np(y_true, self.predict(X))

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.fit(X_train, y_train)

        preds_test = self.predict(X_test)
        preds_train = self.predict(X_train)

        mae = mean_absolute_error_np(y_test, preds_test)
        mse = mean_squared_error_np(y_test, preds_test)
        rmse = float(np.sqrt(mse))
        r2_train = r2_score_np(y_train, preds_train)
        r2_test = r2_score_np(y_test, preds_test)
        evs = explained_variance_score_np(y_test, preds_test)

        train_ll = self.compute_likelihood(X_train, y_train)
        test_ll = self.compute_likelihood(X_test, y_test)
        train_cost = self.compute_cost(X_train, y_train)
        test_cost = self.compute_cost(X_test, y_test)

        cv_mean, cv_std = cross_val_score_simple(
            lambda: LinearRegressionModel(self.model_type, self.lr, self.n_iter, self.alpha),
            X_train,
            y_train,
            r2_score_np,
            k=5,
        )

        print("\n📊 MÉTRIQUES DE PERFORMANCE :")
        print(f"   R² Train/Test: {r2_train:.4f} / {r2_test:.4f}")
        print(f"   MAE/MSE/RMSE: {mae:.4f} / {mse:.4f} / {rmse:.4f}")
        print(f"   Explained Variance: {evs:.4f}")
        print(f"   Log-vraisemblance Train/Test: {train_ll:.4f} / {test_ll:.4f}")
        print(f"   MSE Cost Train/Test: {train_cost:.4f} / {test_cost:.4f}")
        print(f"   CV R² (5 folds): {cv_mean:.4f} (+/- {cv_std:.4f})")

        return {
            "train_r2": r2_train,
            "test_r2": r2_test,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "evs": evs,
            "train_log_likelihood": train_ll,
            "test_log_likelihood": test_ll,
            "train_cost": train_cost,
            "test_cost": test_cost,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "coefficients": self.weights,
            "intercept": self.bias,
            "train_cost_history": self.train_cost_history,
            "val_cost_history": self.val_cost_history,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    models = ["linear", "ridge", "lasso"]
    results = {}

    for model_type in models:
        print("\n" + "=" * 60)
        print(f"Test du modèle: {model_type.upper()} ")
        model = LinearRegressionModel(model_type=model_type, lr=0.05, n_iter=1500, alpha=0.1)
        results[model_type] = model.train_and_evaluate(
            data_prep.X_train, data_prep.X_test, data_prep.y_train_reg, data_prep.y_test_reg
        )

    # Visualisation des R², RMSE et évolution du coût
    fig = plt.figure(figsize=(15, 10))
    
    # Graphique 1: R² par modèle
    ax1 = plt.subplot(2, 2, 1)
    model_names = ["Linear", "Ridge", "Lasso"]
    train_r2 = [results[m]["train_r2"] for m in models]
    test_r2 = [results[m]["test_r2"] for m in models]
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width / 2, train_r2, width, label="Train", color="steelblue", alpha=0.7)
    ax1.bar(x + width / 2, test_r2, width, label="Test", color="indianred", alpha=0.7)
    ax1.set_title("R² par modèle")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Graphique 2: RMSE par modèle
    ax2 = plt.subplot(2, 2, 2)
    rmse_vals = [results[m]["rmse"] for m in models]
    ax2.bar(model_names, rmse_vals, color=["steelblue", "seagreen", "darkorange"])
    ax2.set_title("RMSE par modèle")
    for i, v in enumerate(rmse_vals):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    ax2.grid(True, alpha=0.3, axis="y")

    # Graphique 3: Évolution du coût (MSE) pour chaque modèle
    ax3 = plt.subplot(2, 2, (3, 4))
    colors = ["steelblue", "seagreen", "darkorange"]
    for idx, model_type in enumerate(models):
        train_history = results[model_type].get("train_cost_history", [])
        val_history = results[model_type].get("val_cost_history", [])
        if train_history:
            iterations = range(len(train_history))
            ax3.plot(iterations, train_history, 
                    label=f"{model_type.capitalize()} - Train", 
                    color=colors[idx], linewidth=2, linestyle="-")
            val_losses = [v for v in val_history if v is not None]
            if val_losses:
                val_iterations = [i for i, v in enumerate(val_history) if v is not None]
                ax3.plot(val_iterations, val_losses, 
                        label=f"{model_type.capitalize()} - Validation", 
                        color=colors[idx], linewidth=2, linestyle="--", alpha=0.7)
    ax3.set_xlabel("Itération")
    ax3.set_ylabel("MSE (Mean Squared Error)")
    ax3.set_title("Évolution du coût (MSE) par modèle")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Graphes d'importance des features pour chaque modèle de régression
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors_list = [plt.cm.Blues(np.linspace(0.3, 0.9, len(feature_names))),
                   plt.cm.Greens(np.linspace(0.3, 0.9, len(feature_names))),
                   plt.cm.Oranges(np.linspace(0.3, 0.9, len(feature_names)))]
    
    for idx, (model_type, ax) in enumerate(zip(models, axes)):
        if results[model_type].get('coefficients') is not None:
            weights = np.abs(results[model_type]['coefficients'])
            feature_importance = weights / (weights.sum() + 1e-12)
        else:
            feature_importance = np.zeros(len(feature_names))
        
        ax.barh(feature_names, feature_importance, color=colors_list[idx])
        ax.set_xlabel("Importance (magnitude)")
        ax.set_title(f"Importance des Features - {model_type.capitalize()}")
        ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig("regression_lineaire.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
