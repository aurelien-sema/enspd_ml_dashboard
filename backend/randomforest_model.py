
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from data_preparation import DataPreparation
from arbre_decision import DecisionTree
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


class RandomForest:
    """Random Forest simple avec sous-échantillonnage et sélection de features."""

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features_ratio: float = 0.7,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators  # nombre d'arbres
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_ratio = max_features_ratio  # ratio de features tirées au sort
        self.seed = seed
        self.trees: List[DecisionTree] = []
        self.features_used: List[np.ndarray] = []  # features sélectionnées par arbre

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n_features = X.shape[1]
        self.trees = []
        self.features_used = []

        for _ in range(self.n_estimators):
            # Bootstrap sur les lignes
            indices = rng.integers(0, len(X), len(X))
            X_sample = X[indices]
            y_sample = y[indices]

            # Sous-échantillonnage de features
            k = max(1, int(n_features * self.max_features_ratio))
            feat_idx = rng.choice(n_features, k, replace=False)
            self.features_used.append(feat_idx)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample[:, feat_idx], y_sample)
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Moyenne des prédictions probabilistes (ici 0/1) des arbres
        probs = []
        for tree, feat_idx in zip(self.trees, self.features_used):
            probs.append(tree.predict_proba(X[:, feat_idx]))
        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def feature_importances_(self, n_features: int) -> np.ndarray:
        """
        Importance approximative : fréquence de sélection des features dans les splits racine.
        """
        counts = np.zeros(n_features)
        for feat_idx, tree in zip(self.features_used, self.trees):
            if tree.root and tree.root.feature is not None:
                counts[feat_idx[tree.root.feature]] += 1
        if counts.sum() == 0:
            return counts
        return counts / counts.sum()


class RandomForestModel:
    def __init__(self):
        self.model = RandomForest()
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
        train_ll = np.sum(np.log(np.clip(self.model.predict_proba(X_train), 1e-9, 1)))
        test_ll = np.sum(np.log(np.clip(self.probabilities, 1e-9, 1)))
        train_cost = log_loss_np(y_train, self.model.predict_proba(X_train))
        test_cost = log_loss_np(y_test, self.probabilities)

        cv_mean, cv_std = cross_val_score_simple(
            lambda: RandomForest(
                n_estimators=self.model.n_estimators,
                max_depth=self.model.max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf,
                max_features_ratio=self.model.max_features_ratio,
                seed=self.model.seed,
            ),
            X_train,
            y_train,
            accuracy_score_np,
            k=3,
        )

        feature_importance = self.model.feature_importances_(n_features=X_train.shape[1])
        cm = confusion_matrix_np(y_test, self.predictions)

        print("\n MÉTRIQUES :")
        print(f"   Accuracy Train/Test: {train_acc:.4f} / {test_acc:.4f}")
        print(f"   Precision/Recall/F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        print(f"   Log-vraisemblance Train/Test: {train_ll:.4f} / {test_ll:.4f}")
        print(f"   Log Loss Train/Test: {train_cost:.4f} / {test_cost:.4f}")
        print(f"   CV Accuracy (3 folds): {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"   Matrice de confusion:\n{cm}")

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
            "confusion_matrix": cm,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    model = RandomForestModel()
    results = model.train_and_evaluate(
        data_prep.X_train, data_prep.X_test, data_prep.y_train_class, data_prep.y_test_class
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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
    axes[0].set_title("Courbe ROC (approx)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphe d'importance des features
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    axes[1].barh(feature_names, results["feature_importance"], color=colors)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Importance des Features - Random Forest")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("randomforest_analyse.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
