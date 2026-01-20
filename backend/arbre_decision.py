
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class TreeNode:
    feature: Optional[int] = None  # index de feature utilisé pour le split
    threshold: Optional[float] = None  # seuil numérique pour le split
    left: Optional["TreeNode"] = None  # sous-arbre gauche
    right: Optional["TreeNode"] = None  # sous-arbre droit
    prediction: Optional[int] = None  # valeur prédite si feuille


def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1 - p1
    return 1 - p0**2 - p1**2


class DecisionTree:
    """Arbre de décision très simplifié pour classification binaire."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10, min_samples_leaf: int = 5):
        self.max_depth = max_depth  # profondeur maximale autorisée
        self.min_samples_split = min_samples_split  # minimum d'échantillons pour splitter
        self.min_samples_leaf = min_samples_leaf  # minimum d'échantillons par feuille
        self.root: Optional[TreeNode] = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_feature, best_thresh, best_gain = None, None, -np.inf
        n_samples, n_features = X.shape
        current_impurity = gini_impurity(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                impurity_left = gini_impurity(y[left_mask])
                impurity_right = gini_impurity(y[right_mask])
                n_left, n_right = left_mask.sum(), right_mask.sum()
                weighted_impurity = (n_left * impurity_left + n_right * impurity_right) / n_samples
                gain = current_impurity - weighted_impurity

                if gain > best_gain:
                    best_feature, best_thresh, best_gain = feat, thr, gain

        return best_feature, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode()
        majority_class = int(np.round(np.mean(y)))  # prédiction majoritaire

        # Conditions d'arrêt
        if (
            depth >= self.max_depth
            or len(X) < self.min_samples_split
            or gini_impurity(y) == 0
        ):
            node.prediction = majority_class
            return node

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            node.prediction = majority_class
            return node

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        node.feature = feat
        node.threshold = thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build(X, y, depth=0)

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while node and node.prediction is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction if node else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row) for row in X], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return preds.astype(float)  # probas approximées (0 ou 1)

    def get_depth(self) -> int:
        def _depth(node: TreeNode) -> int:
            if node is None or node.prediction is not None:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root)

    def get_n_leaves(self) -> int:
        def _count(node: TreeNode) -> int:
            if node is None:
                return 0
            if node.prediction is not None:
                return 1
            return _count(node.left) + _count(node.right)

        return _count(self.root)

    def visualize_tree(self, feature_names=None):
        """Visualise l'arbre de décision."""
        if self.root is None:
            print("L'arbre n'a pas été entraîné.")
            return

        depth = self.get_depth()
        # Limite la taille pour éviter des images trop grandes
        figsize_width = min(50 + depth * 3, 200)
        figsize_height = min(20 + depth * 2, 100)
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=100)

        def plot_tree(node, x=0, y=1, dx=1, ax=None, depth=0):
            if node is None:
                return

            if node.prediction is not None:
                # Feuille
                color = "lightgreen" if node.prediction == 1 else "lightcoral"
                ax.scatter(x, y, s=3000, c=color, edgecolors="black", linewidth=1.5, zorder=3)
                ax.text(x, y, f"Class: {node.prediction}", ha="center", va="center", fontsize=9, weight="bold")
            else:
                # Nœud interne
                feat_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
                label = f"{feat_name} ≤ {node.threshold:.2f}"
                ax.scatter(x, y, s=4000, c="lightblue", edgecolors="black", linewidth=1.5, zorder=3)
                ax.text(x, y, label, ha="center", va="center", fontsize=8, weight="bold")

                # Enfant gauche - écartement linéaire
                if node.left:
                    x_left = x - 40
                    y_left = y - 2.5
                    ax.plot([x, x_left], [y, y_left], "k-", lw=1.5, zorder=1)
                    plot_tree(node.left, x_left, y_left, 40, ax, depth + 1)

                # Enfant droit - écartement linéaire
                if node.right:
                    x_right = x + 40
                    y_right = y - 2.5
                    ax.plot([x, x_right], [y, y_right], "k-", lw=1.5, zorder=1)
                    plot_tree(node.right, x_right, y_right, 40, ax, depth + 1)

        plot_tree(self.root, ax=ax)
        # Augmente les limites en fonction de la profondeur (linéaire)
        max_width = 200 + depth * 80
        ax.set_xlim(-max_width, max_width)
        ax.set_ylim(-depth * 3 - 3, 2)
        ax.axis("off")
        ax.set_title("Visualisation de l'Arbre de Décision", fontsize=14, weight="bold")
        return fig, ax


class DecisionTreeModel:
    """Enveloppe d'entraînement/évaluation avec métriques maison."""

    def __init__(self):
        self.model = DecisionTree(max_depth=5, min_samples_split=10, min_samples_leaf=5)
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
            lambda: DecisionTree(
                max_depth=self.model.max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf,
            ),
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
        print(f"   Profondeur / feuilles: {self.model.get_depth()} / {self.model.get_n_leaves()}")
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
            "tree_depth": self.model.get_depth(),
            "n_leaves": self.model.get_n_leaves(),
            "confusion_matrix": cm,
        }


def main():
    data_prep = DataPreparation("processed_final.csv")
    if not data_prep.load_and_prepare_data():
        return

    model = DecisionTreeModel()
    results = model.train_and_evaluate(
        data_prep.X_train, data_prep.X_test, data_prep.y_train_class, data_prep.y_test_class
    )

    # Visualisation de l'arbre de décision
    feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
    fig_tree, ax_tree = model.model.visualize_tree(feature_names=feature_names)
    plt.savefig("arbre_decision_tree.png", dpi=200, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Courbe ROC basique
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

    # Texte récapitulatif
    text = (
        f"Accuracy test: {results['test_accuracy']:.3f}\n"
        f"AUC: {results['auc_roc']:.3f}\n"
        f"Log loss test: {results['test_cost']:.3f}\n"
        f"Profondeur: {results['tree_depth']}\n"
        f"Feuilles: {results['n_leaves']}"
    )
    axes[1].axis("off")
    axes[1].text(0.1, 0.5, text, fontsize=11, va="center")

    plt.tight_layout()
    plt.savefig("arbre_decision_analyse.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
