
from __future__ import annotations

import numpy as np
import pandas as pd
from metrics import (
    standardize_fit_transform,
    standardize_transform,
    train_test_split_simple,
    train_test_split_stratified,
)


class DataPreparation:
    """Charge le CSV, encode la cible et produit des matrices numpy prêtes à l'emploi."""

    def __init__(self, data_path: str):
        self.data_path = data_path  # chemin vers le CSV d'entrée
        self.df = None  # DataFrame initial
        self.X_train = None  # matrice des features pour l'entraînement
        self.X_test = None  # matrice des features pour le test
        self.y_train_class = None  # cible binaire (classification) train
        self.y_test_class = None  # cible binaire (classification) test
        self.y_train_reg = None  # cible continue (régression) train
        self.y_test_reg = None  # cible continue (régression) test
        self._mean = None  # moyennes pour la standardisation
        self._std = None  # écarts-types pour la standardisation

    def load_and_prepare_data(self) -> bool:
        """Lit le CSV et prépare les jeux train/test normalisés."""
        self.df = pd.read_csv(self.data_path)
        self.df.columns = self.df.columns.str.strip()

        # Encodage binaire de la décision : Admis -> 1, sinon 0
        self.df["decision_numeric"] = self.df["decision"].apply(
            lambda x: 1 if str(x).strip() == "Admis" else 0
        )

        features = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
        missing = [f for f in features if f not in self.df.columns]
        if missing:
            print(f"Features manquantes: {missing}")
            return False

        X = self.df[features].to_numpy(dtype=float)
        y_class = self.df["decision_numeric"].to_numpy(dtype=int)
        y_reg = self.df["Moyenne_Final"].to_numpy(dtype=float)

        # Split stratifié pour classification, split simple pour régression
        X_train_c, X_test_c, y_train_class, y_test_class = train_test_split_stratified(
            X, y_class, test_size=0.2, seed=42
        )
        X_train_r, X_test_r, y_train_reg, y_test_reg = train_test_split_simple(
            X, y_reg, test_size=0.2, seed=42
        )

        # Standardisation apprise sur le train classification (mêmes stats pour tous)
        X_train_scaled, mean, std = standardize_fit_transform(X_train_c)
        X_test_scaled = standardize_transform(X_test_c, mean, std)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train_class = y_train_class
        self.y_test_class = y_test_class
        self.y_train_reg = y_train_reg
        self.y_test_reg = y_test_reg
        self._mean = mean
        self._std = std

        print(
            f"Données prêtes. Train={len(self.X_train)} | Test={len(self.X_test)} | "
            f"Proportion Admis={np.mean(y_class):.2%}"
        )
        return True
