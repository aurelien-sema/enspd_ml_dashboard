"""
Export pour Linear Regression Model
Exporte: métriques de régression (R², MAE, MSE, RMSE), coefficients, intercept, training history
"""
import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation import DataPreparation
from regression_lineaire import LinearRegressionModel


def save_json(obj, path):
    """Sauvegarde un objet en JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def serialize_for_json(obj):
    """Convertit les types numpy en types JSON-sérialisables."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def main():
    print("=" * 70)
    print("EXPORT - LINEAR REGRESSION MODEL")
    print("=" * 70)
    
    # Chargement des données
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_final.csv")
    dp = DataPreparation(data_file)
    
    if not dp.load_and_prepare_data():
        print("Erreur: impossible de charger les données.")
        return
    
    # Entraînement du modèle
    print("\nTraining Linear Regression...")
    model = LinearRegressionModel(model_type="linear", lr=0.05, n_iter=1500, alpha=0.1)
    results = model.train_and_evaluate(
        dp.X_train, dp.X_test,
        dp.y_train_class, dp.y_test_class
    )
    
    # Création du dossier de sortie
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "linear_regression")
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n[Linear Regression Export]")
    
    # 1. Sauvegarde du modèle
    try:
        model_path = os.path.join(model_dir, "linear_regression.joblib")
        joblib.dump(model.model, model_path)
        print(f"  ✓ Saved model: {model_path}")
    except Exception as e:
        print(f"  ✗ Model save failed: {e}")
    
    # 2. Sauvegarde des métriques
    try:
        metrics_path = os.path.join(model_dir, "metrics.json")
        out_metrics = serialize_for_json(results)
        save_json(out_metrics, metrics_path)
        print(f"  ✓ Saved metrics: {metrics_path}")
        print(f"    - R² Test: {results['test_r2']:.4f}")
        print(f"    - MAE: {results['mae']:.4f}")
        print(f"    - RMSE: {results['rmse']:.4f}")
    except Exception as e:
        print(f"  ✗ Metrics save failed: {e}")
    
    # 3. Sauvegarde des coefficients du modèle linéaire
    try:
        if hasattr(dp, "feature_names"):
            feature_names = list(dp.feature_names)
        elif hasattr(dp.X_train, "columns"):
            feature_names = list(dp.X_train.columns)
        else:
            feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
        
        coefficient_list = results.get("coefficients", np.zeros(len(feature_names)))
        coefficients = np.abs(coefficient_list)
        if isinstance(coefficients, np.ndarray):
            coefficients = coefficients.tolist()
        
        coef_path = os.path.join(model_dir, "features.json")
        save_json({
            "feature_names": feature_names,
            "importance": coefficients,
            "intercept": float(results.get("intercept", 0.0)),
            "description": "Coefficients (poids) de la régression linéaire et intercept (biais)"
        }, coef_path)
        print(f"  ✓ Saved coefficients: {coef_path}")
    except Exception as e:
        print(f"  ✗ Coefficients save failed: {e}")
    
    # 4. Sauvegarde de l'historique d'entraînement
    try:
        history_data = {
            "train_cost_history": results.get("train_cost_history", []),
            "val_cost_history": results.get("val_cost_history", []),
            "description": "Historique de la MSE (Mean Squared Error) lors de l'entraînement"
        }
        history_path = os.path.join(model_dir, "training_history.json")
        save_json(serialize_for_json(history_data), history_path)
        print(f"  ✓ Saved training history: {history_path}")
    except Exception as e:
        print(f"  ✗ Training history save failed: {e}")
    
    print("\n" + "=" * 70)
    print("EXPORT TERMINÉ - Linear Regression")
    print("=" * 70)


if __name__ == "__main__":
    main()
