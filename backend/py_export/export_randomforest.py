"""
Export pour Random Forest Model
Exporte: métriques de classification, feature importance, confusion matrix, ROC curves
"""
import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation import DataPreparation
from randomforest_model import RandomForestModel


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


def compute_roc_curves(y_test, probabilities):
    """Calcule les courbes ROC."""
    try:
        y_test = np.asarray(y_test).astype(int)
        probs = np.asarray(probabilities, dtype=float)
        
        if not (probs.size and y_test.size == probs.size):
            return None
        
        desc_idx = np.argsort(probs, kind="mergesort")[::-1]
        y_sorted = y_test[desc_idx]
        scores_sorted = probs[desc_idx]

        distinct_mask = np.r_[True, scores_sorted[1:] != scores_sorted[:-1]]
        threshold_idxs = np.where(distinct_mask)[0]
        tps = np.cumsum(y_sorted == 1)[threshold_idxs]
        fps = np.cumsum(y_sorted == 0)[threshold_idxs]

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[np.inf, scores_sorted[threshold_idxs]]

        pos_total = y_test.sum()
        neg_total = y_test.size - pos_total
        
        if pos_total > 0 and neg_total > 0:
            tpr = tps / pos_total
            fpr = fps / neg_total
            return {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
    except Exception as e:
        print(f"  ROC computation error: {e}")
    return None


def main():
    print("=" * 70)
    print("EXPORT - RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Chargement des données
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_final.csv")
    dp = DataPreparation(data_file)
    
    if not dp.load_and_prepare_data():
        print("Erreur: impossible de charger les données.")
        return
    
    # Entraînement du modèle
    print("\nTraining Random Forest...")
    model = RandomForestModel()
    results = model.train_and_evaluate(
        dp.X_train, dp.X_test,
        dp.y_train_class, dp.y_test_class
    )
    
    # Création du dossier de sortie
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "random_forest")
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n[Random Forest Export]")
    
    # 1. Sauvegarde du modèle
    try:
        model_path = os.path.join(model_dir, "random_forest.joblib")
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
        print(f"    - Accuracy: {results['test_accuracy']:.4f}")
        print(f"    - AUC-ROC: {results['auc_roc']:.4f}")
    except Exception as e:
        print(f"  ✗ Metrics save failed: {e}")
    
    # 3. Sauvegarde des features et leur importance
    try:
        if hasattr(dp, "feature_names"):
            feature_names = list(dp.feature_names)
        elif hasattr(dp.X_train, "columns"):
            feature_names = list(dp.X_train.columns)
        else:
            feature_names = ["Attendance", "Projects_Score", "notecc", "SN", "Sleep_hour", "Stress_Level (1-10)"]
        
        importance = results.get("feature_importance", np.zeros(len(feature_names)))
        if isinstance(importance, np.ndarray):
            importance = importance.tolist()
        
        feat_path = os.path.join(model_dir, "features.json")
        save_json({
            "feature_names": feature_names,
            "importance": importance,
            "description": "Importance de chaque feature pour le Random Forest"
        }, feat_path)
        print(f"  ✓ Saved features importance: {feat_path}")
    except Exception as e:
        print(f"  ✗ Features save failed: {e}")
    
    # 4. Sauvegarde des courbes ROC
    try:
        roc_data = compute_roc_curves(dp.y_test_class, model.probabilities)
        if roc_data:
            roc_path = os.path.join(model_dir, "roc_curves.json")
            save_json(roc_data, roc_path)
            print(f"  ✓ Saved ROC curves: {roc_path}")
    except Exception as e:
        print(f"  ✗ ROC save failed: {e}")
    
    # 5. Sauvegarde de la matrice de confusion
    try:
        cm_data = {
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "description": "Matrice de confusion du modèle"
        }
        cm_path = os.path.join(model_dir, "confusion_matrix.json")
        save_json(cm_data, cm_path)
        print(f"  ✓ Saved confusion matrix: {cm_path}")
    except Exception as e:
        print(f"  ✗ Confusion matrix save failed: {e}")
    
    print("\n" + "=" * 70)
    print("EXPORT TERMINÉ - Random Forest")
    print("=" * 70)


if __name__ == "__main__":
    main()
