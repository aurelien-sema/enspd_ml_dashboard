import os
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask import send_file

# chemin absolu du dossier où est app.py
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# chemin où model_export.py a sauvegardé les artefacts
MODELS_DIR = os.path.join(ROOT_DIR, "backend", "models", "models")

# MODELS_DIR = os.path.join(ROOT_DIR, "models")

print("DEBUG: MODELS_DIR =", MODELS_DIR)
STUDENTS_CSV = "backend/models/processed_final.csv" 

app = Flask(__name__)
CORS(app)

# --- Chargement des artefacts ---
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "xgb_model_metrics.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_model_features.json")
ROC_PATH = os.path.join(MODELS_DIR, "xgb_model_roc.npz")
MOYENNE_REG_PATH = os.path.join(MODELS_DIR, "moyenne_regressor.joblib")  

TRAIN_HISTORY_PATH = os.path.join(MODELS_DIR, "xgb_model_training_history.json")
IMPORTANCES_PATH = os.path.join(MODELS_DIR, "xgb_model_importances.json")
BOOSTER_JSON_PATH = os.path.join(MODELS_DIR, "xgb_model_booster.json")

model = None
metrics = None
features_meta = None
roc_data = None
moyenne_regressor = None

if os.path.exists(XGB_MODEL_PATH):
    model = joblib.load(XGB_MODEL_PATH)
else:
    print("Warning: modèle XGB introuvable à", XGB_MODEL_PATH)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        features_meta = json.load(f)

if os.path.exists(ROC_PATH):
    try:
        npz = np.load(ROC_PATH)
        roc_data = {
            "fpr": npz["fpr"].tolist(),
            "tpr": npz["tpr"].tolist(),
            "thresholds": npz["thresholds"].tolist()
        }
    except Exception as e:
        print("Erreur lecture ROC:", e)

if os.path.exists(MOYENNE_REG_PATH):
    try:
        moyenne_regressor = joblib.load(MOYENNE_REG_PATH)
        print("Regressor moyenne chargé.")
    except Exception:
        moyenne_regressor = None

# students dataframe lazy load
def load_students(csv_path=STUDENTS_CSV):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # nettoyage basique: strip spaces in column names
    df.columns = [c.strip() for c in df.columns]
    return df

# util
def to_native(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray, list)):
        return list(map(to_native, list(x)))
    return x

# --- Endpoints ---

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "metrics_loaded": metrics is not None,
        "features_loaded": features_meta is not None
    })

# retourne les features
@app.route("/model/feature_importance", methods=["GET"])
def feature_importance():
    if features_meta is None:
        return jsonify({"error": "Aucun metadata de features trouvé. Exécute model_export.py d'abord."}), 404
    names = features_meta.get("feature_names", [])
    importance = features_meta.get("importance", [])
    # safety: align lengths
    pairs = []
    for i, name in enumerate(names):
        imp = importance[i] if i < len(importance) else None
        pairs.append({"feature": name, "importance": to_native(imp)})
    # tri décroissant
    pairs = sorted(pairs, key=lambda x: (x["importance"] is not None, x["importance"]), reverse=True)
    return jsonify(pairs)

@app.route("/model/metrics", methods=["GET"])
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Aucune métrique trouvée. Exécute model_export.py d'abord."}), 404
    # renvoyer métriques sous forme JSON
    return jsonify(metrics)

@app.route("/model/roc", methods=["GET"])
def get_roc():
    if roc_data is None:
        return jsonify({"error": "Aucune ROC sauvegardée (binaire?). Exécute model_export.py d'abord."}), 404
    return jsonify(roc_data)

@app.route("/model/training_history", methods=["GET"])
def training_history():
    """
    Retourne l'historique d'entraînement (coût/itération) pour tracer le graphe.
    Format renvoyé : { "validation_0": {"logloss": [..], ...}, "validation_1": {...} }
    Front: lire les listes et tracer en fonction de l'index (itération).
    """
    if not os.path.exists(TRAIN_HISTORY_PATH):
        return jsonify({"error": "Training history not found. Run model_export.py with eval_set to generate it."}), 404
    with open(TRAIN_HISTORY_PATH, "r") as f:
        hist = json.load(f)
    series = []
    for dataset_name, metrics_dict in hist.items():
        for metric_name, values in metrics_dict.items():
            series.append({
                "name": f"{dataset_name}_{metric_name}",
                "values": values
            })
    return jsonify({"raw": hist, "series": series})

@app.route("/model/weights", methods=["GET"])
def model_weights():
    """
    Retourne les importances selon différents types : weight, gain, cover
    Format: { "feature_map": ["f0","f1"...], "importance": {"weight": {...}, "gain": {...}, "cover": {...}} }
    """
    if os.path.exists(IMPORTANCES_PATH):
        with open(IMPORTANCES_PATH, "r") as f:
            imps = json.load(f)
        # optional: convert keys 'f0' -> feature names if features_meta exists
        if features_meta and features_meta.get("feature_names"):
            fmap = features_meta["feature_names"]
            # xgboost uses 'f0','f1',... -> map them
            def map_imp(d):
                mapped = {}
                for k, v in d.items():
                    if k.startswith("f"):
                        idx = int(k[1:])
                        fname = fmap[idx] if idx < len(fmap) else k
                        mapped[fname] = v
                    else:
                        mapped[k] = v
                return mapped
            mapped_imps = {t: map_imp(d) for t, d in imps.items()}
            return jsonify({"importances": mapped_imps})
        return jsonify({"importances": imps})
    else:
        # fallback: try getting from loaded model's booster live
        if model is None:
            return jsonify({"error": "No model loaded and no importances JSON found."}), 404
        try:
            booster = model.get_booster()
            imps = {
                "weight": booster.get_score(importance_type='weight'),
                "gain": booster.get_score(importance_type='gain'),
                "cover": booster.get_score(importance_type='cover')
            }
            # map keys if possible same as above
            if features_meta and features_meta.get("feature_names"):
                fmap = features_meta["feature_names"]
                def map_imp(d):
                    mapped = {}
                    for k, v in d.items():
                        if k.startswith("f"):
                            idx = int(k[1:])
                            fname = fmap[idx] if idx < len(fmap) else k
                            mapped[fname] = v
                        else:
                            mapped[k] = v
                    return mapped
                mapped_imps = {t: map_imp(d) for t, d in imps.items()}
                return jsonify({"importances": mapped_imps})
            return jsonify({"importances": imps})
        except Exception as e:
            return jsonify({"error": "Unable to extract importances: " + str(e)}), 500

@app.route("/predict_year", methods=["POST"])
def predict_year():
    """
    JSON body expected:
    { "year": 2027 }
    Returns list of students with predicted decision (+prob) and predicted moyenne (if regressor present).
    Year must be > current year.
    """
    if model is None:
        return jsonify({"error": "Modèle non chargé."}), 500

    body = request.get_json(force=True)
    if not body or "year" not in body:
        return jsonify({"error": "JSON attendu avec clé 'year' (ex: {\"year\":2027})."}), 400

    try:
        year = int(body["year"])
    except Exception:
        return jsonify({"error": "L'année doit être un entier."}), 400

    current_year = datetime.now().year
    if year <= current_year:
        return jsonify({"error": f"L'année doit être STRICTEMENT supérieure à l'année actuelle ({current_year})."}), 400

    df = load_students()
    if df.empty:
        return jsonify({"error": f"Aucun fichier students trouvé à {STUDENTS_CSV}."}), 404

    # Champs utilisateur demandés
    out_cols = []
    # normalize names used in sample: Matricule, First_Name, Last_Name, Department, Level, Moyenne_Final
    # On récupère ce que l'on trouve
    for c in ["Matricule", "First_Name", "Last_Name", "Anonymat_name", "Email", "Department", "Level", "Moyenne_Final"]:
        if c in df.columns:
            out_cols.append(c)

    # Features d'entrée pour le modèle : si features_meta existe, on utilisera ces noms,
    # sinon on tente la liste courante connue.
    if features_meta and features_meta.get("feature_names"):
        model_features = features_meta["feature_names"]
    else:
        # heuristique : colonnes numériques utiles
        candidates = ["Attendance", "Moyenne_Final", "Sleep_hour", "Stress_Level", "Projects_Score", "notecc", "SN"]
        model_features = [c for c in candidates if c in df.columns]
    if not model_features:
        return jsonify({"error": "Impossible de trouver des features numériques pour la prédiction. Ajouter feature_names dans models/... ou nommer colonnes."}), 400

    # Construire X pour prediction (attention aux NaN)
    X = df[model_features].copy()
    # Remplacer NaN par 0 ou mediane (simple)
    X = X.fillna(X.median(numeric_only=True))
    try:
        X_values = X.to_numpy()
    except Exception as e:
        return jsonify({"error": "Erreur lors de la préparation des features: " + str(e)}), 500

    # Predictions
    try:
        preds = model.predict(X_values)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_values)
            # prendre prob classe 1 si binaire (sinon renvoyer vector)
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            else:
                probs = probs.tolist()
    except Exception as e:
        return jsonify({"error": "Erreur prédiction: " + str(e)}), 500

    # Pred Moyenne si regressor exists
    predicted_moyennes = None
    if moyenne_regressor is not None:
        try:
            predicted_moyennes = moyenne_regressor.predict(X_values)
        except Exception as e:
            predicted_moyennes = None

    # Construire la réponse
    results = []
    for i, row in df.iterrows():
        username = None
        if "First_Name" in df.columns and "Last_Name" in df.columns:
            username = f"{row.get('First_Name')} {row.get('Last_Name')}"
        elif "Anonymat_name" in df.columns:
            username = row.get("Anonymat_name")
        elif "Email" in df.columns:
            username = row.get("Email")
        else:
            username = None

        matricule = row.get("Matricule") if "Matricule" in row.index else None
        department = row.get("Department") if "Department" in row.index else None
        level = row.get("Level") if "Level" in row.index else None
        original_moy = row.get("Moyenne_Final") if "Moyenne_Final" in row.index else None

        pred_label = int(preds[i]) if hasattr(preds[i], "__int__") else preds[i]
        prob = None
        if probs is not None:
            prob = float(probs[i]) if not isinstance(probs[i], list) else probs[i]

        pred_moy = None
        if predicted_moyennes is not None:
            pred_moy = float(predicted_moyennes[i])
        else:
            # fallback: utiliser la moyenne actuelle comme estimation
            pred_moy = float(original_moy) if original_moy is not None and not pd.isna(original_moy) else None

        results.append({
            "username": username,
            "matricule": matricule,
            "department": department,
            "level": level,
            "prediction_decision": ("Admis" if pred_label == 1 else "Echec"),
            "prediction_probability_positive": prob,
            "prediction_moyenne": pred_moy,
            "base_moyenne": original_moy
        })

    return jsonify({
        "year_predicted": year,
        "n_students": len(results),
        "model_used": os.path.basename(XGB_MODEL_PATH),
        "features_used": model_features,
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
