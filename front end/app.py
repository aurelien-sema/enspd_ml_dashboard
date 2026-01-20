"""
app.py - Application ORAGE ENSPD
"""
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
import requests
import pandas as pd
import numpy as np
import joblib
import sys

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "orage_ensp_2025_dev_secret")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration backend
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5001")
# Chemins relatifs depuis le dossier frontend
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_MODELS_DIR = os.path.join(ROOT_DIR, "backend", "models", "random_forest")
BACKEND_DATA_DIR = os.path.join(ROOT_DIR, "backend")

# Ajouter le dossier backend au chemin Python pour pouvoir importer les modèles
sys.path.insert(0, BACKEND_DATA_DIR)

# ======================== FONCTION UTILITAIRE ========================
def load_model_safely(model_path):
    """Charge le modèle joblib de manière sécurisée avec gestion d'erreurs"""
    try:
        # Essayer de charger le modèle
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {model_path}: {e}")
        # Si joblib échoue, essayer une approche alternative
        try:
            # Importer la classe RandomForest du backend
            from randomforest_model import RandomForest
            model = joblib.load(model_path)
            return model
        except Exception as e2:
            print(f"Erreur alternative: {e2}")
            raise Exception(f"Impossible de charger le modèle: {str(e)}")

# ======================== AUTHENTIFICATION ========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Base utilisateurs temporaire
users_db = {
    "admin@enspd.cm": {
        "password": generate_password_hash("orage2025"),
        "name": "Admin ENSPD",
        "role": "admin"
    }
}


class User(UserMixin):
    def __init__(self, email):
        self.id = email
        self.email = email
        user_data = users_db.get(email, {})
        self.name = user_data.get("name", "Utilisateur")
        self.role = user_data.get("role", "user")


@login_manager.user_loader
def load_user(email):
    if email in users_db:
        return User(email)
    return None


# ======================== FILTRES JINJA2 ========================
@app.template_filter('format_number')
def format_number_filter(value):
    """Format 1248 → 1 248"""
    try:
        num = int(float(value))
        return f"{num:,}".replace(",", " ")
    except:
        return str(value)


@app.template_filter('format_percentage')
def format_percentage_filter(value):
    """Format 78.4 → 78.4%"""
    try:
        return f"{float(value):.1f}%"
    except:
        return "0%"


@app.template_filter('format_date')
def format_date_filter(value):
    """Format date en français"""
    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        return date_obj.strftime("%d/%m/%Y")
    except:
        return value


# ======================== ROUTES ========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')

        user_data = users_db.get(email)
        if user_data and check_password_hash(user_data['password'], password):
            user = User(email)
            login_user(user)
            flash(f"Bienvenue {user.name} !", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Email ou mot de passe incorrect", "danger")

    return render_template('auth/login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        if email in users_db:
            flash("Cet email existe déjà", "warning")
        elif password != confirm:
            flash("Les mots de passe ne correspondent pas", "warning")
        else:
            users_db[email] = {
                "password": generate_password_hash(password),
                "name": name,
                "role": "user"
            }
            flash("Compte créé ! Connectez-vous.", "success")
            return redirect(url_for('login'))

    return render_template('auth/signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Déconnecté", "info")
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard avec données réelles depuis student_stored_data.csv"""
    try:
        # Charger les données depuis student_stored_data.csv
        csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            total_etudiants = len(df)
            
            # Calculer le taux de réussite : compter les "Admis" et diviser par le total
            if 'decision' in df.columns:
                admis_count = (df['decision'].str.strip() == 'Admis').sum()
                taux_reussite = (admis_count / total_etudiants * 100) if total_etudiants > 0 else 0
            else:
                taux_reussite = 0
            
            # Charger le modèle Random Forest pour identifier les étudiants à risque
            model_path = os.path.join(BACKEND_MODELS_DIR, "random_forest.joblib")
            features_path = os.path.join(BACKEND_MODELS_DIR, "features.json")
            
            etudiants_risque = []
            if os.path.exists(model_path) and os.path.exists(features_path):
                try:
                    model = load_model_safely(model_path)
                    with open(features_path, 'r') as f:
                        features_meta = json.load(f)
                    
                    feature_names = features_meta.get('feature_names', [])
                    # Normaliser les noms de colonnes (gérer les espaces)
                    df_columns_normalized = {col.strip(): col for col in df.columns}
                    
                    # Vérifier que toutes les features existent
                    available_features = []
                    for feat in feature_names:
                        # Chercher la colonne avec ou sans espace
                        if feat in df.columns:
                            available_features.append(feat)
                        elif feat.strip() in df_columns_normalized:
                            available_features.append(df_columns_normalized[feat.strip()])
                        else:
                            # Essayer avec un espace à la fin (comme "Attendance ")
                            feat_with_space = feat + " "
                            if feat_with_space in df.columns:
                                available_features.append(feat_with_space)
                    
                    if len(available_features) == len(feature_names):
                        X = df[available_features].fillna(0).values
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        # Identifier les étudiants à risque : ceux qui ont échoué (prédiction = 0 ou Echec)
                        risk_indices = np.where(predictions == 0)[0]
                        
                        # Trier par probabilité d'échec (plus risqué en premier) et prendre les 15 premiers
                        if probabilities is not None:
                            risk_probs = [(idx, 1 - probabilities[idx]) for idx in risk_indices]
                            risk_probs.sort(key=lambda x: x[1], reverse=True)
                            risk_indices = [idx for idx, _ in risk_probs[:15]]
                        else:
                            risk_indices = risk_indices[:15]
                        
                        for idx in risk_indices:
                            row = df.iloc[idx]
                            prob_echec = float(1 - probabilities[idx]) * 100 if probabilities is not None else 100
                            etudiants_risque.append({
                                'matricule': str(row.get('Matricule', f'STU{idx}')),
                                'nom': f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip() or row.get('Anonymat_name', 'Inconnu'),
                                'filiere': str(row.get('Department', 'N/A')),
                                'niveau': str(row.get('Level', 'N/A')),
                                'risque': int(prob_echec),
                                'facteurs': ['Moyenne', 'Assiduité']
                            })
                except Exception as e:
                    print(f"Erreur chargement modèle: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Facteur clé depuis l'importance des features
            facteur_cle = 'Assiduité'
            if os.path.exists(features_path):
                try:
                    with open(features_path, 'r') as f:
                        features_meta = json.load(f)
                    importances = features_meta.get('importance', [])
                    feature_names = features_meta.get('feature_names', [])
                    if importances and feature_names:
                        max_idx = np.argmax(importances)
                        facteur_cle = feature_names[max_idx]
                except:
                    pass
            
            stats = {
                'total_etudiants': total_etudiants,
                'taux_reussite': round(taux_reussite, 1),
                'etudiants_risque': len(etudiants_risque),
                'facteur_cle': facteur_cle
            }
        else:
            # Données par défaut si fichier non trouvé
            stats = {
                'total_etudiants': 0,
                'taux_reussite': 0,
                'etudiants_risque': 0,
                'facteur_cle': 'Assiduité'
            }
            etudiants_risque = []
    except Exception as e:
        print(f"Erreur dashboard: {e}")
        import traceback
        traceback.print_exc()
        stats = {
            'total_etudiants': 0,
            'taux_reussite': 0,
            'etudiants_risque': 0,
            'facteur_cle': 'Assiduité'
        }
        etudiants_risque = []

    return render_template('dashboard/dashboard.html',
                           stats=stats,
                           etudiants_risque=etudiants_risque)


# ======================== CONTEXTE ========================
@app.context_processor
def utility_processor():
    """Variables disponibles dans tous les templates"""
    return dict(
        now=datetime.now
    )


# ============ AJOUTE CES ROUTES ============

@app.route('/predict')
@login_required
def predict():
    """Page de prédiction"""
    return render_template('dashboard/prediction.html')

@app.route('/performances')
@login_required
def performances():
    """Page des performances avec métriques de tous les modèles"""
    all_metrics = {}
    roc_data = None
    random_forest_features = None
    
    try:
        # Parcourir le dossier models et lire tous les metrics.json
        models_dir = os.path.join(ROOT_DIR, "backend", "models")
        if os.path.exists(models_dir):
            for model_folder in os.listdir(models_dir):
                model_folder_path = os.path.join(models_dir, model_folder)
                if os.path.isdir(model_folder_path):
                    metrics_path = os.path.join(model_folder_path, "metrics.json")
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                model_metrics = json.load(f)
                                all_metrics[model_folder] = model_metrics
                        except Exception as e:
                            print(f"Erreur lecture {metrics_path}: {e}")
        
        # Charger les données ROC de Random Forest
        roc_path = os.path.join(BACKEND_MODELS_DIR, "roc_curves.json")
        if os.path.exists(roc_path):
            with open(roc_path, 'r') as f:
                roc_data = json.load(f)
        
        # Charger l'importance des features Random Forest
        features_path = os.path.join(BACKEND_MODELS_DIR, "features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                random_forest_features = json.load(f)
    except Exception as e:
        print(f"Erreur chargement performances: {e}")
        import traceback
        traceback.print_exc()
    
    return render_template('dashboard/performances.html',
                           all_metrics=all_metrics,
                           roc_data=roc_data,
                           random_forest_features=random_forest_features)

@app.route('/recommendations')
@login_required
def recommendations():
    """Page des recommandations"""
    return render_template('dashboard/recommendations.html')

@app.route('/data')
@login_required
def data():
    """Page des données gérées par student_stored_data.csv"""
    stats = {
        'total_etudiants': 0,
        'total_notes': 0,
        'qualite_donnees': 0
    }
    
    try:
        csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            stats['total_etudiants'] = len(df)
            
            # Compter les notes (colonnes numériques)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats['total_notes'] = len(numeric_cols) * len(df)
            
            # Qualité des données (pourcentage de valeurs non nulles)
            if len(df) > 0:
                stats['qualite_donnees'] = round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
    except Exception as e:
        print(f"Erreur chargement données: {e}")
    
    return render_template('dashboard/data.html', stats=stats)

@app.route('/account')
@login_required
def account():
    """Page mon compte"""
    return render_template('dashboard/account.html')

@app.route('/help')
@login_required
def help():
    """Page d'aide"""
    return render_template('dashboard/help.html')


# ======================== API ROUTES ========================

@app.route('/api/predict/individual', methods=['POST'])
@login_required
def api_predict_individual():
    """API pour prédiction individuelle avec Random Forest et enregistrement automatique"""
    try:
        data = request.get_json()
        
        # Mapper les données du formulaire aux features du modèle
        model_path = os.path.join(BACKEND_MODELS_DIR, "random_forest.joblib")
        features_path = os.path.join(BACKEND_MODELS_DIR, "features.json")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            return jsonify({"error": "Modèle non disponible"}), 500
        
        with open(features_path, 'r') as f:
            features_meta = json.load(f)
        
        feature_names = features_meta.get('feature_names', [])
        model = load_model_safely(model_path)
        
        # Préparer les features selon les nouvelles valeurs du formulaire
        # SN: /50, notecc: /30, Attendance: 0-10
        sn_value = float(data.get('moyenne_generale', 25))  # Valeur SN (0-50)
        notecc_value = float(data.get('notes_semestre', 15))  # Valeur CC (0-30)
        attendance_value = float(data.get('assiduite', 8))  # Attendance 0-10
        
        # Mapping des features
        mapping = {
            'Attendance': float(attendance_value),  # 0-10 directement
            'Projects_Score': {'faible': 2.5, 'moyen': 5.0, 'eleve': 7.5, 'excellent': 10.0}.get(data.get('pap', 'moyen'), 5.0),
            'notecc': float(notecc_value),  # Valeur CC directement (0-30)
            'SN': float(sn_value),  # Valeur SN directement (0-50)
            'Sleep_hour': float(data.get('sleep_hour', 7.0)),  # Heures de sommeil
            'Stress_Level (1-10)': float(data.get('stress_level', 5.0))  # Niveau de stress 1-10
        }
        
        feature_values = []
        for feat in feature_names:
            if feat in mapping:
                feature_values.append(mapping[feat])
            else:
                feature_values.append(0.0)
        
        X = np.array([feature_values])
        
        # Prédiction avec Random Forest
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else (1.0 if prediction == 1 else 0.0)
        
        # Déterminer si admissible ou recalable
        decision = 'Admis   ' if prediction == 1 else 'Echec   '  # Format avec espaces comme dans le CSV
        admissible = prediction == 1
        
        # Enregistrer la prédiction dans student_stored_data.csv
        try:
            student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
            
            # Générer un matricule unique
            if os.path.exists(student_csv_path):
                existing_df = pd.read_csv(student_csv_path)
                last_matricule = int(existing_df['Matricule'].str.replace('S', '').astype(int).max())
                new_matricule = f"S{last_matricule + 1}"
            else:
                new_matricule = "S5000"
            
            # Créer une ligne pour le nouvel étudiant
            new_student = {
                'Matricule': new_matricule,
                'First_Name': data.get('first_name', 'Prédiction'),
                'Last_Name': data.get('last_name', 'Automatique'),
                'Email': data.get('email', f'{new_matricule}@prediction.local'),
                'Anonymat_name': data.get('anonymat_name', f'Stud {new_matricule}'),
                'Password': '1234',
                'Gender': data.get('gender', 'Unknown'),
                'Age': int(data.get('age', 20)) if data.get('age') else 20,
                'Department': data.get('department', 'GIT'),
                'Level': data.get('level', "High School"),
                'Attendance ': float(attendance_value),
                'Projects_Score': {'faible': 2.5, 'moyen': 5.0, 'eleve': 7.5, 'excellent': 10.0}.get(data.get('pap', 'moyen'), 5.0),
                'notecc': float(notecc_value),
                'SN': float(sn_value),
                'Moyenne_Final': (float(sn_value) * 0.6 + float(notecc_value) * 0.4) / 3.0,  # Estimation
                'Sleep_hour': float(data.get('sleep_hour', 7.0)),
                'Stress_Level (1-10)': float(data.get('stress_level', 5.0)),
                'decision': decision,
                'Annee': datetime.now().year
            }
            
            # Ajouter au CSV
            if os.path.exists(student_csv_path):
                existing_df = pd.read_csv(student_csv_path)
                new_row_df = pd.DataFrame([new_student])
                combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                combined_df.to_csv(student_csv_path, index=False)
            else:
                new_df = pd.DataFrame([new_student])
                new_df.to_csv(student_csv_path, index=False)
        except Exception as e:
            print(f"Erreur lors de l'enregistrement: {e}")
            import traceback
            traceback.print_exc()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'decision': decision.strip(),
            'admissible': admissible,
            'recalable': not admissible,
            'probability_percent': round(probability * 100, 1),
            'matricule': new_matricule if os.path.exists(os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")) else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
@login_required
def api_predict_batch():
    """API pour prédiction multiple via CSV - doit suivre l'ordre de student_stored_data.csv"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Fichier vide"}), 400
        
        # Sauvegarder temporairement
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Charger le CSV
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        # Charger le modèle Random Forest
        model_path = os.path.join(BACKEND_MODELS_DIR, "random_forest.joblib")
        features_path = os.path.join(BACKEND_MODELS_DIR, "features.json")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            os.remove(filepath)
            return jsonify({"error": "Modèle non disponible"}), 500
        
        with open(features_path, 'r') as f:
            features_meta = json.load(f)
        
        feature_names = features_meta.get('feature_names', [])
        model = load_model_safely(model_path)
        
        # Normaliser les noms de colonnes (gérer les espaces comme "Attendance ")
        df_columns_normalized = {col.strip(): col for col in df.columns}
        available_features = []
        for feat in feature_names:
            if feat in df.columns:
                available_features.append(feat)
            elif feat.strip() in df_columns_normalized:
                available_features.append(df_columns_normalized[feat.strip()])
            else:
                feat_with_space = feat + " "
                if feat_with_space in df.columns:
                    available_features.append(feat_with_space)
        
        if len(available_features) != len(feature_names):
            missing = [f for f in feature_names if f not in available_features]
            os.remove(filepath)
            return jsonify({"error": f"Colonnes manquantes: {', '.join(missing)}. Le CSV doit suivre l'ordre de student_stored_data.csv"}), 400
        
        # Préparer les données
        X = df[available_features].fillna(0).values
        
        # Prédictions avec Random Forest
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else predictions.astype(float)
        
        # Construire les résultats
        results = []
        for idx, row in df.iterrows():
            decision = 'Admis' if predictions[idx] == 1 else 'Echec'
            results.append({
                'matricule': str(row.get('Matricule', f'STU{idx}')),
                'nom': f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip() or row.get('Anonymat_name', 'Inconnu'),
                'filiere': str(row.get('Department', 'N/A')),
                'moyenne': float(row.get('Moyenne_Final', 0)) if 'Moyenne_Final' in row else None,
                'assiduite': float(row.get('Attendance', 0)) if 'Attendance' in row else None,
                'reussite': decision,
                'admissible': predictions[idx] == 1,
                'recalable': predictions[idx] == 0,
                'probabilite': round(float(probabilities[idx]) * 100, 1)
            })
        
        # Ajouter les nouveaux étudiants à student_stored_data.csv
        try:
            student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
            if os.path.exists(student_csv_path):
                # Lire le fichier existant
                existing_df = pd.read_csv(student_csv_path)
                existing_df.columns = existing_df.columns.str.strip()
                
                # Ajouter les colonnes decision et Annee si elles n'existent pas dans le nouveau CSV
                if 'decision' not in df.columns:
                    df['decision'] = [('Admis' if p == 1 else 'Echec') for p in predictions]
                if 'Annee' not in df.columns:
                    from datetime import datetime
                    df['Annee'] = datetime.now().year
                
                # S'assurer que toutes les colonnes correspondent
                for col in existing_df.columns:
                    if col not in df.columns:
                        df[col] = None
                
                # Réorganiser les colonnes pour correspondre
                df = df[existing_df.columns]
                
                # Ajouter à la suite du fichier existant
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(student_csv_path, index=False)
            else:
                # Créer le fichier si il n'existe pas
                if 'decision' not in df.columns:
                    df['decision'] = [('Admis' if p == 1 else 'Echec') for p in predictions]
                if 'Annee' not in df.columns:
                    from datetime import datetime
                    df['Annee'] = datetime.now().year
                df.to_csv(student_csv_path, index=False)
        except Exception as e:
            print(f"Erreur lors de l'ajout à student_stored_data.csv: {e}")
        
        # Nettoyer
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'n_students': len(results),
            'results': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
@login_required
def api_metrics():
    """API pour récupérer les métriques Random Forest"""
    try:
        metrics_path = os.path.join(BACKEND_MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        return jsonify({"error": "Métriques non disponibles"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/roc', methods=['GET'])
@login_required
def api_roc():
    """API pour récupérer les données ROC"""
    try:
        roc_path = os.path.join(BACKEND_MODELS_DIR, "roc_curves.json")
        if os.path.exists(roc_path):
            with open(roc_path, 'r') as f:
                roc_data = json.load(f)
            return jsonify(roc_data)
        return jsonify({"error": "Données ROC non disponibles"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/import', methods=['POST'])
@login_required
def api_data_import():
    """API pour importer des données - ajoute à student_stored_data.csv"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Aucun fichier fourni"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Fichier vide"}), 400
        
        # Sauvegarder temporairement
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Charger le nouveau CSV
        new_df = pd.read_csv(filepath)
        new_df.columns = new_df.columns.str.strip()
        
        # Charger student_stored_data.csv existant
        student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        
        if os.path.exists(student_csv_path):
            existing_df = pd.read_csv(student_csv_path)
            existing_df.columns = existing_df.columns.str.strip()
            
            # S'assurer que toutes les colonnes correspondent
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = None
            
            # Réorganiser les colonnes
            new_df = new_df[existing_df.columns]
            
            # Ajouter à la suite
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(student_csv_path, index=False)
        else:
            # Créer le fichier si il n'existe pas
            new_df.to_csv(student_csv_path, index=False)
        
        # Nettoyer
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Fichier {filename} importé avec succès et ajouté à student_stored_data.csv',
            'n_new_records': len(new_df)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/csv', methods=['GET'])
@login_required
def api_export_csv():
    """Exporte toutes les métriques en CSV"""
    try:
        from io import StringIO
        
        # Récupérer tous les modèles et leurs métriques
        models_dir = os.path.join(ROOT_DIR, "backend", "models")
        all_metrics = {}
        
        if os.path.exists(models_dir):
            for model_folder in os.listdir(models_dir):
                model_folder_path = os.path.join(models_dir, model_folder)
                if os.path.isdir(model_folder_path):
                    metrics_path = os.path.join(model_folder_path, "metrics.json")
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                model_metrics = json.load(f)
                                all_metrics[model_folder] = model_metrics
                        except Exception as e:
                            print(f"Erreur lecture {metrics_path}: {e}")
        
        # Créer un CSV avec les métriques principales
        csv_data = StringIO()
        csv_data.write("Modèle,Précision,Rappel,F1-Score,AUC-ROC,Accuracy Test\n")
        
        for model_name, metrics in all_metrics.items():
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1_score = metrics.get('f1_score', 'N/A')
            auc_roc = metrics.get('auc_roc', 'N/A')
            test_accuracy = metrics.get('test_accuracy', 'N/A')
            
            csv_data.write(f"{model_name},{precision},{recall},{f1_score},{auc_roc},{test_accuracy}\n")
        
        # Retourner comme fichier téléchargeable
        response = make_response(csv_data.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=metriques_modeles.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/json', methods=['GET'])
@login_required
def api_export_json():
    """Exporte toutes les métriques en JSON"""
    try:
        models_dir = os.path.join(ROOT_DIR, "backend", "models")
        all_metrics = {}
        
        if os.path.exists(models_dir):
            for model_folder in os.listdir(models_dir):
                model_folder_path = os.path.join(models_dir, model_folder)
                if os.path.isdir(model_folder_path):
                    metrics_path = os.path.join(model_folder_path, "metrics.json")
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                model_metrics = json.load(f)
                                all_metrics[model_folder] = model_metrics
                        except Exception as e:
                            print(f"Erreur lecture {metrics_path}: {e}")
        
        response = make_response(json.dumps(all_metrics, indent=2))
        response.headers["Content-Disposition"] = "attachment; filename=metriques_modeles.json"
        response.headers["Content-Type"] = "application/json"
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/results/<path:filename>', methods=['GET'])
@login_required
def api_export_results(filename):
    """Exporte les résultats de prédiction"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({"error": "Fichier non trouvé"}), 404
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/template', methods=['GET'])
@login_required
def api_download_template():
    """Télécharge un modèle de fichier CSV pour les prédictions"""
    try:
        from io import StringIO
        
        # Charger les features depuis le modèle Random Forest
        features_path = os.path.join(BACKEND_MODELS_DIR, "features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                features_meta = json.load(f)
            feature_names = features_meta.get('feature_names', [])
        else:
            feature_names = ['Attendance', 'Projects_Score', 'notecc', 'SN', 'Sleep_hour', 'Stress_Level (1-10)']
        
        # Créer un CSV template
        csv_data = StringIO()
        csv_data.write(','.join(feature_names) + '\n')
        # Ajouter une ligne d'exemple
        csv_data.write(','.join([str(5.0)] * len(feature_names)) + '\n')
        
        from flask import make_response
        response = make_response(csv_data.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=template_predictions.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/stats', methods=['GET'])
@login_required
def api_data_stats():
    """API pour récupérer les statistiques du fichier student_stored_data.csv"""
    try:
        student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        
        if not os.path.exists(student_csv_path):
            return jsonify({
                'total_students': 0,
                'admis_count': 0,
                'echec_count': 0,
                'last_update': 'N/A',
                'departments': {},
                'levels': {}
            })
        
        df = pd.read_csv(student_csv_path)
        df.columns = df.columns.str.strip()
        
        # Statistiques de base
        total_students = len(df)
        admis_count = len(df[df['decision'].str.strip() == 'Admis'])
        echec_count = len(df[df['decision'].str.strip() == 'Echec'])
        
        # Département
        departments = df['Department'].value_counts().to_dict() if 'Department' in df.columns else {}
        
        # Niveau d'étude
        levels = df['Level'].value_counts().to_dict() if 'Level' in df.columns else {}
        
        # Dernière mise à jour (date du fichier)
        last_update = datetime.fromtimestamp(os.path.getmtime(student_csv_path)).strftime('%d/%m/%Y %H:%M')
        
        # Moyennes
        avg_attendance = df['Attendance '].mean() if 'Attendance ' in df.columns else 0
        avg_projects = df['Projects_Score'].mean() if 'Projects_Score' in df.columns else 0
        avg_sn = df['SN'].mean() if 'SN' in df.columns else 0
        avg_cc = df['notecc'].mean() if 'notecc' in df.columns else 0
        
        return jsonify({
            'total_students': int(total_students),
            'admis_count': int(admis_count),
            'echec_count': int(echec_count),
            'admission_rate': round((admis_count / total_students * 100) if total_students > 0 else 0, 1),
            'last_update': last_update,
            'departments': departments,
            'levels': levels,
            'avg_attendance': round(float(avg_attendance), 2),
            'avg_projects': round(float(avg_projects), 2),
            'avg_sn': round(float(avg_sn), 2),
            'avg_cc': round(float(avg_cc), 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/list', methods=['GET'])
@login_required
def api_data_list():
    """API pour récupérer les données du fichier student_stored_data.csv"""
    try:
        student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        
        if not os.path.exists(student_csv_path):
            return jsonify({'data': []})
        
        # Récupérer les paramètres de pagination et filtrage
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '', type=str)
        decision_filter = request.args.get('decision', '', type=str)
        department_filter = request.args.get('department', '', type=str)
        
        df = pd.read_csv(student_csv_path)
        df.columns = df.columns.str.strip()
        
        # Appliquer les filtres
        if search:
            search_lower = search.lower()
            df = df[
                df['Matricule'].astype(str).str.lower().str.contains(search_lower, na=False) |
                df['First_Name'].astype(str).str.lower().str.contains(search_lower, na=False) |
                df['Last_Name'].astype(str).str.lower().str.contains(search_lower, na=False) |
                df['Email'].astype(str).str.lower().str.contains(search_lower, na=False)
            ]
        
        if decision_filter:
            df = df[df['decision'].str.strip() == decision_filter]
        
        if department_filter:
            df = df[df['Department'] == department_filter]
        
        # Pagination
        total_count = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        df_page = df.iloc[start_idx:end_idx]
        
        # Convertir en JSON
        data = []
        for _, row in df_page.iterrows():
            data.append({
                'matricule': str(row.get('Matricule', 'N/A')),
                'first_name': str(row.get('First_Name', '')),
                'last_name': str(row.get('Last_Name', '')),
                'email': str(row.get('Email', '')),
                'gender': str(row.get('Gender', 'N/A')),
                'age': int(row.get('Age', 0)) if pd.notna(row.get('Age')) else 0,
                'department': str(row.get('Department', 'N/A')),
                'level': str(row.get('Level', 'N/A')),
                'attendance': round(float(row.get('Attendance ', 0)), 2) if pd.notna(row.get('Attendance ')) else 0,
                'projects_score': round(float(row.get('Projects_Score', 0)), 2) if pd.notna(row.get('Projects_Score')) else 0,
                'sn': round(float(row.get('SN', 0)), 2) if pd.notna(row.get('SN')) else 0,
                'cc': round(float(row.get('notecc', 0)), 2) if pd.notna(row.get('notecc')) else 0,
                'moyenne_final': round(float(row.get('Moyenne_Final', 0)), 2) if pd.notna(row.get('Moyenne_Final')) else 0,
                'decision': str(row.get('decision', 'N/A')).strip(),
                'annee': int(row.get('Annee', 2024)) if pd.notna(row.get('Annee')) else 2024
            })
        
        return jsonify({
            'data': data,
            'total_count': int(total_count),
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/export', methods=['GET'])
@login_required
def api_data_export():
    """API pour exporter les données du fichier student_stored_data.csv"""
    try:
        student_csv_path = os.path.join(BACKEND_DATA_DIR, "student_stored_data.csv")
        
        if not os.path.exists(student_csv_path):
            return jsonify({"error": "Aucune donnée à exporter"}), 404
        
        # Lire le fichier
        df = pd.read_csv(student_csv_path)
        
        # Créer la réponse
        from io import StringIO
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        response = make_response(csv_buffer.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=student_data_export.csv"
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
