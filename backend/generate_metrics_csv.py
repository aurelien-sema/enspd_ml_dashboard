"""
Génère un fichier CSV avec les métriques de tous les modèles
Lit le contenu de metrics.json de chaque modèle et crée un fichier CSV
"""
import os
import json
import csv
from collections import OrderedDict

# Dossier contenant les modèles
MODELS_DIR = "models"
OUTPUT_FILE = "metrics_comparison.csv"

# Clés à exclure
EXCLUDE_KEYS = {"feature_importance", "confusion_matrix", "train_cost_history", "val_cost_history"}


def get_all_metric_keys():
    """Collecte toutes les clés de métriques de tous les modèles."""
    all_keys = set()
    
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.isdir(model_path):
            continue
        
        metrics_file = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                for key in metrics.keys():
                    if key not in EXCLUDE_KEYS:
                        all_keys.add(key)
    
    return sorted(list(all_keys))


def load_model_metrics(model_name):
    """Charge les métriques d'un modèle."""
    metrics_file = os.path.join(MODELS_DIR, model_name, "metrics.json")
    
    if not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Filtrer les clés exclues
    filtered_metrics = {k: v for k, v in metrics.items() if k not in EXCLUDE_KEYS}
    
    return filtered_metrics


def main():
    print("=" * 70)
    print("GÉNÉRATION DU FICHIER CSV - COMPARAISON DES MÉTRIQUES")
    print("=" * 70)
    
    # Vérifier que le dossier models existe
    if not os.path.isdir(MODELS_DIR):
        print(f"Erreur: le dossier '{MODELS_DIR}' n'existe pas.")
        return
    
    # Obtenir tous les noms de modèles
    model_names = sorted([d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))])
    
    if not model_names:
        print(f"Erreur: aucun modèle trouvé dans '{MODELS_DIR}'.")
        return
    
    print(f"\nModèles trouvés: {', '.join(model_names)}")
    
    # Obtenir toutes les clés de métriques
    all_metric_keys = get_all_metric_keys()
    
    print(f"\nMétriques collectées: {len(all_metric_keys)}")
    for key in all_metric_keys:
        print(f"  - {key}")
    
    # Créer le fichier CSV
    print(f"\nGénération du fichier CSV: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        # En-têtes: modèle + toutes les métriques
        fieldnames = ["Model"] + all_metric_keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Écrire l'en-tête
        writer.writeheader()
        
        # Écrire les données de chaque modèle
        for model_name in model_names:
            metrics = load_model_metrics(model_name)
            
            if metrics is not None:
                row = {"Model": model_name}
                
                # Ajouter les métriques
                for key in all_metric_keys:
                    value = metrics.get(key, "")
                    # Formater les floats
                    if isinstance(value, float):
                        row[key] = f"{value:.6f}"
                    else:
                        row[key] = value
                
                writer.writerow(row)
                print(f"  ✓ {model_name}")
            else:
                print(f"  ✗ {model_name} - metrics.json non trouvé")
    
    print("\n" + "=" * 70)
    print(f"FICHIER CSV GÉNÉRÉ: {OUTPUT_FILE}")
    print("=" * 70)
    
    # Afficher un aperçu du fichier
    print("\nAperçu du fichier:")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5]):
            print(line.rstrip())
        if len(lines) > 5:
            print(f"... ({len(lines) - 5} lignes supplémentaires)")


if __name__ == "__main__":
    main()
