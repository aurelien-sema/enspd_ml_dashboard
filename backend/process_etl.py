#!/usr/bin/env python3
"""
process_etl.py
Script ETL qui traite final.csv et produit processed_final.csv
Fonctionnalités:
 - calcule notecc = ((Assignments_Avg + Quizzes_Avg + Participation_Score)/3)*0.3
 - calcule SN = ((Midterm_Score + Final_Score)/2)*0.5
 - remplace Attendance par Attendance * 0.1
 - remplace Projects_Score par Projects_Score * 0.1
 - calcule Moyenne_Final = ((notecc + SN + Attendance + Projects_Score)) * 0.2
 - calcule Sleep_hour = Study_Hours_per_Week / Sleep_Hours_per_Night
 - ajoute Anonymat_name = "Stud 1", "Stud 2", ...
 - ajoute Password = "1234"
 - remplace aléatoirement Department par valeurs de la liste fournie
 - remplit Parent_Education_Level manquants par tirage aléatoire
 - ajoute la colonne 'decision' (Admis / Echec selon Moyenne_Final >= 12)
 - ajoute la colonne 'Annee' (2020..2024 réparties par blocs d'environ 1000 lignes)
 - sauvegarde le CSV traité
"""
import pandas as pd
import numpy as np
import os

# Chemins 
input_path = "./final.csv"
output_path = "./processed_final.csv"

# Lecture 
try:
    df = pd.read_csv(input_path, low_memory=False)
except Exception:
    df = pd.read_csv(input_path, encoding='latin-1', low_memory=False)

print(f"Fichier lu : {input_path} — {df.shape[0]} lignes × {df.shape[1]} colonnees")


# Détection des colonnes (ton choix explicite)
assign_col = "Assignments_Avg"
quiz_col = "Quizzes_Avg"
part_col = "Participation_Score"
mid_col = "Midterm_Score"
final_col = "Final_Score"
attendance_col = "Attendance (%)"
projects_col = "Projects_Score"
study_col = "Study_Hours_per_Week"
sleep_col = "Sleep_Hours_per_Night"
dept_col = "Department"
Level_col = "Parent_Education_Level"

# Afficher les colonnes détectées
print("Colonnes détectées :")
print(f"  assignments: {assign_col}")
print(f"  quizzes:    {quiz_col}")
print(f"  participation: {part_col}")
print(f"  midterm:    {mid_col}")
print(f"  final:      {final_col}")
print(f"  attendance: {attendance_col}")
print(f"  projects:   {projects_col}")
print(f"  study_hours:{study_col}")
print(f"  sleep_hours:{sleep_col}")
print(f"  department: {dept_col}")
print(f"  parent_education_level: {Level_col}")

# Conversion en numérique pour les colonnes nécessaires (coerce -> NaN si invalide)
for col in [assign_col, quiz_col, part_col, mid_col, final_col, projects_col, study_col, sleep_col, attendance_col]:
    if col is not None and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Calcul notecc
if assign_col in df.columns and quiz_col in df.columns and part_col in df.columns:
    df['notecc'] = ((df[assign_col].fillna(0) + df[quiz_col].fillna(0) + df[part_col].fillna(0)) / 3.0) * 0.3
else:
    df['notecc'] = np.nan
    print("notecc non calculée (une ou plusieurs colonnes manquantes)")

# Calcul SN
if mid_col in df.columns and final_col in df.columns:
    df['SN'] = ((df[mid_col].fillna(0) + df[final_col].fillna(0)) / 2.0) * 0.5
else:
    df['SN'] = np.nan
    print("SN non calculée (midterm ou final manquant)")

# Modifier Attendance -> *0.1
if attendance_col in df.columns:
    df[attendance_col] = pd.to_numeric(df[attendance_col], errors='coerce').astype(float) * 0.1
else:
    print("Attention: colonne Attendance introuvable — aucune modification appliquée.")

# Modifier Projects_Score -> *0.1
if projects_col in df.columns:
    df[projects_col] = pd.to_numeric(df[projects_col], errors='coerce').astype(float) * 0.1
else:
    print("Attention: colonne Projects_Score introuvable — aucune modification appliquée.")

# Calcul Moyenne_Final = (notecc + SN + Attendance + Projects_Score) * 0.2  (0-100 -> 0-20)
if ('notecc' in df.columns) and ('SN' in df.columns) and (attendance_col in df.columns) and (projects_col in df.columns):
    df['Moyenne_Final'] = ((df['notecc'].fillna(0) + df['SN'].fillna(0) + df[attendance_col].fillna(0) + df[projects_col].fillna(0))) * 0.2
else:
    df['Moyenne_Final'] = np.nan
    print("Moyenne_Final partielle ou non calculée (manque notecc/SN/attendance/projects).")

# Sleep_hour = Study_Hours_per_Week / Sleep_Hours_per_Night
if study_col in df.columns and sleep_col in df.columns:
    # éviter divisions par zéro
    df['Sleep_hour'] = df[study_col].replace({0: np.nan}) / df[sleep_col].replace({0: np.nan})
else:
    df['Sleep_hour'] = np.nan
    print("Sleep_hour non calculée (study or sleep column missing).")

# Anonymat_name & Password
df['Anonymat_name'] = ["Stud " + str(i) for i in range(1, len(df) + 1)]
df['Password'] = "1234"

# Remplacement aléatoire de Department par la liste fournie
departments = ["GIT","GESI","QHSEI","GAM","GMP","GPR","GE","GM","GP","GCI"]
rng = np.random.default_rng(42)  # seed fixe pour reproductibilité 
if dept_col in df.columns:
    df[dept_col] = rng.choice(departments, size=len(df))
else:
    df['Department'] = rng.choice(departments, size=len(df))

# Remplacement des valeurs manquantes dans Parent_Education_Level    
if Level_col not in df.columns:
    # Si la colonne n'existe pas, on la crée (optionnel)
    df[Level_col] = np.nan

# Détecter manquants : NaN OU chaînes vides ou 'nan' en texte
is_missing = df[Level_col].isna() | df[Level_col].astype(str).str.strip().isin(["", "nan", "None", "NaN"])
n_missing = int(is_missing.sum())
print(f"Colonne trouvée : '{Level_col}'. Valeurs manquantes détectées : {n_missing}")

# Remplacement aléatoire (seed pour reproductibilité)
choices = ["Master's", "High School", "PhD", "Bachelor's"]
rng2 = np.random.default_rng(42)
if n_missing > 0:
    df.loc[is_missing, Level_col] = rng2.choice(choices, size=n_missing)
    print(f"{n_missing} valeurs manquantes remplacées par tirage aléatoire parmi {choices}.")
else:
    print("Aucune valeur manquante à remplacer.")    

# ---------------------------
# NOUVEAUX AJOUTS : decision & Annee
# ---------------------------

# 1) Colonne "decision" : "Admis" si Moyenne_Final >= 12, sinon "Echec"
#    On considère que les NaN de Moyenne_Final donnent "Echec" (option conservatrice).
df['decision'] = np.where(df['Moyenne_Final'].fillna(-1) >= 12, "Admis", "Echec")

# 2) Colonne "Annee" : répartir par blocs d'environ 1000 lignes
n = len(df)
idx = np.arange(n)  # 0..n-1
# Attribution par tranche
df['Annee'] = 2024  # valeur par défaut (fin)
df.loc[idx < 1000, 'Annee'] = 2020
df.loc[(idx >= 1000) & (idx < 2000), 'Annee'] = 2021
df.loc[(idx >= 2000) & (idx < 3000), 'Annee'] = 2022
df.loc[(idx >= 3000) & (idx < 4000), 'Annee'] = 2023
# les idx >= 4000 restent 2024 (par défaut)

# Sauvegarde du fichier traité
df.to_csv(output_path, index=False)
print(f"Fichier traité sauvegardé : {output_path}")
print("Exécution terminée.")



# """
# process_etl.py
# Script ETL qui traite final.csv et produit processed_final.csv
# Fonctionnalités:
#  - calcule notecc = ((Assignments_Avg + Quizzes_Avg + Participation_Score)/3)*0.3
#  - calcule SN = ((Midterm_Score + Final_Score)/2)*0.5
#  - remplace Attendance par Attendance * 0.1
#  - remplace Projects_Score par Projects_Score * 0.1
#  - calcule Moyenne_Final = ((notecc + SN + Attendance + Projects_Score))*0.2
#  - calcule Sleep_hour = Study_Hours_per_Week / Sleep_Hours_per_Night
#  - ajoute Anonymat_name = "Stud 1", "Stud 2", ...
#  - ajoute Password = "1234"
#  - remplace aléatoirement Department par valeurs de la liste fournie
#  - sauvegarde le CSV traité
# """
# import pandas as pd
# import numpy as np
# import os

# # Chemins 
# input_path = "./final.csv"
# output_path = "./processed_final.csv"

# # Lecture 
# try:
#     df = pd.read_csv(input_path, low_memory=False)
# except Exception:
#     df = pd.read_csv(input_path, encoding='latin-1', low_memory=False)

# print(f"Fichier lu : {input_path} — {df.shape[0]} lignes × {df.shape[1]} colonnees")


# # Détection des colonnes 
# assign_col = "Assignments_Avg"
# quiz_col = "Quizzes_Avg"
# part_col = "Participation_Score"
# mid_col = "Midterm_Score"
# final_col = "Final_Score"
# attendance_col = "Attendance (%)"
# projects_col = "Projects_Score"
# study_col = "Study_Hours_per_Week"
# sleep_col = "Sleep_Hours_per_Night"
# dept_col = "Department"
# Level_col = "Parent_Education_Level"

# # Afficher les colonnes détectées
# print("Colonnes détectées :")
# print(f"  assignments: {assign_col}")
# print(f"  quizzes:    {quiz_col}")
# print(f"  participation: {part_col}")
# print(f"  midterm:    {mid_col}")
# print(f"  final:      {final_col}")
# print(f"  attendance: {attendance_col}")
# print(f"  projects:   {projects_col}")
# print(f"  study_hours:{study_col}")
# print(f"  sleep_hours:{sleep_col}")
# print(f"  department: {dept_col}")
# print(f"  parent_education_level: {Level_col}")

# # Conversion en numérique pour les colonnes nécessaires (coerce -> NaN si invalide)
# for col in [assign_col, quiz_col, part_col, mid_col, final_col, projects_col, study_col, sleep_col, attendance_col]:
#     if col is not None:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

# # Calcul notecc
# if assign_col and quiz_col and part_col:
#     df['notecc'] = ((df[assign_col].fillna(0) + df[quiz_col].fillna(0) + df[part_col].fillna(0)) / 3.0) * 0.3
# else:
#     df['notecc'] = np.nan
#     print("notecc non calculée (une ou plusieurs colonnes manquantes)")

# # Calcul SN
# if mid_col and final_col:
#     df['SN'] = ((df[mid_col].fillna(0) + df[final_col].fillna(0)) / 2.0) * 0.5
# else:
#     df['SN'] = np.nan
#     print("SN non calculée (midterm ou final manquant)")

# # Modifier Attendance -> *0.1
# if attendance_col:
#     df[attendance_col] = df[attendance_col].astype(float) * 0.1
# else:
#     print("Attention: colonne Attendance introuvable — aucune modification appliquée.")

# # Modifier Projects_Score -> *0.1
# if projects_col:
#     df[projects_col] = df[projects_col].astype(float) * 0.1
# else:
#     print("Attention: colonne Projects_Score introuvable — aucune modification appliquée.")

# # Calcul Moyenne_Final = ((notecc + SN + Attendance + Projects_Score)) * 0.2
# if ('notecc' in df.columns) and ('SN' in df.columns) and (attendance_col is not None) and (projects_col is not None):
#     df['Moyenne_Final'] = ((df['notecc'].fillna(0) + df['SN'].fillna(0) + df[attendance_col].fillna(0) + df[projects_col].fillna(0))) * 0.2
# else:
#     df['Moyenne_Final'] = np.nan
#     print("Moyenne_Final partielle ou non calculée (manque notecc/SN/attendance/projects).")

# # Sleep_hour = Study_Hours_per_Week / Sleep_Hours_per_Night
# if study_col and sleep_col:
#     # éviter divisions par zéro
#     df['Sleep_hour'] = df[study_col].replace({0: np.nan}) / df[sleep_col].replace({0: np.nan})
# else:
#     df['Sleep_hour'] = np.nan
#     print("Sleep_hour non calculée (study or sleep column missing).")

# # Anonymat_name & Password
# df['Anonymat_name'] = ["Stud " + str(i) for i in range(1, len(df) + 1)]
# df['Password'] = "1234"

# # Remplacement aléatoire de Department par la liste fournie
# departments = ["GIT","GESI","QHSEI","GAM","GMP","GPR","GE","GM","GP","GCI"]
# rng = np.random.default_rng(42)  # seed fixe pour reproductibilité 
# if dept_col:
#     df[dept_col] = rng.choice(departments, size=len(df))
# else:
#     df['Department'] = rng.choice(departments, size=len(df))

# # Remplacement des valeurs manquantes dans Parent_Education_Level    
# if Level_col is None:
#     raise KeyError("Colonne 'Parent_Education_Level' introuvable — vérifie le nom exact dans ton CSV.")

# # Détecter manquants : NaN OU chaînes vides ou 'nan' en texte
# is_missing = df[Level_col].isna() | df[Level_col].astype(str).str.strip().isin(["", "nan", "None", "NaN"])

# n_missing = int(is_missing.sum())
# print(f"Colonne trouvée : '{Level_col}'. Valeurs manquantes détectées : {n_missing}")

# # Remplacement aléatoire (seed pour reproductibilité)
# choices = ["Master's", "High School", "PhD", "Bachelor's"]
# rng = np.random.default_rng(42)

# if n_missing > 0:
#     df.loc[is_missing, Level_col] = rng.choice(choices, size=n_missing)
#     print(f"{n_missing} valeurs manquantes remplacées par tirage aléatoire parmi {choices}.")
# else:
#     print("Aucune valeur manquante à remplacer.")    

# # Sauvegarde du fichier traité
# df.to_csv(output_path, index=False)
# print(f"Fichier traité sauvegardé : {output_path}")
# print("Exécution terminée.")
