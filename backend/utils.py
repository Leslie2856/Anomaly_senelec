import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import logging
# Ajouter ces lignes
from keras.models import load_model
from keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np  # Déjà présent, mais assurez-vous
import os
from typing import List, Tuple, Dict
import io
logger = logging.getLogger("uvicorn.error")

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Charger le scaler global à l'initialisation de l'application
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_global.pkl")
scaler = joblib.load(SCALER_PATH)



# --- Utility Functions for Anomaly Detection ---
def categoriser_partenaire(partenaire):
    partenaire = partenaire.lower()
    if 'brioché' in partenaire or 'brioche' in partenaire:
        return 'Brioche'
    elif 'hotel' in partenaire:
        return 'Hotel'
    elif 'restaurant' in partenaire:
        return 'Restaurant'
    elif 'glace' in partenaire:
        return 'Glace'
    else:
        return 'Autre'

def export_low_consumption(df_step, mois_courant, do_export=False):
    df_step['year_month'] = df_step['Data Time'].dt.to_period('M')
    df_month = df_step[df_step['year_month'] == mois_courant]
    if df_month.empty:
        return pd.DataFrame()
    
    def all_below_5kw(group):
        return group if group['Active power (+) total(kW)'].max() < 5 else pd.DataFrame()
    
    low_consumption = df_month.groupby('Meter No.', group_keys=False).apply(all_below_5kw)
    if low_consumption.empty:
        return pd.DataFrame()
    
    if do_export:
        cols_info = ['Meter No.', 'Rue', 'Categorie', 'Partenaire', 'Saison', 'Climat', 'year_month']
        low_consumption_unique = low_consumption[cols_info].drop_duplicates().reset_index(drop=True)
        return low_consumption_unique
    return low_consumption

def remove_non_consuming_meters(df, energy_col="+A (kWh)"):
    """
    Supprime les compteurs dont l'énergie active est toujours nulle.
    energy_col : nom exact de la colonne d'énergie dans tes fichiers (ex: 'Active energy (+) total(kWh)')
    """
    if energy_col not in df.columns:
        raise ValueError(f"Colonne {energy_col} non trouvée dans le DataFrame")

    # Regrouper par compteur et vérifier si toute la consommation est nulle
    meters_to_keep = df.groupby("Meter No.")[energy_col].transform("sum") > 0
    df_filtered = df[meters_to_keep].copy()

    return df_filtered

def load_and_clean(file_list: List[str], is_train: bool = True) -> pd.DataFrame:
    """
    Charge et nettoie les données à partir des fichiers.
    
    Args:
        file_list: Liste des chemins de fichiers
        is_train: Si True, supprime les outliers pour l'entraînement
    
    Returns:
        DataFrame nettoyé avec données quotidiennes
    """
    df_all = pd.DataFrame()
    
    for fichier in file_list:
        try:
            if fichier.endswith(".csv"):
                df = pd.read_csv(fichier, parse_dates=['Data Time'])
            else:
                df = pd.read_excel(fichier, parse_dates=['Data Time'])
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {fichier}: {e}")
            continue

        # Feature engineering
        df['hour'] = df['Data Time'].dt.hour
        df['day_of_week'] = df['Data Time'].dt.weekday
        df['month'] = df['Data Time'].dt.month
        df['Saison'] = df['month'].apply(lambda x: 'Sèche' if x in [11, 12, 1, 2, 3, 4, 5] else 'Pluvieuse')
        df['Climat'] = df['month'].apply(lambda x: 'Chaud' if x in [2, 3, 4, 5] else 'Froid' if x in [12, 1] else 'Modérée')
        df['year_month'] = df['Data Time'].dt.to_period('M')
        
        # Interpolation
        df = df.groupby('Meter No.', group_keys=False).apply(interpolate_group)
        df_all = pd.concat([df_all, df], ignore_index=True)

    # Suppression des outliers pour l'entraînement
    if is_train:
        df_all = df_all.groupby('Meter No.', group_keys=False).apply(detect_outliers_group)
        df_all = df_all[~df_all['is_outlier']].drop(columns=['is_outlier'], errors='ignore')

    # Agrégation quotidienne
    df_daily = df_all.groupby(['Meter No.', df_all['Data Time'].dt.date]).agg({
        'Active power (+) total(kW)': ['sum', 'mean', 'std', 'max', 'min'],
        'hour': 'mean', 
        'day_of_week': 'first', 
        'month': 'first',
        'Saison': 'first', 
        'Climat': 'first', 
        'Categorie': 'first'
    }).reset_index()
    
    df_daily.columns = ['Meter No.', 'date', 'sum_kW', 'mean_kW', 'std_kW', 'max_kW', 'min_kW',
                       'hour_mean', 'day_of_week', 'month', 'Saison', 'Climat', 'Categorie']
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['std_kW'] = df_daily['std_kW'].fillna(0)
    
    # Encodage one-hot
    df_daily['Saison'] = pd.Categorical(df_daily['Saison'], categories=['Sèche', 'Pluvieuse'])
    df_daily['Climat'] = pd.Categorical(df_daily['Climat'], categories=['Chaud', 'Froid', 'Modérée'])
    df_daily = pd.get_dummies(df_daily, columns=['Saison', 'Climat'], prefix=['Saison', 'Climat'], dtype=int)
    
    # Assurer que toutes les colonnes features sont présentes
    for col in FEATURE_COLUMNS:
        if col not in df_daily.columns:
            df_daily[col] = 0
            
    return df_daily

def format_month_french(period):
    """Convertit une période (ex: '2025-07') en format français (ex: 'juillet 2025')"""
    if not period:
        return ""
    
    try:
        year, month = map(int, str(period).split('-'))
        month_names = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        return f"{month_names[month-1]} {year}"
    except (ValueError, IndexError):
        return str(period)
    
def interpolate_group(group):
    group = group.sort_values('Data Time')
    group['Active power (+) total(kW)'] = group['Active power (+) total(kW)'].interpolate(method='linear', limit=4).ffill().bfill()
    group['is_weekend'] = group['Data Time'].dt.weekday >= 5
    # Ajouter : Exemple de is_holiday (adaptez selon vos besoins réels ; ici, exemple pour 1er jan et 1er juil comme dans Kaggle)
    group['is_holiday'] = (group['Data Time'].dt.month.isin([1, 7]) & (group['Data Time'].dt.day == 1)).astype(int)
    return group

def detect_outliers_group(group):
    def detect_without_replace(sub):
        mean, std = sub['Active power (+) total(kW)'].mean(), sub['Active power (+) total(kW)'].std()
        is_outlier_low = (mean - sub['Active power (+) total(kW)']) > 4.5 * std
        is_outlier_high = (sub['Active power (+) total(kW)'] - mean) > 6.5 * std
        sub['is_outlier'] = is_outlier_low | is_outlier_high
        sub['Motif'] = np.where(sub['is_outlier'], 
                                np.where(sub['Active power (+) total(kW)'] < mean, 
                                         'Consommation faible par rapport à la moyenne', 
                                         'Consommation élevée par rapport à la moyenne'), 
                                '')
        return sub
    return group.groupby(['hour', 'is_weekend', 'Saison', 'Climat'], group_keys=False).apply(detect_without_replace)

def prepare_daily_data(df, scaler=None):
    df_daily = df.groupby(['Meter No.', df['Data Time'].dt.date]).agg({
        'Active power (+) total(kW)': ['sum', 'mean', 'std', 'max', 'min'],
        'hour': 'mean',
        'day_of_week': 'first',
        'month': 'first',
        'Saison': 'first',
        'Climat': 'first',
        'Categorie': 'first',
        'is_holiday': 'first'
    }).reset_index()
    
    df_daily.columns = ['Meter No.', 'date', 'sum_kW', 'mean_kW', 'std_kW', 'max_kW', 'min_kW', 
                        'hour_mean', 'day_of_week', 'month', 'Saison', 'Climat', 'Categorie', 'is_holiday']
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['std_kW'] = df_daily['std_kW'].fillna(0)
    df_daily = df_daily.dropna(subset=['sum_kW', 'mean_kW', 'max_kW', 'min_kW', 'hour_mean'])
    
    df_daily['Saison'] = pd.Categorical(df_daily['Saison'], categories=['Sèche', 'Pluvieuse'])
    df_daily['Climat'] = pd.Categorical(df_daily['Climat'], categories=['Chaud', 'Froid', 'Modérée'])
    df_daily = pd.get_dummies(df_daily, columns=['Saison', 'Climat'], prefix=['Saison', 'Climat'], dtype=int)
    
    # Ajouter les colonnes manquantes de FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        if col not in df_daily.columns:
            df_daily[col] = 0

    # Appliquer le scaler si fourni
    if scaler is not None:
        df_scaled = pd.DataFrame(scaler.transform(df_daily[FEATURE_COLUMNS]), 
                                 columns=FEATURE_COLUMNS, 
                                 index=df_daily.index)
        df_daily[FEATURE_COLUMNS] = df_scaled

    return df_daily

def create_sequences_keras_v2(df: pd.DataFrame, time_steps: int, feature_columns: List[str]) -> np.ndarray:
    """
    Crée des séquences temporelles pour Keras.
    - Si un compteur a moins de `time_steps` jours, complète la séquence avec la première valeur répétée.
    - Sinon, crée des séquences glissantes.
    """
    sequences = []
    meter_numbers = df['Meter No.'].unique()
    
    for meter in meter_numbers:
        df_meter = df[df['Meter No.'] == meter].sort_values('date')
        X_meter = df_meter[feature_columns].values
        n_days = len(X_meter)

        if n_days == 0:
            continue

        # Si la séquence est plus courte que time_steps, compléter avec la première valeur répétée
        if n_days < time_steps:
            padding = np.repeat(X_meter[0:1, :], time_steps - n_days, axis=0)
            X_padded = np.vstack([padding, X_meter])
            sequences.append(X_padded)
        else:
            # Séquences glissantes
            for i in range(n_days - time_steps + 1):
                sequences.append(X_meter[i:i + time_steps])
    
    return np.array(sequences)

def evaluate_autoencoder(model, X_sequences, df_meter, time_steps=24, threshold_multiplier=2.0):
    """
    Évalue l'autoencodeur sur les séquences et retourne les erreurs de reconstruction et les labels.
    Version adaptée de l'approche Kaggle.
    """
    if X_sequences.shape[0] == 0:
        logger.warning("Aucune séquence valide pour évaluation.")
        return np.array([]), np.array([])
    
    try:
        # Prédire les reconstructions
        X_recon = model.predict(X_sequences, batch_size=32, verbose=0)
        
        # Calculer l'erreur quadratique moyenne (MSE) par séquence
        mse_per_sequence = np.mean((X_sequences - X_recon) ** 2, axis=1)  # shape = (n_sequences, n_features)
        mse_per_sequence = mse_per_sequence.mean(axis=1)  # réduire sur features si nécessaire

        
        # Appliquer le seuil d'anomalie (approche Kaggle)
        mean_error = np.mean(mse_per_sequence)
        std_error = np.std(mse_per_sequence)
        threshold = mean_error * threshold_multiplier
        
        # Déterminer les anomalies
        is_anomaly_sequence = mse_per_sequence > threshold
        
        # Associer chaque séquence à son jour de départ
        n_days = len(df_meter)
        errors_per_day = np.full(n_days, np.nan)
        labels_per_day = np.zeros(n_days, dtype=int)
        
        for i in range(len(mse_per_sequence)):
            day_index = i  # Chaque séquence correspond à un jour
            if day_index < n_days:
                errors_per_day[day_index] = mse_per_sequence[i]
                if is_anomaly_sequence[i]:
                    labels_per_day[day_index] = 1
        
        # Remplir les NaN avec la moyenne
        errors_per_day = np.nan_to_num(errors_per_day, nan=np.nanmean(errors_per_day) if not np.all(np.isnan(errors_per_day)) else 0.0)
        
        return errors_per_day, labels_per_day
        
    except Exception as e:
        logger.error(f"Erreur dans evaluate_autoencoder: {str(e)}")
        return np.array([]), np.array([])
            
# --- Constants ---
# Remplacer par :
FEATURE_COLUMNS = [
    'sum_kW', 'mean_kW', 'std_kW', 'max_kW', 'min_kW', 
    'hour_mean', 'day_of_week', 'month', 
    'Saison_Sèche', 'Saison_Pluvieuse', 
    'Climat_Chaud', 'Climat_Froid', 'Climat_Modérée'
]

# Constants for data processing
SAISON_CATEGORIES = ['Sèche', 'Pluvieuse']
CLIMAT_CATEGORIES = ['Chaud', 'Froid', 'Modérée']

# Configuration (identique à Kaggle)
CONFIG = {
    'time_steps': 24,
    'hidden_dim': 50,
    'batch_size': 64,
    'lr': 0.001,
    'patience': 25,
    'feature_columns': FEATURE_COLUMNS
}


METER_NUMBERS = [
    '35000279', '35000641', '35000713', '35000926', '35001011', '35001110', '35001811', 
    '35002528', '35002865', '35002997', '35003442', '35005131', '35006365', '35006912', 
    '35006955', '35007175', '35007551', '35007552', '35007555', '35008076', '35008434', 
    '35008682', '35009119', '35010274', '35010423', '35010447', '35010501', '35011003', 
    '35011015', '35011064', '35011307', '35012040', '35012046', '35012186', '35012210', 
    '35012322', '35012739', '35012916', '35014138', '35014700', '35014834', '35014912', 
    '35015618', '35016185', '35016566', '35017173', '35017418', '35017601', '35017616', 
    '35017882', '35017981', '35100115', '35100149', '35100249', '35100310', '35100341', 
    '35100395', '35100396', '35100429', '35100434', '35100435', '35100507', '35100537', 
    '35100608', '35100645', '35100678', '35100681', '35100786', '35100834', '35100845', 
    '35100961', '36000110', '36000119', '36000270', '36000281', '36000292', '36000502', 
    '36000532', '36000568', '36000667', '36000683', '36000871', '36000876', '36001100', 
    '36001237', '36001968', '36001975', '36001990', '36002087', '36002222', '36002277', 
    '36002386', '36002408', '36002561', '36002568', '36002595', '36002685', '36002740', 
    '36002826', '36002863', '36002929', '36003006', '36003229', '36003287', '36003311', 
    '36003326', '36003473', '36003476', '36003478', '36003582', '36003585', '36003732', 
    '36003802', '36003808', '36003844', '36003940', '36003949', '36004140', '36004293', 
    '36004297', '36004353', '36004356', '36004363', '36004389', '36004391', '36004574', 
    '36004623', '36004643', '36004657', '36004679', '36004743', '36004759', '36004763', 
    '36004771', '36004845', '36004943', '36004981', '36004999', '36005021', '36005026', 
    '36005027', '36005045'
]

# --- Existing Utility Functions ---
def clean_and_preprocess(df1: pd.DataFrame, df2: pd.DataFrame, df_list: pd.DataFrame) -> pd.DataFrame:
    # --- Définition des colonnes standard ---
    standard_columns = [
        "No.", "Area Name", "Meter No.", "Freeze Date", "Consume Type", "Account ID", "Month",
        "A1T1(KWh)", "A1T2(KWh)",
        "+A(KWh)", "+A T1(KWh)", "+A T2(KWh)", "+A T3(KWh)", "+A T4(KWh)", "+A T5(KWh)", "+A T6(KWh)", "+A T7(KWh)", "+A T8(KWh)",
        "-A(KWh)", "-A T1(KWh)", "-A T2(KWh)", "-A T3(KWh)", "-A T4(KWh)", "-A T5(KWh)", "-A T6(KWh)", "-A T7(KWh)", "-A T8(KWh)",
        "+A L1(KWh)", "+A L2(KWh)", "+A L3(KWh)", "-A L1(KWh)", "-A L2(KWh)", "-A L3(KWh)",
        "+R(kvarh)", "+R T1(kvarh)", "+R T2(kvarh)", "+R T3(kvarh)", "+R T4(kvarh)", "+R T5(kvarh)", "+R T6(kvarh)", "+R T7(kvarh)", "+R T8(kvarh)",
        "-R(kvarh)", "-R T1(kvarh)", "-R T2(kvarh)", "-R T3(kvarh)", "-R T4(kvarh)", "-R T5(kvarh)", "-R T6(kvarh)", "-R T7(kvarh)", "-R T8(kvarh)",
        "+R L1(kvarh)", "+R L2(kvarh)", "+R L3(kvarh)", "-R L1(kvarh)", "-R L2(kvarh)", "-R L3(kvarh)",
        "+MDA(kW)", "+MDA Time Stamp", "MD+L1(kW)", "MD+L2(kW)", "MD+L3(kW)", "-MDA(kW)", "PowerOnTime(h)", "-MDA Time Stamp",
        "+MDR(kW)", "+MDR Time Stamp", "Balance", "Model Name", "Manufacturer",
    ]

    # --- Étape 1 : Standardiser df1 et df2 ---
    dfs = []
    for df in [df1, df2]:
        df.columns = [str(c).strip() for c in df.columns]
        col_map = {col: std for std in standard_columns for col in df.columns if col.lower() == std.lower()}
        df = df.rename(columns=col_map)
        for col in standard_columns:
            if col not in df.columns:
                df[col] = pd.NA
        dfs.append(df[standard_columns])

    df_final = pd.concat(dfs, ignore_index=True)

    # --- Étape 2 : Colonnes pertinentes ---
    colonnes = ["Meter No.", "Freeze Date", "+A(KWh)", "+A T1(KWh)", "+A T2(KWh)", 
                "-A(KWh)", "+A L1(KWh)", "+A L2(KWh)", "+A L3(KWh)", "+R(kvarh)", "-R(kvarh)"]
    df_cleaned = df_final[colonnes].copy()

    # --- Étape 3 : Préparer df_list ---
    df_list.columns = [str(c).strip() for c in df_list.columns]
    col_map_list = {col: "Meter No." for col in df_list.columns if col.lower() == "numero de serie (numero compteur)".lower()}
    df_list = df_list.rename(columns=col_map_list)
    if "Meter No." not in df_list.columns:
        raise ValueError("La colonne 'Numero de serie (Numero compteur)' est absente dans df_list")
    
    # Nettoyer les compteurs dans df_list en enlevant les zéros initiaux
    df_list["Meter No."] = df_list["Meter No."].astype(str).str.strip().str.lstrip("0")
    # Nettoyer df_cleaned Meter No.
    df_cleaned["Meter No."] = df_cleaned["Meter No."].astype(str).str.strip()

    # --- Étape 4 : Convertir en str, strip et majuscules ---
    df_cleaned["Meter No."] = df_cleaned["Meter No."].astype(str).str.strip().str.upper()
    df_list["Meter No."] = df_list["Meter No."].astype(str).str.strip().str.upper()

    # --- Étape 5 : Ajouter Partenaire et Rue depuis df_list ---
    df_cleaned = df_cleaned.merge(df_list[["Meter No.", "Partenaire", "Rue"]], on="Meter No.", how="left")

    # --- Étape 6 : Filtrer selon df_list ---
    valid_meters = df_list["Meter No."].dropna().unique()
    df_cleaned = df_cleaned[df_cleaned["Meter No."].isin(valid_meters)].reset_index(drop=True)

    # --- Étape 7 : Date au 1er jour du mois ---
    df_cleaned["Freeze Date"] = pd.to_datetime(df_cleaned["Freeze Date"], errors="coerce")
    df_cleaned = df_cleaned[df_cleaned["Freeze Date"].dt.day == 1].reset_index(drop=True)

    # --- Étape 8 : Supprimer +A(KWh) nul ---
    df_cleaned = df_cleaned[~((df_cleaned["+A(KWh)"].isna()) | (df_cleaned["+A(KWh)"] == 0))].reset_index(drop=True)

    # --- Étape 9 : Remplissages ---
    df_cleaned["+A T1(KWh)"] = df_cleaned["+A T1(KWh)"].fillna(df_cleaned["+A(KWh)"] * 0.95)
    df_cleaned["+A T2(KWh)"] = df_cleaned["+A T2(KWh)"].fillna(df_cleaned["+A(KWh)"] * 0.05)
    for phase in ['L1', 'L2', 'L3']:
        col = f'+A {phase}(KWh)'
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned["+A(KWh)"] / 3)

    # --- Étape 10 : Supprimer lignes avec NaN réactif ---
    cols_test = ["+R(kvarh)", "-A(KWh)", "-R(kvarh)"]
    df_cleaned.dropna(subset=cols_test, inplace=True)

    # --- Étape 11 : Nettoyage final ---
    df_cleaned.drop_duplicates(inplace=True)

    df_cleaned.sort_values(by=["Meter No.", "Freeze Date"], inplace=True)
    last_two_months = (
        df_cleaned["Freeze Date"]
        .dropna()
        .sort_values()
        .dt.to_period("M")
        .drop_duplicates()
        .iloc[-2:]
    )

    df_cleaned = df_cleaned[df_cleaned["Freeze Date"].dt.to_period("M").isin(last_two_months)].reset_index(drop=True)

    compteur_counts = df_cleaned["Meter No."].value_counts()
    compteur_valides = compteur_counts[compteur_counts == 2].index
    df_cleaned = df_cleaned[df_cleaned["Meter No."].isin(compteur_valides)].reset_index(drop=True)

    return df_cleaned

mois_fr = {
    "01": "Janvier", "02": "Février", "03": "Mars", "04": "Avril",
    "05": "Mai", "06": "Juin", "07": "Juillet", "08": "Août",
    "09": "Septembre", "10": "Octobre", "11": "Novembre", "12": "Décembre"
}

def calculate_evolutions(df: pd.DataFrame):
    df = df.copy()

    # S'assurer que les dates sont bien en datetime
    df["Freeze Date"] = pd.to_datetime(df["Freeze Date"], errors="coerce")

    # Trier pour bien aligner les valeurs mois précédent / mois courant
    df.sort_values(by=["Meter No.", "Freeze Date"], inplace=True)

    # Calcul évolution +A
    df['prev_+A(KWh)'] = df.groupby("Meter No.")["+A(KWh)"].shift(1)
    df['diff_+A(KWh)'] = df["+A(KWh)"] - df['prev_+A(KWh)']

    # Identifier les compteurs non consommants dans le mois courant (diff == 0)
    df['is_last_month'] = df.groupby("Meter No.")["Freeze Date"].transform("max") == df["Freeze Date"]
    anomalies_non_evol = df[df['is_last_month'] & (df['diff_+A(KWh)'].fillna(0) == 0)].copy()

    anomalies_non_evol = anomalies_non_evol[[
        "Meter No.", "Partenaire", "Rue", "prev_+A(KWh)", "+A(KWh)", "diff_+A(KWh)"
    ]]
    anomalies_non_evol.rename(columns={
        "prev_+A(KWh)": "+A mois précédent",
        "+A(KWh)": "+A mois en cours",
        "diff_+A(KWh)": "évol +A"
    }, inplace=True)
    anomalies_non_evol["Anomalie"] = "Compteur non consommant"

    # Supprimer ces compteurs du df principal
    df = df[~df["Meter No."].isin(anomalies_non_evol["Meter No."])].copy()

    # Calculs des évolutions
    df['evol_+A_T2'] = df.groupby('Meter No.')['+A T2(KWh)'].diff().fillna(df['+A T2(KWh)'])
    df['evol_-A'] = df.groupby('Meter No.')['-A(KWh)'].diff().fillna(df['-A(KWh)'])

    for phase in ['L1', 'L2', 'L3']:
        col = f'+A {phase}(KWh)'
        df[f'evol_+A_{phase}'] = df.groupby('Meter No.')[col].diff().fillna(df[col])

    df['cumul_evol_AL'] = df[[f'evol_+A_L1', f'evol_+A_L2', f'evol_+A_L3']].sum(axis=1)

    for phase in ['L1', 'L2', 'L3']:
        evol_col = f'evol_+A_{phase}'
        df[f'pct_evol_{phase}'] = (
            df[evol_col] / df['cumul_evol_AL']
        ).replace([np.inf, -np.inf], 0).fillna(0) * 100

    df['rapport_R'] = np.where(df['-R(kvarh)'] == 0, 0,
                              (df['+R(kvarh)'] / df['-R(kvarh)']) * 100)

    # Supprimer les lignes avec évolutions négatives et leur ligne précédente
    lignes_a_supprimer = set()
    for col in ['evol_+A_T2', 'evol_-A', 'evol_+A_L1', 'evol_+A_L2', 'evol_+A_L3']:
        idx = df[df[col] < 0].index
        lignes_a_supprimer.update(idx)
        lignes_a_supprimer.update([i - 1 for i in idx if i > 0])
    df = df.drop(index=lignes_a_supprimer).reset_index(drop=True)

    # Dernier mois en texte
    dernier_mois = df["Freeze Date"].dt.to_period("M").max()
    mois_string = f"{mois_fr[dernier_mois.strftime('%m')]} {dernier_mois.strftime('%Y')}"

    df = df[df["Freeze Date"].dt.to_period("M") == dernier_mois].reset_index(drop=True)

    return df, anomalies_non_evol, mois_string

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['K2 bloque'] = np.where(df['evol_+A_T2'] == 0, 'K2 bloque', 'Normal')
    df['Inversion'] = np.where(df['evol_-A'] > 0, 'Inversion', 'Normal')
    df['Mono 1 bloque'] = np.where(df['evol_+A_L1'] == 0, 'Mono 1 bloque', 'Normal')
    df['Mono 2 bloque'] = np.where(df['evol_+A_L2'] == 0, 'Mono 2 bloque', 'Normal')
    df['Mono 3 bloque'] = np.where(df['evol_+A_L3'] == 0, 'Mono 3 bloque', 'Normal')
    df['Mono 1 faible'] = np.where(df['pct_evol_L1'] < 5, 'Mono 1 faible', 'Normal')
    df['Mono 2 faible'] = np.where(df['pct_evol_L2'] < 5, 'Mono 2 faible', 'Normal')
    df['Mono 3 faible'] = np.where(df['pct_evol_L3'] < 5, 'Mono 3 faible', 'Normal')
    df['Surcompensation'] = np.where(df['rapport_R'] < 10, 'Surcompensation', 'Normal')
    return df

def generate_report(df, mois_string):
    f5_dict = {}

    # --- K2 bloque ---
    df_k2 = df[df["K2 bloque"] == "K2 bloque"].copy()
    df_k2["Période"] = mois_string
    df_k2["Ancien index +A T2 (KWh)"] = df_k2["+A T2(KWh)"] - df_k2["evol_+A_T2"]
    df_k2["Anomalie"] = "K2 bloque"
    df_k2 = df_k2.sort_values(by="+A T2(KWh)", ascending=False)
    f5_dict["K2 bloque"] = df_k2[[
        "Meter No.", "Partenaire", "Rue", "Ancien index +A T2 (KWh)", "+A T2(KWh)", "evol_+A_T2", "Anomalie"
    ]].rename(columns={"+A T2(KWh)": "Nouvel index +A T2 (KWh)"})

    # --- Inversion ---
    df_inv = df[df["Inversion"] == "Inversion"].copy()
    df_inv["Période"] = mois_string
    df_inv["Ancien index -A (KWh)"] = df_inv["-A(KWh)"] - df_inv["evol_-A"]
    df_inv["Anomalie"] = "Inversion"
    df_inv = df_inv.sort_values(by="evol_-A", ascending=False)
    f5_dict["Inversion"] = df_inv[[
        "Meter No.", "Partenaire", "Rue", "Ancien index -A (KWh)", "-A(KWh)", "evol_-A", "Anomalie"
    ]].rename(columns={"-A(KWh)": "Nouvel index -A (KWh)"})

    # --- Mono bloque ---
    df_mono = df[
        (df['Mono 1 bloque'] == 'Mono 1 bloque') |
        (df['Mono 2 bloque'] == 'Mono 2 bloque') |
        (df['Mono 3 bloque'] == 'Mono 3 bloque')
    ].copy()
    df_mono["Période"] = mois_string
    df_mono["Anomalie"] = df_mono[['Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque']].apply(
        lambda row: ' + '.join([v for v in row if v != 'Normal']), axis=1)
    for phase in ['L1', 'L2', 'L3']:
        df_mono[f"Ancien index {phase}"] = df_mono[f"+A {phase}(KWh)"] - df_mono[f"evol_+A_{phase}"]
        df_mono.rename(columns={f"+A {phase}(KWh)": f"Nouvel index {phase}"}, inplace=True)
    df_mono["nb_bloque"] = df_mono[['Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque']].apply(
        lambda row: sum(cell != 'Normal' for cell in row), axis=1)
    df_mono = df_mono.sort_values(
        by=["nb_bloque", "evol_+A_L1", "evol_+A_L2", "evol_+A_L3"],
        ascending=[False, False, False, False]
    )
    f5_dict["Mono bloque"] = df_mono[[
        "Meter No.", "Partenaire", "Rue", 
        "Ancien index L1", "Nouvel index L1", "evol_+A_L1",
        "Ancien index L2", "Nouvel index L2", "evol_+A_L2",
        "Ancien index L3", "Nouvel index L3", "evol_+A_L3",
        "Anomalie"
    ]]

    # --- Mono faible ---
    df_faible = df[
        (df['Mono 1 faible'] == 'Mono 1 faible') |
        (df['Mono 2 faible'] == 'Mono 2 faible') |
        (df['Mono 3 faible'] == 'Mono 3 faible')
    ].copy()
    df_faible["Période"] = mois_string
    df_faible["Anomalie"] = df_faible[['Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible']].apply(
        lambda row: ' + '.join([v for v in row if v != 'Normal']), axis=1)
    for phase in ['L1', 'L2', 'L3']:
        df_faible[f"Ancien index {phase}"] = df_faible[f"+A {phase}(KWh)"] - df_faible[f"evol_+A_{phase}"]
        df_faible.rename(columns={f"+A {phase}(KWh)": f"Nouvel index {phase}"}, inplace=True)
    df_faible["nb_faible"] = df_faible[['Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible']].apply(
        lambda row: sum(cell != 'Normal' for cell in row), axis=1)
    df_faible = df_faible.sort_values(
        by=["nb_faible", "evol_+A_L1", "evol_+A_L2", "evol_+A_L3"],
        ascending=[False, False, False, False]
    )
    f5_dict["Mono faible"] = df_faible[[
        "Meter No.", "Partenaire", "Rue", 
        "Ancien index L1", "Nouvel index L1", "evol_+A_L1", "pct_evol_L1",
        "Ancien index L2", "Nouvel index L2", "evol_+A_L2", "pct_evol_L2",
        "Ancien index L3", "Nouvel index L3", "evol_+A_L3", "pct_evol_L3",
        "Anomalie"
    ]]

    # --- Surcompensation ---
    df_surcomp = df[df["Surcompensation"] == "Surcompensation"].copy()
    df_surcomp["Période"] = mois_string
    df_surcomp["Anomalie"] = "Surcompensation"
    df_surcomp = df_surcomp.sort_values(by="rapport_R", ascending=True)
    f5_dict["Surcompensation"] = df_surcomp[[
        "Meter No.", "Partenaire", "Rue", "+R(kvarh)", "-R(kvarh)", "rapport_R", "Anomalie"
    ]]

    # --- Résumé des anomalies ---
    df_anomaly_counts = count_anomalies(df)
    f5_dict["Nombre_anomalies"] = df_anomaly_counts

    # Assurer que toutes les clés existent même si vides
    for key in ["K2 bloque", "Inversion", "Mono bloque", "Mono faible", "Surcompensation"]:
        if key not in f5_dict:
            f5_dict[key] = pd.DataFrame()

    return f5_dict

def prepare_analysis_report(df_anomalies):
    anomalies_cols = [
        'K2 bloque', 'Inversion',
        'Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque',
        'Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible',
        'Surcompensation'
    ]
    
    df_filtered = df_anomalies[df_anomalies[anomalies_cols].apply(
        lambda row: any(val not in ["Normal", ""] for val in row), axis=1
    )]

    return df_filtered[[
        'Meter No.', 'Partenaire', 'Rue', 'evol_+A_T2', 'evol_-A',
        'pct_evol_L1', 'pct_evol_L2', 'pct_evol_L3',
        'rapport_R', 'K2 bloque', 'Inversion',
        'Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque',
        'Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible',
        'Surcompensation'
    ]]

def count_anomalies(df_anomalies: pd.DataFrame) -> pd.DataFrame:
    anomaly_columns = [
        'K2 bloque',
        'Inversion',
        'Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque',
        'Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible',
        'Surcompensation'
    ]
    
    anomaly_counts = {}
    for anomaly in anomaly_columns:
        if anomaly in df_anomalies.columns:
            count = df_anomalies[anomaly].eq(anomaly).sum()
            anomaly_counts[anomaly] = count
    
    df_counts = pd.DataFrame.from_dict(anomaly_counts, orient='index', columns=['Nombre']).reset_index()
    df_counts.columns = ['Type d\'anomalie', 'Nombre']
    
    total = df_counts['Nombre'].sum()
    df_counts = pd.concat([
        df_counts,
        pd.DataFrame([{'Type d\'anomalie': 'Total', 'Nombre': total}])
    ], ignore_index=True)
    
    return df_counts