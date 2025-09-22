import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.losses import MeanSquaredError
from typing import List, Tuple
import logging
import io

logger = logging.getLogger(__name__)

# Configuration identique à celle fournie
CONFIG = {
    'time_steps': 24,
    'feature_columns': [
        'sum_kW', 'mean_kW', 'std_kW', 'max_kW', 'min_kW',
        'hour_mean', 'day_of_week', 'month',
        'Saison_Sèche', 'Saison_Pluvieuse',
        'Climat_Chaud', 'Climat_Froid', 'Climat_Modérée'
    ],
    'saison_categories': ['Sèche', 'Pluvieuse'],
    'climat_categories': ['Chaud', 'Froid', 'Modérée']
}

def inspect_columns(df, file_name):
    logger.info(f"Colonnes du fichier {file_name}: {df.columns.tolist()}")

def categoriser_partenaire(partenaire):
    partenaire = str(partenaire).lower()
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

def export_low_consumption_per_step(df_step, mois_courant):
    logger.info(f"Export des compteurs <5kW pour {mois_courant}")
    df_step['year_month'] = df_step['Data Time'].dt.to_period('M')
    df_month = df_step[df_step['year_month'] == mois_courant]
    if df_month.empty:
        logger.warning(f"Aucun relevé pour {mois_courant}")
        return pd.DataFrame()
    
    def all_below_5kw(group):
        if group['Active power (+) total(kW)'].max() < 5:
            return group
        return pd.DataFrame()
    
    low_consumption = df_month.groupby('Meter No.', group_keys=False).apply(all_below_5kw)
    if low_consumption.empty:
        logger.info(f"Aucun compteur avec toutes les mesures <5kW pour {mois_courant}")
        return pd.DataFrame()
    
    colonnes_info = ['Meter No.', 'Rue', 'Categorie', 'Partenaire', 'Saison', 'Climat', 'year_month']
    low_consumption_unique = low_consumption[colonnes_info].drop_duplicates().reset_index(drop=True)
    return low_consumption_unique

def clean_group(group):
    logger.info(f"Nettoyage du groupe pour compteur {group['Meter No.'].iloc[0]}")
    group = group.sort_values('Data Time')
    group['Active power (+) total(kW)'] = pd.to_numeric(group['Active power (+) total(kW)'], errors='coerce')
    group['Active power (+) total(kW)'] = group['Active power (+) total(kW)'].interpolate(method='linear', limit=4)
    group['Active power (+) total(kW)'] = group['Active power (+) total(kW)'].ffill().bfill()
    group['is_weekend'] = group['day_of_week'] >= 5

    def detect_outliers(sub):
        mean = sub['Active power (+) total(kW)'].mean()
        std = sub['Active power (+) total(kW)'].std()
        sub['is_outlier'] = False
        sub.loc[sub['Active power (+) total(kW)'] < mean, 'is_outlier'] = (mean - sub['Active power (+) total(kW)']) > 5 * std
        sub.loc[sub['Active power (+) total(kW)'] > mean, 'is_outlier'] = (sub['Active power (+) total(kW)'] - mean) > 8 * std
        sub['Motif'] = np.where(
            sub['is_outlier'],
            np.where(sub['Active power (+) total(kW)'] < mean, 'Consommation faible par rapport à la moyenne', 'Consommation élevée par rapport à la moyenne'),
            ''
        )
        return sub

    group = group.groupby(['hour', 'is_weekend', 'Saison', 'Climat'], group_keys=False).apply(detect_outliers)
    return group

def interpolate_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('Data Time')
    group['Active power (+) total(kW)'] = group['Active power (+) total(kW)'].interpolate(method='linear', limit=4).ffill().bfill()
    return group

def load_and_preprocess(df_filtre: pd.DataFrame) -> pd.DataFrame:
    logger.info("Pré-traitement des données pour l'autoencodeur")
    df = df_filtre.groupby('Meter No.', group_keys=False).apply(interpolate_group)

    df_daily = df.groupby(['Meter No.', df['Data Time'].dt.date]).agg({
        'Active power (+) total(kW)': ['sum', 'mean', 'std', 'max', 'min'],
        'hour': 'mean', 'day_of_week': 'first', 'month': 'first',
        'Saison': 'first', 'Climat': 'first',
        'Categorie': 'first', 'Partenaire': 'first', 'Rue': 'first'
    }).reset_index()

    df_daily.columns = ['Meter No.', 'date', 'sum_kW', 'mean_kW', 'std_kW', 'max_kW', 'min_kW',
                        'hour_mean', 'day_of_week', 'month', 'Saison', 'Climat', 'Categorie', 'Partenaire', 'Rue']
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['std_kW'] = df_daily['std_kW'].fillna(0)

    df_daily['Saison_original'] = df_daily['Saison']
    df_daily['Climat_original'] = df_daily['Climat']
    df_daily['Partenaire_original'] = df_daily['Partenaire']
    df_daily['Rue_original'] = df_daily['Rue']

    df_daily['Saison'] = pd.Categorical(df_daily['Saison'], categories=CONFIG['saison_categories'])
    df_daily['Climat'] = pd.Categorical(df_daily['Climat'], categories=CONFIG['climat_categories'])
    df_daily = pd.get_dummies(df_daily, columns=['Saison', 'Climat'], prefix=['Saison', 'Climat'], dtype=int)

    for col in CONFIG['feature_columns']:
        if col not in df_daily.columns:
            df_daily[col] = 0

    return df_daily

def create_sequences_keras(df: pd.DataFrame, time_steps: int, feature_columns: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    sequences, sequence_info = [], []
    for meter in df['Meter No.'].unique():
        df_meter = df[df['Meter No.'] == meter].sort_values('date')
        X_meter = df_meter[feature_columns].values
        if len(X_meter) < time_steps: continue
        for i in range(0, len(X_meter) - time_steps + 1):
            sequences.append(X_meter[i:i + time_steps])
            sequence_info.append({
                'Meter No.': meter,
                'Start Date': df_meter['date'].iloc[i],
                'End Date': df_meter['date'].iloc[i + time_steps - 1],
                'Saison': df_meter['Saison_original'].iloc[i],
                'Climat': df_meter['Climat_original'].iloc[i],
                'Categorie': df_meter['Categorie'].iloc[i],
                'Partenaire': df_meter['Partenaire_original'].iloc[i],
                'Rue': df_meter['Rue_original'].iloc[i]
            })
    return np.array(sequences), pd.DataFrame(sequence_info)

def detect_additional_anomalies(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    logger.info("Détection d'anomalies métier supplémentaires")
    df_cleaned['Data Time'] = pd.to_datetime(df_cleaned['Data Time'])
    year_month = df_cleaned['Data Time'].dt.to_period('M').unique()[0]
    df_cleaned = df_cleaned[(df_cleaned['Data Time'].dt.month == year_month.month) & (df_cleaned['Data Time'].dt.year == year_month.year)]

    # Critère 1: Consommation max <0.5kW
    df_monthly = df_cleaned.groupby('Meter No.').agg({'Active power (+) total(kW)': 'max'}).reset_index()
    low_consumption = df_monthly[df_monthly['Active power (+) total(kW)'] < 0.5]['Meter No.'].unique()

    # Critère 2: Suspicion de fraude
    fraud_meters = []
    for meter in df_cleaned['Meter No.'].unique():
        df_meter = df_cleaned[df_cleaned['Meter No.'] == meter].sort_values('Data Time')
        values = df_meter['Active power (+) total(kW)'].values
        count, seq = 0, []
        for i, v in enumerate(values):
            if v < 0.5:
                count += 1
                seq.append(i)
                if count >= 20:  # 5h
                    other_idx = [j for j in range(len(values)) if j not in seq]
                    if len(other_idx) > 0 and (values[other_idx] > 17).all():
                        fraud_meters.append(meter)
                        break
            else:
                count, seq = 0, []

    additional = set(low_consumption).union(set(fraud_meters))
    anomalies = []
    for meter in additional:
        reason = "Low Consumption <0.5kW" if meter in low_consumption else "Suspected Fraud (≥5h<0.5kW & elsewhere >17kW)"
        row = df_cleaned[df_cleaned['Meter No.'] == meter].iloc[0]
        anomalies.append({
            'Meter No.': meter, 'Partenaire': row['Partenaire'], 'Rue': row['Rue'],
            'Saison': row['Saison'], 'Climat': row['Climat'], 'Categorie': row['Categorie'],
            'Anomaly Type': reason, 'MSE Score': None, 'Threshold': None
        })
    return pd.DataFrame(anomalies)

def detect_anomalies_autoencoder(model, X, info: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    logger.info("Détection d'anomalies avec l'autoencodeur")
    recon = model.predict(X)
    mse = np.mean(np.square(X - recon), axis=(1, 2))
    th = threshold * np.mean(mse)
    mask = mse > th
    df_out = info[mask].copy()
    df_out['MSE Score'] = mse[mask]
    df_out['Threshold'] = th
    df_out['Anomaly Type'] = "Autoencoder Anomaly"
    df_out = df_out[['Meter No.', 'Partenaire', 'Rue', 'Saison', 'Climat', 'Categorie', 'Anomaly Type', 'MSE Score', 'Threshold']]
    return df_out

def process_anomaly_detection(load_profile_content: bytes, clients_content: bytes, models_dir: str) -> dict:
    logger.info("Début du traitement des fichiers pour détection d'anomalies")

    # Lecture des fichiers uploadés
    df_load = pd.read_csv(io.BytesIO(load_profile_content), sep=';', encoding='utf-8-sig', on_bad_lines='skip', low_memory=False)
    df_clients = pd.read_excel(io.BytesIO(clients_content))
    inspect_columns(df_load, "Courbe de charge")
    inspect_columns(df_clients, "Liste clients")

    required_columns = ["METER_NO", "FREEZE_DATE", "P8040"]
    for col in required_columns:
        if col not in df_load.columns:
            raise KeyError(f"Colonne manquante dans le CSV : {col}")

    df_cleaned = df_load.loc[:, required_columns].copy()

    df_clients = df_clients.rename(columns={'Numero de serie (Numero compteur)': 'METER_NO'})
    df_clients["METER_NO"] = df_clients["METER_NO"].astype(str).str.strip().str.lstrip("0")
    df_cleaned["METER_NO"] = df_cleaned["METER_NO"].astype(str).str.strip()

    df = df_cleaned.merge(df_clients[['METER_NO', 'Partenaire', 'Rue']], on='METER_NO', how='inner')
    df['Categorie'] = df['Partenaire'].fillna('').apply(categoriser_partenaire)

    df = df.rename(columns={
        'METER_NO': 'Meter No.',
        'FREEZE_DATE': 'Data Time',
        'P8040': 'Active power (+) total(kW)'
    })

    df['Data Time'] = pd.to_datetime(df['Data Time'], format='%d-%b-%y %I.%M.%S.%f %p', errors='coerce')
    df['Active power (+) total(kW)'] = df['Active power (+) total(kW)'].astype(str).str.replace(',', '.', regex=False)
    df['Active power (+) total(kW)'] = pd.to_numeric(df['Active power (+) total(kW)'], errors='coerce')

    df['hour'] = df['Data Time'].dt.hour
    df['day_of_week'] = df['Data Time'].dt.weekday
    df['month'] = df['Data Time'].dt.month
    df['Saison'] = df['month'].apply(lambda x: 'Sèche' if x in [11,12,1,2,3,4,5] else 'Pluvieuse')
    df['Climat'] = df['month'].apply(lambda x: 'Chaud' if x in [2,3,4,5] else 'Froid' if x in [12,1] else 'Modérée')
    df['year_month'] = df['Data Time'].dt.to_period('M')
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    df = df.sort_values(by=["Meter No.", "Data Time"]).reset_index(drop=True)
    df_filtre = df[df['Active power (+) total(kW)'].notna()].reset_index(drop=True)

    # Extraction des compteurs <5kW
    mois_courants = df_filtre['year_month'].unique()
    low_consumption_dfs = []
    for mois in mois_courants:
        low_consumption_df = export_low_consumption_per_step(df_filtre, mois)
        if not low_consumption_df.empty:
            low_consumption_dfs.append(low_consumption_df)

    low_consumption_combined = pd.concat(low_consumption_dfs).drop_duplicates().reset_index(drop=True) if low_consumption_dfs else pd.DataFrame()

    # Suppression des compteurs <5kW
    if not low_consumption_combined.empty:
        compteurs_low_consumption = low_consumption_combined['Meter No.'].unique()
        logger.info(f"Suppression de {len(compteurs_low_consumption)} compteurs <5kW")
        df_filtre = df_filtre[~df_filtre['Meter No.'].isin(compteurs_low_consumption)].reset_index(drop=True)

    # Détection des outliers
    df_with_outliers = df_filtre.groupby('Meter No.', group_keys=False).apply(clean_group).reset_index(drop=True)
    df_outliers = df_with_outliers[df_with_outliers['is_outlier'] == True].copy()
    df_outliers = df_outliers[['Meter No.', 'Data Time', 'Active power (+) total(kW)', 'Rue', 'Categorie', 'Partenaire', 'Saison', 'Climat', 'year_month', 'hour', 'is_weekend', 'Motif']]

    # Prétraitement pour autoencodeur (génération de preprocessed_data_raw en mémoire)
    df_test = load_and_preprocess(df_filtre)

    # Chargement du scaler et normalisation (génération de preprocessed_data en mémoire)
    scaler_path = os.path.join(models_dir, "scaler_global.pkl")
    scaler = joblib.load(scaler_path)
    df_test[CONFIG['feature_columns']] = scaler.transform(df_test[CONFIG['feature_columns']])

    # Création des séquences
    X_test_seq, sequence_info = create_sequences_keras(df_test, CONFIG['time_steps'], CONFIG['feature_columns'])

    # Chargement du modèle
    model_path = os.path.join(models_dir, "autoencoder_keras.h5")
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    # Détection anomalies autoencodeur
    anomalies_auto = detect_anomalies_autoencoder(model, X_test_seq, sequence_info)

    # Anomalies supplémentaires
    additional_anomalies = detect_additional_anomalies(df_filtre)

    # Fusion
    combined_anomalies = pd.concat([anomalies_auto, additional_anomalies], ignore_index=True)

    # Mois pour le nommage
    month = df_filtre['year_month'].unique()[0].strftime('%Y-%m')

    logger.info("Fin du traitement des fichiers pour détection d'anomalies")
    return {
        "cleaned": df_filtre,
        "low_consumption": low_consumption_combined,
        "outliers": df_outliers,
        "anomalies": combined_anomalies,
        "month": month
    }