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

    df['rapport_R'] = np.where(df['-R(kvarh)'] == 0, 11, 
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