from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import os
import io
import tempfile
import atexit
from uuid import uuid4 
import logging
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import time
from typing import List
from openpyxl import load_workbook
# Ajouter ces lignes
from keras.models import load_model
from keras.metrics import MeanSquaredError
import tensorflow as tf

logger = logging.getLogger("uvicorn.error")


# main.py
from backend.auth import (
    pwd_context,
    hash_password,
    verify_password,
    UserDB,
    change_password,
    reset_password
)


# --- Modules personnalisés ---

from backend.utils import (
    categoriser_partenaire,
    export_low_consumption,
    interpolate_group,
    detect_outliers_group,
    prepare_daily_data,
    evaluate_autoencoder,
    remove_non_consuming_meters,
    FEATURE_COLUMNS,
    METER_NUMBERS,
    clean_and_preprocess,
    calculate_evolutions,
    detect_anomalies,
    generate_report,
    prepare_analysis_report,
    count_anomalies,
    create_sequences_keras_v2,
    load_and_clean,
    format_month_french,
    mois_fr
)

# Configuration identique à Kaggle
CONFIG = {
    'time_steps': 24,
    'hidden_dim': 50,
    'batch_size': 64,
    'lr': 0.001,
    'patience': 25,
    'feature_columns': FEATURE_COLUMNS
}

from backend.auth import (
    authenticate_user, create_user, delete_user,require_auth,
    list_visible_users, get_current_user,User
)
from backend.database import get_db

# --- Application ---
app = FastAPI()
app.add_middleware(
    SessionMiddleware, 
    secret_key="SENELEC_SECRET_KEY",
    max_age=3600,  # 1 heure
    same_site="lax",
    https_only=False  # Mettez True en production
)

# --- Répertoires ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Charger le scaler global à l'initialisation de l'application
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_global.pkl")
scaler = joblib.load(SCALER_PATH)

# --- Caches ---
results_cache = {}
session_store = {}

# ===========================
# AUTHENTIFICATION
# ===========================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Identifiants incorrects"})
    request.session.clear()
    request.session["user"] = {"username": user.username, "role": user.role}
    return RedirectResponse(url="/accueil", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ===========================
# PAGE ACCUEIL
# ===========================
# Redirection automatique de / vers /accueil
@app.get("/")
def root():
    return RedirectResponse(url="/accueil")

@app.get("/accueil", response_class=HTMLResponse)
async def accueil(request: Request, current_user: User = Depends(require_auth)):
    session_id = request.cookies.get("session_id")
    data = session_store.get(session_id)

    context = {"request": request, "files_ready": False, "current_user": current_user}

    if data and "accueil" in data:
        context.update({
            "files_ready": True,
            "month": data["accueil"]["month"],
            "cleaned_preview": data["accueil"]["cleaned_preview"],
            "anomalies_preview": data["accueil"]["anomalies_preview"],
            "analysis_preview": data["accueil"]["analysis_preview"],
            "non_evolving_preview": data["accueil"]["non_evolving_preview"],
            "has_full_report": True
        })

    return templates.TemplateResponse("index1.html", context)

@app.post("/accueil", response_class=HTMLResponse)
async def process_files_accueil(
    request: Request,
    current_user: User = Depends(require_auth),
    file_current: UploadFile = File(...),
    file_previous: UploadFile = File(...),
    file_list: UploadFile = File(...)
):
    session_id = request.cookies.get("session_id") or str(uuid4())

    df_current = pd.read_excel(io.BytesIO(await file_current.read()), skiprows=2)
    df_previous = pd.read_excel(io.BytesIO(await file_previous.read()), skiprows=2)
    df_list = pd.read_excel(io.BytesIO(await file_list.read()))

    df_cleaned = clean_and_preprocess(df_current, df_previous, df_list)
    df_processed, anomalies_non_evol, mois_string = calculate_evolutions(df_cleaned)
    df_anomalies = detect_anomalies(df_processed)
    detailed_report = generate_report(df_anomalies, mois_string)
    analysis_report = prepare_analysis_report(df_anomalies)

    cleaned_preview = df_cleaned.head(10).to_html(classes="table", index=False)
    anomalies_preview = df_anomalies.head(10).to_html(classes="table", index=False)
    analysis_preview = analysis_report.head(10).to_html(classes="table", index=False)
    non_evolving_preview = anomalies_non_evol.head(10).to_html(classes="table", index=False) if not anomalies_non_evol.empty else ""
    anomaly_counts_preview = detailed_report["Nombre_anomalies"].head(10).to_html(classes="table", index=False)

    results_cache.update({
        "cleaned": df_cleaned,
        "non_evolving": anomalies_non_evol,
        "anomalies": df_anomalies,
        "analysis": analysis_report,
        "detailed": detailed_report,
        "month": mois_string
    })

    session_store[session_id] = session_store.get(session_id, {})
    session_store[session_id]["accueil"] = {
        "month": mois_string,
        "cleaned_preview": cleaned_preview,
        "anomalies_preview": anomalies_preview,
        "analysis_preview": analysis_preview,
        "non_evolving_preview": non_evolving_preview,
        "anomaly_counts_preview": anomaly_counts_preview
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
            df_cleaned.to_excel(writer, sheet_name="Données traitees", index=False)
            if not anomalies_non_evol.empty:
                anomalies_non_evol.to_excel(writer, sheet_name="Compteurs non consommants", index=False)
            analysis_report.to_excel(writer, sheet_name="Rapport d'anomalies", index=False)
            for sheet_name, df in results_cache["detailed"].items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        results_cache["full_report_path"] = tmp.name

    response = RedirectResponse(url="/accueil", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/reset-accueil")
async def reset_accueil(request: Request, current_user: User = Depends(require_auth)):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("accueil", None)
    return RedirectResponse(url="/accueil", status_code=303)
# ===========================
# PAGE ANALYSE
# ===========================

@app.get("/analyse", response_class=HTMLResponse)
async def analyse_page(request: Request, current_user: User = Depends(require_auth)):
    session_id = request.cookies.get("session_id")
    data = session_store.get(session_id)

    context = {"request": request, "files_ready": False, "current_user": current_user}

    if data and "analyse" in data:
        context.update({
            "files_ready": True,
            "month": data["analyse"]["month"],
            "cleaned_preview": data["analyse"]["cleaned_preview"],
            "anomalies_preview": data["analyse"]["anomalies_preview"],
            "analysis_preview": data["analyse"]["analysis_preview"],
            "non_evolving_preview": data["analyse"]["non_evolving_preview"],
            "has_full_report": True
        })

    return templates.TemplateResponse("index.html", context)

@app.post("/analyse", response_class=HTMLResponse)
async def process_files_analyse(
    request: Request,
    current_user: User = Depends(require_auth),
    file_current: UploadFile = File(...),
    file_previous: UploadFile = File(...),
    file_list: UploadFile = File(...)
):
    user = get_current_user(request)
    session_id = request.cookies.get("session_id") or str(uuid4())

    df_current = pd.read_excel(io.BytesIO(await file_current.read()), skiprows=2)
    df_previous = pd.read_excel(io.BytesIO(await file_previous.read()), skiprows=2)
    df_list = pd.read_excel(io.BytesIO(await file_list.read()))

    df_cleaned = clean_and_preprocess(df_current, df_previous, df_list)
    df_processed, anomalies_non_evol, mois_string = calculate_evolutions(df_cleaned)
    df_anomalies = detect_anomalies(df_processed)
    detailed_report = generate_report(df_anomalies, mois_string)
    analysis_report = prepare_analysis_report(df_anomalies)

    cleaned_preview = df_cleaned.head(10).to_html(classes="table", index=False)
    anomalies_preview = df_anomalies.head(10).to_html(classes="table", index=False)
    analysis_preview = analysis_report.head(10).to_html(classes="table", index=False)
    non_evolving_preview = anomalies_non_evol.head(10).to_html(classes="table", index=False) if not anomalies_non_evol.empty else ""
    anomaly_counts_preview = detailed_report["Nombre_anomalies"].head(10).to_html(classes="table", index=False)

    results_cache.update({
        "cleaned": df_cleaned,
        "non_evolving": anomalies_non_evol,
        "anomalies": df_anomalies,
        "analysis": analysis_report,
        "detailed": detailed_report,
        "month": mois_string
    })

    session_store[session_id] = session_store.get(session_id, {})
    session_store[session_id]["analyse"] = {
        "month": mois_string,
        "cleaned_preview": cleaned_preview,
        "anomalies_preview": anomalies_preview,
        "analysis_preview": analysis_preview,
        "non_evolving_preview": non_evolving_preview,
        "anomaly_counts_preview": anomaly_counts_preview
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
            df_cleaned.to_excel(writer, sheet_name="Données traitees", index=False)
            if not anomalies_non_evol.empty:
                anomalies_non_evol.to_excel(writer, sheet_name="Compteurs non consommants", index=False)
            analysis_report.to_excel(writer, sheet_name="Rapport d'anomalies", index=False)
            for sheet_name, df in results_cache["detailed"].items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        results_cache["full_report_path"] = tmp.name

    response = RedirectResponse(url="/analyse", status_code=303)
    response.set_cookie("session_id", session_id)
    return response

@app.post("/reset-analyse")
async def reset_analyse(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("analyse", None)
    return RedirectResponse(url="/analyse", status_code=303)

#------dashboard 
# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # Gère np.nan, pd.NA, etc.
            return None
        elif isinstance(obj, pd.Timestamp):  # Gère les Timestamp
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(require_auth)):
    session_id = request.cookies.get("session_id")
    data = session_store.get(session_id)

    context = {"request": request, "files_ready": False, "current_user": current_user}

    # Vérification indépendante pour l'analyse
    if data and "analyse" in data and results_cache:
        try:
            df_anomalies = results_cache.get("anomalies")
            df_cleaned = results_cache.get("cleaned")
            df_non_evolving = results_cache.get("non_evolving")
            detailed_report = results_cache.get("detailed")

            if df_anomalies is not None and df_cleaned is not None and df_non_evolving is not None and detailed_report is not None:
                # Supprimer la ligne "Total" dans Nombre_anomalies
                df_nombre_anomalies = detailed_report["Nombre_anomalies"]
                if "Type d'anomalie" in df_nombre_anomalies.columns:
                    df_nombre_anomalies = df_nombre_anomalies[df_nombre_anomalies["Type d'anomalie"] != "Total"].copy()
                else:
                    context["error"] = "La colonne 'Type d'anomalie' est absente dans Nombre_anomalies."
                    return templates.TemplateResponse("dashboard.html", context)

                nb_consommants = len(df_cleaned)
                nb_non_consommants = len(df_non_evolving)

                anomalies_cols = [
                    'K2 bloque', 'Inversion', 'Mono 1 bloque', 'Mono 2 bloque', 'Mono 3 bloque',
                    'Mono 1 faible', 'Mono 2 faible', 'Mono 3 faible', 'Surcompensation'
                ]
                df_anomalies['nb_anomalies'] = df_anomalies[anomalies_cols].apply(
                    lambda row: sum(1 for val in row if val != 'Normal'), axis=1
                )
                anomalies_par_nb = df_anomalies['nb_anomalies'].value_counts().to_dict()
                total_avec_anomalies = sum(anomalies_par_nb.values())
                nb_sans_anomalie = nb_consommants - total_avec_anomalies
                anomalies_par_nb[0] = nb_sans_anomalie
                cas_normaux = nb_sans_anomalie
                cas_avec_anomalies = total_avec_anomalies

                nombre_anomalies = json.dumps(df_nombre_anomalies.to_dict("records"), cls=NumpyEncoder)

            # Mettre à jour le contexte pour l'analyse
            context.update({
                "analyse_ready": True,
                "month": data["analyse"]["month"],
                "nb_sans_anomalie": nb_sans_anomalie if 'nb_sans_anomalie' in locals() else 0,
                "nb_consommants": nb_consommants if 'nb_consommants' in locals() else 0,
                "nb_non_consommants": nb_non_consommants if 'nb_non_consommants' in locals() else 0,
                "cas_normaux": cas_normaux if 'cas_normaux' in locals() else 0,
                "cas_avec_anomalies": cas_avec_anomalies if 'cas_avec_anomalies' in locals() else 0,
                "anomalies_par_nb": anomalies_par_nb if 'anomalies_par_nb' in locals() else {},
                "nombre_anomalies": nombre_anomalies if 'nombre_anomalies' in locals() else json.dumps([]),
            })

        except Exception as e:
            context["error"] = f"Erreur lors du traitement des données d'analyse : {str(e)}"
            print("Erreur dans dashboard (analyse):", str(e))

    # Vérification indépendante pour la détection d'anomalies
    if data and "anomaly_detection" in data and results_cache:
        try:
            df_anomaly_cleaned = results_cache.get("anomaly_cleaned")
            df_anomaly_anomalies = results_cache.get("anomaly_anomalies")
            df_anomaly_low_consumption = results_cache.get("anomaly_low_consumption")
            df_anomaly_outliers = results_cache.get("anomaly_outliers")
            anomaly_month = results_cache.get("anomaly_month", "")

            if df_anomaly_cleaned is not None and df_anomaly_anomalies is not None and df_anomaly_low_consumption is not None and df_anomaly_outliers is not None:
                cleaned_preview = df_anomaly_cleaned.head(10).to_html(classes="table", index=False)
                anomalies_preview = df_anomaly_anomalies.head(10).to_html(classes="table", index=False)
                low_consumption_preview = df_anomaly_low_consumption.head(10).to_html(classes="table", index=False)
                outliers_preview = df_anomaly_outliers.head(10).to_html(classes="table", index=False)

            # Mettre à jour le contexte pour la détection d'anomalies
            context.update({
                "detection_ready": True,
                "month": anomaly_month,
                "cleaned_preview": cleaned_preview if 'cleaned_preview' in locals() else "",
                "anomalies_preview": anomalies_preview if 'anomalies_preview' in locals() else "",
                "low_consumption_preview": low_consumption_preview if 'low_consumption_preview' in locals() else "",
                "outliers_preview": outliers_preview if 'outliers_preview' in locals() else "",
            })

        except Exception as e:
            context["error"] = f"Erreur lors du traitement des données de détection : {str(e)}"
            print("Erreur dans dashboard (détection):", str(e))

    return templates.TemplateResponse("dashboard.html", context)

# --- New Routes for Anomaly Detection ---
@app.get("/anomaly-detection", response_class=HTMLResponse)
async def anomaly_detection_page(request: Request, current_user: User = Depends(require_auth)):
    session_id = request.cookies.get("session_id")
    data = session_store.get(session_id)

    context = {"request": request, "files_ready": False, "current_user": current_user}

    if data and "anomaly_detection" in data:
        context.update({
            "files_ready": True,
            "month": data["anomaly_detection"]["month"],
            "cleaned_preview": data["anomaly_detection"]["cleaned_preview"],
            "anomalies_preview": data["anomaly_detection"]["anomalies_preview"],
            "low_consumption_preview": data["anomaly_detection"]["low_consumption_preview"],
            "outliers_preview": data["anomaly_detection"]["outliers_preview"]
        })

    return templates.TemplateResponse("anomaly_detection.html", context)

from fastapi import HTTPException
import pandas as pd
import os
import tempfile
import time
from openpyxl import load_workbook
from typing import List
import io

@app.post("/anomaly-detection", response_class=HTMLResponse)
async def process_anomaly_detection(
    request: Request,
    file_report: UploadFile = File(...),
    file_clients: UploadFile = File(...)
):
    user = get_current_user(request)
    session_id = request.cookies.get("session_id") or str(uuid4())

    # Vérification des types de fichiers
    if not (file_report.filename.endswith(('.csv', '.xlsx', '.xls'))):
        raise HTTPException(status_code=400, detail="Le fichier de rapport doit être au format CSV ou Excel")
    if not (file_clients.filename.endswith(('.csv', '.xlsx', '.xls'))):
        raise HTTPException(status_code=400, detail="Le fichier des clients doit être au format CSV ou Excel")

    # Lecture du fichier report (CSV ou Excel) par chunks
    start = time.time()
    logger.info(f"Début de la lecture du fichier report: {file_report.filename}")
    
    if file_report.filename.endswith('.csv'):
        file_content = io.BytesIO(await file_report.read())
        sample = pd.read_csv(file_content, nrows=1, sep=';', encoding='utf-8')
        available_cols = [col.strip() for col in sample.columns]
        expected_cols = ["METER_NO", "FREEZE_DATE", "P8040"]
        missing_cols = [col for col in expected_cols if col.upper() not in [c.upper() for c in available_cols]]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Les colonnes manquantes dans le fichier CSV : {', '.join(missing_cols)}")
        file_content.seek(0)
        chunks = pd.read_csv(
            file_content,
            usecols=[col for col in available_cols if col.upper() in [c.upper() for c in expected_cols]],
            dtype={'METER_NO': str},
            chunksize=10000,
            sep=';',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        df_chunks: List[pd.DataFrame] = []
        for chunk in chunks:
            if not chunk.empty:
                chunk["METER_NO"] = chunk["METER_NO"].str.strip().str.lstrip("0")
                df_chunks.append(chunk)
        df_report = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()
    else:
        excel_data = await file_report.read()
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_excel:
            tmp_excel.write(excel_data)
            tmp_excel_path = tmp_excel.name
        try:
            wb = load_workbook(tmp_excel_path)
            ws = wb.active
            df_chunks = []
            chunk_size = 10000
            rows = list(ws.rows)
            for i in range(0, len(rows), chunk_size):
                chunk_data = []
                for row in rows[i:i + chunk_size]:
                    values = [cell.value for cell in row[:3]]
                    if all(v is not None for v in values):
                        chunk_data.append(values)
                if chunk_data:
                    chunk_df = pd.DataFrame(chunk_data, columns=["METER_NO", "FREEZE_DATE", "P8040"])
                    chunk_df["METER_NO"] = chunk_df["METER_NO"].astype(str).str.strip().str.lstrip("0")
                    df_chunks.append(chunk_df)
            df_report = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()
        finally:
            os.unlink(tmp_excel_path)

    logger.info(f"Temps de lecture du fichier report: {time.time() - start} secondes")

    # Lecture du fichier clients (CSV ou Excel)
    start = time.time()
    logger.info(f"Début de la lecture du fichier clients: {file_clients.filename}")
    file_clients_content = await file_clients.read()
    
    if file_clients.filename.endswith('.csv'):
        chunks = pd.read_csv(
            io.BytesIO(file_clients_content),
            usecols=["Numero de serie (Numero compteur)", "Partenaire", "Rue"],
            dtype={'Numero de serie (Numero compteur)': str},
            chunksize=10000,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        df_chunks_clients: List[pd.DataFrame] = []
        for chunk in chunks:
            if not chunk.empty:
                chunk["Numero de serie (Numero compteur)"] = chunk["Numero de serie (Numero compteur)"].str.strip().str.lstrip("0")
                df_chunks_clients.append(chunk)
        df_clients = pd.concat(df_chunks_clients, ignore_index=True) if df_chunks_clients else pd.DataFrame()
    else:
        df_clients = pd.read_excel(
            io.BytesIO(file_clients_content),
            usecols=["Numero de serie (Numero compteur)", "Partenaire", "Rue"],
            engine='openpyxl',
            dtype={'Numero de serie (Numero compteur)': str}
        )
        df_clients["Numero de serie (Numero compteur)"] = df_clients["Numero de serie (Numero compteur)"].str.strip().str.lstrip("0")

    logger.info(f"Temps de lecture du fichier clients: {time.time() - start} secondes")

    # Vérification des DataFrames
    if df_report.empty or df_clients.empty:
        raise HTTPException(status_code=400, detail="Un des fichiers est vide ou mal formé")

    # Prétraitement
    df_clients = df_clients.rename(columns={'Numero de serie (Numero compteur)': 'METER_NO'})

    # Fusion
    df_list = []
    if not df_report.empty:
        for chunk in [df_report] if isinstance(df_report, pd.DataFrame) else df_chunks:
            merged_chunk = chunk.merge(df_clients[['METER_NO', 'Partenaire', 'Rue']], on='METER_NO', how='inner')
            merged_chunk['Categorie'] = merged_chunk['Partenaire'].fillna('').apply(categoriser_partenaire)
            df_list.append(merged_chunk)
    df = pd.concat(df_list, ignore_index=True)

    df = df.rename(columns={
        'METER_NO': 'Meter No.',
        'FREEZE_DATE': 'Data Time',
        'P8040': 'Active power (+) total(kW)'
    })
    
    # Conversion explicite en numérique
    df['Active power (+) total(kW)'] = pd.to_numeric(df['Active power (+) total(kW)'], errors='coerce')

    # Gestion des formats de date
    try:
        df['Data Time'] = pd.to_datetime(df['Data Time'], format='%d-%b-%y %I.%M.%S.%f %p', errors='coerce')
    except ValueError:
        df['Data Time'] = pd.to_datetime(df['Data Time'], errors='coerce')

    df = df.sort_values(by=["Meter No.", "Data Time"], ascending=[True, True]).reset_index(drop=True)

    # Ingénierie des fonctionnalités
    df['hour'] = df['Data Time'].dt.hour
    df['day_of_week'] = df['Data Time'].dt.weekday
    df['month'] = df['Data Time'].dt.month
    df['Saison'] = df['month'].apply(lambda x: 'Sèche' if x in [11, 12, 1, 2, 3, 4, 5] else 'Pluvieuse')
    df['Climat'] = df['month'].apply(lambda x: 'Chaud' if x in [2, 3, 4, 5] else 'Froid' if x in [12, 1] else 'Modérée')
    df['year_month'] = df['Data Time'].dt.to_period('M')

    df['Saison'] = df['Saison'].apply(lambda x: x if x in ['Sèche', 'Pluvieuse'] else 'Sèche')
    df['Climat'] = df['Climat'].apply(lambda x: x if x in ['Chaud', 'Froid', 'Modérée'] else 'Modérée')

    # Vérification des colonnes après prétraitement
    logger.info(f"Colonnes dans df après prétraitement : {df.columns.tolist()}")

    # Interpolation et détection
    df = remove_non_consuming_meters(df, energy_col="Active power (+) total(kW)")
    df_cleaned = df.groupby('Meter No.', group_keys=False).apply(interpolate_group)
    logger.info(f"Colonnes dans df_cleaned après interpolate_group : {df_cleaned.columns.tolist()}")
    
    mois_courant = df['year_month'].iloc[0] if not df.empty else None
    mois_courant_str = format_month_french(mois_courant) if mois_courant else ""

    df_low_consumption = export_low_consumption(df_cleaned, mois_courant, do_export=True) if mois_courant else pd.DataFrame()
    logger.info(f"Colonnes dans df_low_consumption : {df_low_consumption.columns.tolist()}")
    
    df_with_outliers = df_cleaned.groupby('Meter No.', group_keys=False).apply(detect_outliers_group)
    df_outliers = df_with_outliers[df_with_outliers['is_outlier']].copy() if not df_with_outliers.empty else pd.DataFrame()

    # Charger modèle et scaler global avant prepare_daily_data
    model_path = os.path.join(MODELS_DIR, 'autoencoder_keras.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler_global.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=500, detail="Modèle ou scaler manquant")

    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load(scaler_path)

    # Préparer les données quotidiennes
    df_daily = prepare_daily_data(df_cleaned) if not df_cleaned.empty else pd.DataFrame()
    logger.info(f"Colonnes dans df_daily après prepare_daily_data : {df_daily.columns.tolist()}")

    # Vérifier que toutes les colonnes de FEATURE_COLUMNS sont présentes
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_daily.columns]
    if missing_cols:
        logger.error(f"Colonnes manquantes dans df_daily : {missing_cols}")
        raise HTTPException(status_code=500, detail=f"Colonnes manquantes dans df_daily : {missing_cols}")

    # Détection d'anomalies
    # Détection d'anomalies
    anomalies_list = []
    if not df_daily.empty:
        METER_NUMBERS = df_daily['Meter No.'].unique().tolist()
        logger.info(f"Compteurs uniques dans df_daily : {len(METER_NUMBERS)}")

        for meter in METER_NUMBERS:
            df_meter = df_daily[df_daily['Meter No.'] == meter].copy()
            if df_meter.empty or len(df_meter) < CONFIG['time_steps']:  # Minimum time_steps jours pour une séquence
                logger.warning(f"Skipping meter {meter}: too few days ({len(df_meter)})")
                continue

            X_test = df_meter[FEATURE_COLUMNS].copy()
            # Ajouter du bruit pour éviter std=0
            for col in FEATURE_COLUMNS[:5]:
                if X_test[col].std() == 0:
                    X_test[col] += np.random.normal(0, 1e-6, size=X_test[col].shape)

            # Normaliser les données
            try:
                X_test_scaled = scaler.transform(X_test)
            except ValueError as e:
                logger.error(f"Erreur lors de la normalisation pour le compteur {meter}: {str(e)}")
                continue

            # Créer séquences (approche Kaggle)
            X_sequences = create_sequences_keras_v2(df_meter, CONFIG['time_steps'], FEATURE_COLUMNS)
            if X_sequences.shape[0] == 0:
                logger.warning(f"Pas de séquences créées pour le compteur {meter}")
                continue

            # Évaluer avec l'autoencodeur (approche Kaggle)
            try:
                test_error, labels = evaluate_autoencoder(model, X_sequences, df_meter, 
                                                        time_steps=CONFIG['time_steps'], 
                                                        threshold_multiplier=2.0)
                logger.info(f"Compteur {meter}: test_error shape={test_error.shape}, df_meter shape={len(df_meter)}")
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de l'autoencodeur pour le compteur {meter}: {str(e)}")
                continue

            if len(test_error) > 0:
                df_meter_anomalies = df_meter.copy()
                if len(test_error) == len(df_meter):
                    df_meter_anomalies['reconstruction_error'] = test_error
                    df_meter_anomalies['anomaly'] = labels
                else:
                    logger.warning(f"Longueur mismatch pour compteur {meter}: test_error({len(test_error)}) vs df_meter({len(df_meter)})")
                    padding_length = len(df_meter) - len(test_error)
                    if padding_length > 0:
                        df_meter_anomalies['reconstruction_error'] = np.pad(test_error, (0, padding_length), mode='constant', constant_values=np.nan)
                        df_meter_anomalies['anomaly'] = np.pad(labels, (0, padding_length), mode='constant', constant_values=0)
                    else:
                        df_meter_anomalies['reconstruction_error'] = test_error[:len(df_meter)]
                        df_meter_anomalies['anomaly'] = labels[:len(df_meter)]
                anomalies_list.append(df_meter_anomalies[df_meter_anomalies['anomaly'] == 1])
            else:
                logger.warning(f"Aucune erreur ou label valide pour le compteur {meter}")

    logger.info("Début de la concaténation des anomalies")
    start_concat = time.time()
    df_anomalies = pd.concat(anomalies_list, ignore_index=True) if anomalies_list else pd.DataFrame()
    logger.info(f"Temps de concaténation des anomalies: {time.time() - start_concat} secondes")
    
    if not df_anomalies.empty:
        logger.info("Début du tri et de la fusion des anomalies")
        start_sort = time.time()
        df_anomalies = df_anomalies.merge(
            df_anomalies.groupby('Meter No.')['mean_kW'].mean().rename('mean_kW_meter'),
            on='Meter No.'
        )
        df_anomalies.sort_values(by=['mean_kW_meter', 'Meter No.', 'date'], ascending=[False, True, True], inplace=True)
        df_anomalies.drop(columns='mean_kW_meter', inplace=True)
        logger.info(f"Temps de tri et fusion des anomalies: {time.time() - start_sort} secondes")

    # Générer les aperçus
    logger.info("Début de la génération des aperçus HTML")
    start_preview = time.time()
    cleaned_preview = df_cleaned.head(10).to_html(classes="table", index=False) if not df_cleaned.empty else ""
    anomalies_preview = df_anomalies.head(10).to_html(classes="table", index=False) if not df_anomalies.empty else ""
    low_consumption_preview = df_low_consumption.head(10).to_html(classes="table", index=False) if not df_low_consumption.empty else ""
    outliers_preview = df_outliers.head(10).to_html(classes="table", index=False) if not df_outliers.empty else ""
    logger.info(f"Temps de génération des aperçus HTML: {time.time() - start_preview} secondes")

    # Mettre à jour le cache
    logger.info("Début de la mise à jour du cache")
    results_cache.update({
        "anomaly_cleaned": df_cleaned,
        "anomaly_anomalies": df_anomalies,
        "anomaly_low_consumption": df_low_consumption,
        "anomaly_outliers": df_outliers,
        "anomaly_month": str(mois_courant) if mois_courant else ""
    })

    # Stocker les résultats dans la session
    logger.info("Début de la mise à jour de la session")
    session_store[session_id] = session_store.get(session_id, {})
    session_store[session_id]["anomaly_detection"] = {
        "month": format_month_french(mois_courant) if mois_courant else "",
        "cleaned_preview": cleaned_preview,
        "anomalies_preview": anomalies_preview,
        "low_consumption_preview": low_consumption_preview,
        "outliers_preview": outliers_preview
    }

    # Sauvegarder les résultats dans un fichier temporaire
    logger.info("Début de l'écriture du fichier Excel")
    start_excel = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
            if not df_cleaned.empty:
                df_cleaned.to_excel(writer, sheet_name="Données nettoyées", index=False)
            if not df_anomalies.empty:
                df_anomalies.to_excel(writer, sheet_name="Anomalies", index=False)
            if not df_low_consumption.empty:
                df_low_consumption.to_excel(writer, sheet_name="Compteurs <5kW", index=False)
            if not df_outliers.empty:
                df_outliers.to_excel(writer, sheet_name="Outliers", index=False)
        results_cache["anomaly_full_report_path"] = tmp.name
    logger.info(f"Temps d'écriture du fichier Excel: {time.time() - start_excel} secondes")

    logger.info(f"Temps total de traitement: {time.time() - start} secondes")
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/reset-anomaly-detection")
async def reset_anomaly_detection(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("anomaly_detection", None)
    return RedirectResponse(url="/dashboard", status_code=303)  # Rediriger vers /dashboard

@app.post("/reset-dashboard")
async def reset_dashboard(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("analyse", None)
        session_store[session_id].pop("anomaly_detection", None)
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/preview-anomaly/{file_type}")
async def preview_anomaly(file_type: str, limit: int = 10):
    file_key_map = {
        "cleaned": "anomaly_cleaned",
        "anomalies": "anomaly_anomalies",
        "low_consumption": "anomaly_low_consumption",
        "outliers": "anomaly_outliers"
    }
    if file_type not in file_key_map or file_key_map[file_type] not in results_cache:
        return {"error": "Aucun aperçu disponible"}
    df = results_cache[file_key_map[file_type]]
    return {"html": df.head(limit).to_html(classes="table", index=False)} if limit != -1 else {"html": df.to_html(classes="table", index=False)}

@app.get("/download-anomaly/{filekey}")
async def download_anomaly_file(filekey: str):
    file_key_map = {
        "cleaned": "anomaly_cleaned",
        "anomalies": "anomaly_anomalies",
        "low_consumption": "anomaly_low_consumption",
        "outliers": "anomaly_outliers"
    }
    if filekey not in file_key_map or file_key_map[filekey] not in results_cache:
        return HTMLResponse("Fichier non trouvé", status_code=404)
    
    df = results_cache[file_key_map[filekey]]
    file_bytes = df_to_excel_bytes(df)
    filename_map = {
        "cleaned": "donnees_nettoyees",
        "anomalies": "anomalies",
        "low_consumption": "compteurs_moins_5kw",
        "outliers": "outliers"
    }
    filename = f"{filename_map[filekey]}_{results_cache['anomaly_month']}.xlsx"
    
    return StreamingResponse(
        file_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# === FONCTIONS POUR LA VISUALISATION ===

def prepare_visualization_data():
    """Prépare les données pour la visualisation"""
    if not all(key in results_cache for key in ["anomaly_cleaned", "anomaly_anomalies", 
                                              "anomaly_low_consumption", "anomaly_outliers"]):
        return None
    
    df_cleaned = results_cache["anomaly_cleaned"]
    df_anomalies = results_cache["anomaly_anomalies"]
    df_low_consumption = results_cache["anomaly_low_consumption"]
    df_outliers = results_cache["anomaly_outliers"]
    
    # Compter les différents types de compteurs
    total_meters = df_cleaned['Meter No.'].nunique()
    anomaly_meters = df_anomalies['Meter No.'].nunique() if not df_anomalies.empty else 0
    low_consumption_meters = df_low_consumption['Meter No.'].nunique() if not df_low_consumption.empty else 0
    outlier_meters = df_outliers['Meter No.'].nunique() if not df_outliers.empty else 0
    
    # Calculer les compteurs sans anomalies
    normal_meters = total_meters - (anomaly_meters + low_consumption_meters + outlier_meters)
    
    # Séparer les outliers en haute et basse consommation
    if not df_outliers.empty:
        # Calculer la consommation moyenne par compteur
        avg_consumption = df_cleaned.groupby('Meter No.')['Active power (+) total(kW)'].mean()
        
        # Identifier les outliers haute et basse consommation
        high_outliers = []
        low_outliers = []
        
        for meter in df_outliers['Meter No.'].unique():
            meter_avg = avg_consumption.get(meter, 0)
            # Si la consommation moyenne est supérieure à la médiane, c'est un haut outlier
            if meter_avg > avg_consumption.median():
                high_outliers.append(meter)
            else:
                low_outliers.append(meter)
        
        high_outlier_count = len(high_outliers)
        low_outlier_count = len(low_outliers)
    else:
        high_outlier_count = 0
        low_outlier_count = 0
    
    # Préparer les données pour les graphiques
    visualization_data = {
        "counts": {
            "total_meters": total_meters,
            "normal_meters": normal_meters,
            "anomaly_meters": anomaly_meters,
            "low_consumption_meters": low_consumption_meters,
            "outlier_meters": outlier_meters,
            "high_outlier_count": high_outlier_count,
            "low_outlier_count": low_outlier_count
        },
        "month": results_cache.get("anomaly_month", "")
    }
    
    return visualization_data

def get_consumption_curves(meter_numbers, month_period):
    """Récupère les courbes de consommation pour les compteurs spécifiés"""
    if "anomaly_cleaned" not in results_cache:
        return None
    
    df_cleaned = results_cache["anomaly_cleaned"]
    
    # Filtrer pour le mois spécifié
    df_month = df_cleaned[df_cleaned['year_month'] == month_period]
    
    # Filtrer pour les compteurs spécifiés
    meter_numbers_str = [str(m) for m in meter_numbers]
    df_filtered = df_month[df_month['Meter No.'].isin(meter_numbers_str)]
    
    if df_filtered.empty:
        return None
    
    # Agrégation quotidienne
    df_daily = df_filtered.groupby(['Meter No.', df_filtered['Data Time'].dt.date]).agg({
        'Active power (+) total(kW)': 'sum'
    }).reset_index()
    df_daily.columns = ['Meter No.', 'date', 'sum_kW']
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Calcul consommation moyenne par compteur
    df_mean = df_daily.groupby('Meter No.')['sum_kW'].mean().reset_index()
    df_mean.columns = ['Meter No.', 'mean_consumption']
    
    # Classification en 3 groupes (faible, moyenne, forte)
    try:
        df_mean['group'] = pd.qcut(
            df_mean['mean_consumption'], q=3, labels=['faible', 'moyenne', 'forte']
        )
    except:
        # Si pas assez de données pour 3 quantiles, utiliser des seuils fixes
        thresholds = [0, df_mean['mean_consumption'].quantile(0.33), 
                     df_mean['mean_consumption'].quantile(0.66), df_mean['mean_consumption'].max()]
        if thresholds[1] == thresholds[2]:  # Cas où les quantiles sont identiques
            df_mean['group'] = 'moyenne'
        else:
            df_mean['group'] = pd.cut(
                df_mean['mean_consumption'], 
                bins=thresholds, 
                labels=['faible', 'moyenne', 'forte'],
                include_lowest=True
            )
    
    df_daily = df_daily.merge(df_mean[['Meter No.', 'group']], on='Meter No.')
    
    # Préparer les données pour le frontend
    curves_data = {}
    for group in ['faible', 'moyenne', 'forte']:
        group_meters = df_daily[df_daily['group'] == group]['Meter No.'].unique()
        group_data = {}
        
        for meter in group_meters:
            df_meter = df_daily[df_daily['Meter No.'] == meter].sort_values('date')
            group_data[str(meter)] = {
                'dates': df_meter['date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': df_meter['sum_kW'].tolist()
            }
        
        curves_data[group] = group_data
    
    return curves_data

# === NOUVELLES ROUTES ===

@app.get("/anomaly-visualization-data")
async def get_anomaly_visualization_data():
    """Endpoint pour récupérer les données de visualisation"""
    viz_data = prepare_visualization_data()
    if not viz_data:
        return {"error": "Aucune donnée disponible pour la visualisation"}
    
    return viz_data

@app.get("/anomaly-consumption-curves")
async def get_anomaly_consumption_curves():
    """Endpoint pour récupérer les courbes de consommation des compteurs avec anomalies"""
    if "anomaly_anomalies" not in results_cache or results_cache["anomaly_anomalies"].empty:
        return {"error": "Aucune anomalie détectée"}
    
    # Récupérer les compteurs avec anomalies
    anomaly_meters = results_cache["anomaly_anomalies"]['Meter No.'].unique().tolist()
    month_period = results_cache.get("anomaly_month", "")
    
    curves_data = get_consumption_curves(anomaly_meters, month_period)
    if not curves_data:
        return {"error": "Impossible de générer les courbes de consommation"}
    
    return curves_data

# GESTION UTILISATEURS
# ===========================

@app.get("/users")
async def users_page(
    request: Request,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    users = list_visible_users(db, current_user)
    return templates.TemplateResponse("users.html", {
        "request": request,
        "users": users,
        "current_user": current_user
    })

@app.post("/users/create")
def create_new_user(
    request: Request,  # <-- Correction : pas de Depends()
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    create_user(db, username, password, role, current_user.username)
    return RedirectResponse(url="/users", status_code=303)


@app.post("/users/delete/{username}")
async def delete_user_route(
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    if current_user.role not in ["super_admin", "admin"]:
        return RedirectResponse(url="/", status_code=303)
    try:
        delete_user(db, username, current_user)
        
        # Si l'utilisateur supprimé est celui connecté, on vide la session
        if current_user.username == username:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        
        return RedirectResponse(url="/users", status_code=303)
    except (PermissionError, ValueError) as e:
        users = list_visible_users(db, current_user)
        return templates.TemplateResponse(
            "users.html",
            {
                "request": request,
                "users": users,
                "current_user": current_user,
                "error": str(e)
            },
            status_code=403
        )


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, current_user: User = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("change_password.html", {
        "request": request,
        "current_user": current_user
    })

@app.post("/change-password")
async def change_password_route(
    request: Request,
    current_user: User = Depends(get_current_user),
    old_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    try:
        change_password(db, current_user.username, old_password, new_password)
        request.session.clear()
        return RedirectResponse(url="/login?message=Password+changed+successfully", status_code=303)
    except ValueError as e:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "current_user": current_user,
            "error": str(e)
        }, status_code=400)
    
@app.post("/users/change-password")
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Récupère l'utilisateur dans la base
    user_db = db.query(UserDB).filter(UserDB.username == current_user.username).first()
    if not user_db or not verify_password(old_password, user_db.password_hash):
        # Si ancien mot de passe incorrect, renvoyer erreur
        users = list_visible_users(db, current_user) if current_user.role != "simple" else []
        return templates.TemplateResponse(
            "users.html",
            {
                "request": request,
                "users": users,
                "current_user": current_user,
                "error": "Ancien mot de passe incorrect"
            }
        )

    # Met à jour le mot de passe
    user_db.password_hash = hash_password(new_password)
    db.commit()

    users = list_visible_users(db, current_user) if current_user.role != "simple" else []
    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "users": users,
            "current_user": current_user,
            "error": None,
            "message": "Mot de passe modifié avec succès"
        }
    )


@app.get("/reset-password/{username}", response_class=HTMLResponse)
async def reset_password_page(
    request: Request,
    username: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role not in ["super_admin", "admin"]:
        return RedirectResponse(url="/", status_code=303)
    
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        return RedirectResponse(url="/users", status_code=303)
    
    return templates.TemplateResponse("reset_password.html", {
        "request": request,
        "current_user": current_user,
        "target_user": user
    })

@app.post("/reset-password/{username}")
async def reset_password_route(
    request: Request,
    username: str,
    new_password: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user or current_user.role not in ["super_admin", "admin"]:
        return RedirectResponse(url="/", status_code=303)
    
    try:
        reset_password(db, username, new_password, current_user)
        return RedirectResponse(url="/users?message=Password+reset+successfully", status_code=303)
    except (ValueError, PermissionError) as e:
        user = db.query(UserDB).filter(UserDB.username == username).first()
        return templates.TemplateResponse("reset_password.html", {
            "request": request,
            "current_user": current_user,
            "target_user": user,
            "error": str(e)
        }, status_code=400)

    
# ===========================
# PREVIEW ET EXPORT
# ===========================

@app.get("/preview/{file_type}")
def get_preview(file_type: str, limit: int = 10):
    if file_type == "anomaly_counts":
        if "detailed" not in results_cache or "Nombre_anomalies" not in results_cache["detailed"]:
            return {"error": "Résumé des anomalies non disponible"}
        df = results_cache["detailed"]["Nombre_anomalies"]
    elif file_type not in results_cache:
        return {"error": "Aucun aperçu disponible"}
    else:
        df = results_cache[file_type]
    return {"html": df.head(limit).to_html(classes="table", index=False)} if limit != -1 else {"html": df.to_html(classes="table", index=False)}

@app.get("/preview-full-report")
async def preview_full_report(limit: int = 10):
    if "detailed" not in results_cache:
        return {"error": "Rapport complet non disponible"}
    return {
        sheet_name: (df.to_html(classes="table", index=False) if limit == -1 else df.head(limit).to_html(classes="table", index=False))
        for sheet_name, df in results_cache["detailed"].items()
    }

@app.get("/download/{filekey}")
async def download_file(filekey: str, current_user: User = Depends(require_auth)):
    if filekey not in results_cache:
        return HTMLResponse("Fichier non trouvé", status_code=404)

    if filekey == "detailed":
        file_bytes = dict_dfs_to_excel_bytes(results_cache[filekey])
        filename = f"rapport_detaille_par_anomalie_{results_cache['month']}.xlsx"
    else:
        file_bytes = df_to_excel_bytes(results_cache[filekey])
        filename_map = {
            "cleaned": "donnees_traitees",
            "non_evolving": "compteurs_non_consommants",
            "analysis": "rapport_anomalie"
        }
        filename = f"{filename_map[filekey]}_{results_cache['month']}.xlsx"

    return StreamingResponse(
        file_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/view-full-report")
async def view_full_report(current_user: User = Depends(require_auth)):
    if "full_report_path" not in results_cache or not os.path.exists(results_cache["full_report_path"]):
        return HTMLResponse("Rapport complet non disponible", status_code=404)
    month = results_cache.get("month", "donnees")
    return FileResponse(
        results_cache["full_report_path"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=rapport_complet_{month}.xlsx"}
    )

# ===========================
# UTILITAIRES
# ===========================

def df_to_excel_bytes(df: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

def dict_dfs_to_excel_bytes(dfs_dict: dict) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    output.seek(0)
    return output

@atexit.register
def cleanup():
    if "full_report_path" in results_cache and os.path.exists(results_cache["full_report_path"]):
        try:
            os.unlink(results_cache["full_report_path"])
        except:
            pass
