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

# --- Modules personnalisés ---
from backend.utils import (
    clean_and_preprocess,
    calculate_evolutions,
    detect_anomalies,
    generate_report,
    prepare_analysis_report,
    count_anomalies,
    mois_fr
)

from backend.utils2 import process_anomaly_detection

from backend.auth import (
    authenticate_user, create_user, delete_user, require_auth,
    list_visible_users, get_current_user, User,
    pwd_context, hash_password, verify_password, UserDB,
    change_password, reset_password
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
MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder_keras.h5")
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
    data = session_store.get(session_id, {})

    context = {"request": request, "files_ready": False, "current_user": current_user}

    # Vérification pour l'analyse
    if data and "analyse" in data and results_cache:
        try:
            df_anomalies = results_cache.get("anomalies")
            df_cleaned = results_cache.get("cleaned")
            df_non_evolving = results_cache.get("non_evolving")
            detailed_report = results_cache.get("detailed")

            if df_anomalies is not None and df_cleaned is not None and df_non_evolving is not None and detailed_report is not None:
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

                context.update({
                    "analyse_ready": True,
                    "month": data["analyse"]["month"],
                    "nb_sans_anomalie": nb_sans_anomalie,
                    "nb_consommants": nb_consommants,
                    "nb_non_consommants": nb_non_consommants,
                    "cas_normaux": cas_normaux,
                    "cas_avec_anomalies": cas_avec_anomalies,
                    "anomalies_par_nb": anomalies_par_nb,
                    "nombre_anomalies": nombre_anomalies,
                })
        except Exception as e:
            context["error"] = f"Erreur lors du traitement des données d'analyse : {str(e)}"
            logger.error(str(e))

    # Vérification pour la détection d'anomalies
    if data and "anomaly_detection" in data:
        try:
            df_anomaly_cleaned = results_cache.get("anomaly_cleaned")
            df_anomaly_anomalies = results_cache.get("anomaly_anomalies")
            df_anomaly_low_consumption = results_cache.get("anomaly_low_consumption")
            df_anomaly_outliers = results_cache.get("anomaly_outliers")
            anomaly_month = results_cache.get("anomaly_month", "")

            if all(df is not None for df in [df_anomaly_cleaned, df_anomaly_anomalies, df_anomaly_low_consumption, df_anomaly_outliers]):
                context.update({
                    "detection_ready": True,
                    "month": anomaly_month,
                    "cleaned_preview": data["anomaly_detection"]["cleaned_preview"],
                    "anomalies_preview": data["anomaly_detection"]["anomalies_preview"],
                    "low_consumption_preview": data["anomaly_detection"]["low_consumption_preview"],
                    "outliers_preview": data["anomaly_detection"]["outliers_preview"],
                })
            else:
                context["error"] = "Données de détection d'anomalies incomplètes dans results_cache."
        except Exception as e:
            context["error"] = f"Erreur lors du traitement des données de détection : {str(e)}"
            logger.error(str(e))

    return templates.TemplateResponse("dashboard.html", context)

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

@app.post("/anomaly-detection", response_class=HTMLResponse)
async def process_anomaly_detection_files(
    request: Request,
    current_user: User = Depends(require_auth),
    file_report: UploadFile = File(...),
    file_clients: UploadFile = File(...)
):
    session_id = request.cookies.get("session_id") or str(uuid4())

    load_profile_content = await file_report.read()
    clients_content = await file_clients.read()

    try:
        results = process_anomaly_detection(load_profile_content, clients_content, MODELS_DIR)
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {str(e)}")
        return templates.TemplateResponse("anomaly_detection.html", {"request": request, "error": str(e), "current_user": current_user})

    # Effacer les anciennes clés de results_cache pour anomaly_detection
    keys_to_remove = [
        "anomaly_cleaned", "anomaly_anomalies", 
        "anomaly_low_consumption", "anomaly_outliers", 
        "anomaly_month"
    ]
    for key in keys_to_remove:
        results_cache.pop(key, None)

    # Mettre à jour results_cache avec les nouveaux résultats
    df_cleaned = results["cleaned"]
    df_anomalies = results["anomalies"]
    df_low_consumption = results["low_consumption"]
    df_outliers = results["outliers"]
    month = results["month"]

    cleaned_preview = df_cleaned.head(10).to_html(classes="table", index=False)
    anomalies_preview = df_anomalies.head(10).to_html(classes="table", index=False)
    low_consumption_preview = df_low_consumption.head(10).to_html(classes="table", index=False)
    outliers_preview = df_outliers.head(10).to_html(classes="table", index=False)

    results_cache.update({
        "anomaly_cleaned": df_cleaned,
        "anomaly_anomalies": df_anomalies,
        "anomaly_low_consumption": df_low_consumption,
        "anomaly_outliers": df_outliers,
        "anomaly_month": month
    })

    session_store[session_id] = session_store.get(session_id, {})
    session_store[session_id]["anomaly_detection"] = {
        "month": month,
        "cleaned_preview": cleaned_preview,
        "anomalies_preview": anomalies_preview,
        "low_consumption_preview": low_consumption_preview,
        "outliers_preview": outliers_preview
    }

    response = RedirectResponse(url="/anomaly-detection", status_code=303)
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.get("/download/anomaly_cleaned")
async def download_anomaly_cleaned(current_user: User = Depends(require_auth)):
    if "anomaly_cleaned" not in results_cache:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return StreamingResponse(
        df_to_excel_bytes(results_cache["anomaly_cleaned"]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=anomaly_cleaned_{results_cache['anomaly_month']}.xlsx"}
    )

@app.get("/download/anomaly_anomalies")
async def download_anomaly_anomalies(current_user: User = Depends(require_auth)):
    if "anomaly_anomalies" not in results_cache:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return StreamingResponse(
        df_to_excel_bytes(results_cache["anomaly_anomalies"]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=anomalies_{results_cache['anomaly_month']}.xlsx"}
    )

@app.get("/download/anomaly_low_consumption")
async def download_anomaly_low_consumption(current_user: User = Depends(require_auth)):
    if "anomaly_low_consumption" not in results_cache:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return StreamingResponse(
        df_to_excel_bytes(results_cache["anomaly_low_consumption"]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=compteurs_basse_consommation_{results_cache['anomaly_month']}.xlsx"}
    )

@app.get("/download/anomaly_outliers")
async def download_anomaly_outliers(current_user: User = Depends(require_auth)):
    if "anomaly_outliers" not in results_cache:
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return StreamingResponse(
        df_to_excel_bytes(results_cache["anomaly_outliers"]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=outliers_{results_cache['anomaly_month']}.xlsx"}
    )

@app.post("/reset-anomaly-detection")
async def reset_anomaly_detection(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("anomaly_detection", None)
    # Effacer les clés spécifiques à anomaly_detection dans results_cache
    keys_to_remove = [
        "anomaly_cleaned", "anomaly_anomalies", 
        "anomaly_low_consumption", "anomaly_outliers", 
        "anomaly_month"
    ]
    for key in keys_to_remove:
        results_cache.pop(key, None)
    return RedirectResponse(url="/anomaly-detection", status_code=303)

@app.post("/reset-dashboard")
async def reset_dashboard(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        session_store[session_id].pop("analyse", None)
        session_store[session_id].pop("anomaly_detection", None)
    # Effacer toutes les clés dans results_cache
    keys_to_remove = [
        "cleaned", "non_evolving", "anomalies", "analysis", "detailed", "month",
        "anomaly_cleaned", "anomaly_anomalies", "anomaly_low_consumption", 
        "anomaly_outliers", "anomaly_month", "full_report_path"
    ]
    for key in keys_to_remove:
        results_cache.pop(key, None)
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
    request: Request,
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

def read_csv_with_dtype(file_content, sep=';', encoding='utf-8-sig'):
    """
    Lit un fichier CSV avec des types de colonnes spécifiques pour éviter les warnings
    """
    dtype_spec = {
        'meter_no': str,
        'freeze_date': str,
        'p8040': str,  # On convertira en numérique plus tard
        'numero_compteur': str,
        'partenaire': str,
        'rue': str,
        'client': str,
        'nom_client': str
    }
    
    return pd.read_csv(
        io.BytesIO(file_content), 
        sep=sep, 
        encoding=encoding, 
        low_memory=False,
        dtype=dtype_spec
    )

def prepare_visualization_data():
    """Prépare les données pour la visualisation"""
    required_keys = ["anomaly_cleaned", "anomaly_anomalies", "anomaly_low_consumption", "anomaly_outliers"]
    if not all(key in results_cache for key in required_keys):
        logger.error("Données manquantes dans results_cache: %s", required_keys)
        return {"error": "Données manquantes dans results_cache pour la visualisation"}

    df_cleaned = results_cache["anomaly_cleaned"]
    df_anomalies = results_cache["anomaly_anomalies"]
    df_low_consumption = results_cache["anomaly_low_consumption"]
    df_outliers = results_cache["anomaly_outliers"]

    total_meters = df_cleaned['Meter No.'].nunique() if not df_cleaned.empty else 0
    anomaly_meters = df_anomalies['Meter No.'].nunique() if not df_anomalies.empty else 0
    low_consumption_meters = df_low_consumption['Meter No.'].nunique() if not df_low_consumption.empty else 0
    outlier_meters = df_outliers['Meter No.'].nunique() if not df_outliers.empty else 0

    # Débogage : Afficher le nombre d'outliers
    logger.info(f"Nombre de compteurs outliers détectés: {outlier_meters}")
    if outlier_meters > 0:
        logger.info(f"Compteurs outliers: {df_outliers['Meter No.'].unique().tolist()}")

    # Éviter les compteurs dupliqués dans le calcul des compteurs normaux
    all_anomaly_meters = set(df_anomalies['Meter No.'].unique()).union(
        set(df_low_consumption['Meter No.'].unique()),
        set(df_outliers['Meter No.'].unique())
    )
    normal_meters = total_meters - len(all_anomaly_meters)

    # Compter les compteurs par motif dans df_outliers
    high_outlier_count = 0
    low_outlier_count = 0
    high_outliers = []
    low_outliers = []
    if not df_outliers.empty:
        # Compter les compteurs uniques par motif
        low_outlier_meters = df_outliers[df_outliers['Motif'] == 'Consommation faible par rapport à la moyenne']['Meter No.'].unique()
        high_outlier_meters = df_outliers[df_outliers['Motif'] == 'Consommation élevée par rapport à la moyenne']['Meter No.'].unique()
        
        low_outlier_count = len(low_outlier_meters)
        high_outlier_count = len(high_outlier_meters)
        low_outliers = low_outlier_meters.tolist()
        high_outliers = high_outlier_meters.tolist()
        
        logger.info(f"Outliers basse consommation: {low_outlier_count} ({low_outliers})")
        logger.info(f"Outliers haute consommation: {high_outlier_count} ({high_outliers})")
    else:
        logger.warning("Aucun outlier trouvé dans df_outliers")

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

    logger.info(f"Données de visualisation préparées: {visualization_data}")
    return visualization_data

def get_consumption_curves(meter_numbers, month_period):
    if "anomaly_cleaned" not in results_cache or results_cache["anomaly_cleaned"].empty:
        return None

    df_cleaned = results_cache["anomaly_cleaned"]
    
    # Convertir month_period en Period pour correspondre à year_month
    try:
        month_period = pd.Period(month_period, freq='M')
    except ValueError:
        return None

    df_month = df_cleaned[df_cleaned['year_month'] == month_period]
    
    if df_month.empty:
        return None

    meter_numbers_str = [str(m) for m in meter_numbers]
    df_filtered = df_month[df_month['Meter No.'].isin(meter_numbers_str)]
    
    if df_filtered.empty:
        return None

    df_daily = df_filtered.groupby(['Meter No.', df_filtered['Data Time'].dt.date]).agg({
        'Active power (+) total(kW)': 'sum'
    }).reset_index()
    df_daily.columns = ['Meter No.', 'date', 'sum_kW']
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    df_mean = df_daily.groupby('Meter No.')['sum_kW'].mean().reset_index()
    df_mean.columns = ['Meter No.', 'mean_consumption']
    
    try:
        df_mean['group'] = pd.qcut(
            df_mean['mean_consumption'], q=3, labels=['faible', 'moyenne', 'forte']
        )
    except:
        thresholds = [0, df_mean['mean_consumption'].quantile(0.33), 
                     df_mean['mean_consumption'].quantile(0.66), df_mean['mean_consumption'].max()]
        if thresholds[1] == thresholds[2]:
            df_mean['group'] = 'moyenne'
        else:
            df_mean['group'] = pd.cut(
                df_mean['mean_consumption'], 
                bins=thresholds, 
                labels=['faible', 'moyenne', 'forte'],
                include_lowest=True
            )
    
    df_daily = df_daily.merge(df_mean[['Meter No.', 'group']], on='Meter No.')
    
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

@app.get("/anomaly-visualization-data")
async def get_anomaly_visualization_data():
    viz_data = prepare_visualization_data()
    if not viz_data:
        return {"error": "Aucune donnée disponible pour la visualisation"}
    
    return viz_data

@app.get("/anomaly-consumption-curves")
async def get_anomaly_consumption_curves():
    if "anomaly_anomalies" not in results_cache or results_cache["anomaly_anomalies"].empty:
        return {"error": "Aucune anomalie détectée"}
    
    anomaly_meters = results_cache["anomaly_anomalies"]['Meter No.'].unique().tolist()
    month_period = results_cache.get("anomaly_month", "")
    
    curves_data = get_consumption_curves(anomaly_meters, month_period)
    if not curves_data:
        return {"error": "Impossible de générer les courbes de consommation"}
    
    return curves_data
