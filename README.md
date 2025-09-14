Plateforme de Détection d’Anomalies Électriques
🚀 Description

Cette plateforme a été développée pour analyser les données de consommation électrique et détecter automatiquement des anomalies telles que :

Les compteurs non consommant.

Les monophases bloqués ou faibles.

Les dérives de consommation.

Elle combine :

FastAPI (backend) pour le traitement et l’exposition des résultats.

Jinja2/HTML pour l’interface utilisateur.

Pandas / NumPy pour la préparation et le nettoyage des données.

TensorFlow / PyTorch pour l’entraînement de modèles de détection.

⚙️ Fonctionnalités principales

📂 Upload de fichiers Excel/CSV (rapports & clients).

🔍 Détection automatique :

Compteurs non consommant.

Mono bloqué (L1, L2, L3).

Mono faible (L1, L2, L3).

📊 Analyse de l’évolution de la consommation (+A KWh).

🤖 Détection de dérives via modèles de machine learning.

📑 Génération de rapports détaillés (Excel/HTML).

🛠️ Installation
1. Cloner le dépôt
git clone https://github.com/<ton-utilisateur>/<ton-repo>.git
cd <ton-repo>

2. Créer et activer un environnement virtuel
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows

3. Installer les dépendances
pip install -r requirements.txt

▶️ Utilisation
Lancer l’application
uvicorn main:app --reload --port 10000

Accéder à l’interface

Ouvrir http://localhost:10000
 dans le navigateur.

Exemple de workflow

Charger deux fichiers (report.xlsx et clients.xlsx).

Lancer l’analyse.

Consulter les résultats :

Tableaux HTML interactifs.

Téléchargement des rapports Excel.

📂 Organisation du projet
.
├── main.py                # Point d’entrée FastAPI
├── templates/             # Interfaces Jinja2 (HTML)
├── static/                # CSS, JS, images
├── utils/                 # Fonctions de traitement (prétraitement, anomalies, ML)
├── models/                # Sauvegarde des modèles entraînés
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation

📊 Résultats de test

Détection précise des anomalies > 95%.

Robustesse face aux données bruitées.

Visualisation claire des anomalies détectées.

🤝 Contribution

Forker le projet.

Créer une branche (feature/ma-fonction).

Commit et push.

Créer une Pull Request.
