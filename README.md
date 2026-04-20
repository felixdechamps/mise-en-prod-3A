# Projet de mise en production - ENSAE 3A
Auteurs : *Thomas Chen, Félix de Champs, Clément Destouesse, Thomas Roussaux*

## Objectif
Ce projet met en place une chaîne MLOps complète pour estimer le risque d'incendie forestier à partir de variables météo.

Le projet couvre :
- préparation de données depuis des sources publiques,
- entraînement et comparaison de modèles,
- tracking des expériences avec MLflow,
- exposition du modèle via une API FastAPI,
- interface utilisateur Streamlit,
- conteneurisation Docker,
- déploiement Kubernetes,
- publication d'une page de présentation Quarto.

## Architecture
- Data prep : `src/get_data.py` et `src/data_prep.py` (lecture/écriture S3 via `MY_BUCKET`)
- Training : `src/train.py` (Logistic Regression, AdaBoost, XGBoost + MLflow)
- API : `app/api.py` (chargement du modèle `model.skops` depuis S3, endpoint `/predict`)
- Dashboard : `streamlit_app/` (multi-pages Streamlit)
- Infra : `Dockerfile`, `kubernetes/*.yaml`
- CI/CD : `.github/workflows/prod.yml`, `.github/workflows/test.yaml`, `.github/workflows/website.yaml`

Le repo GitOps associé se trouve sur https://github.com/clemdst/mise-en-prod-deployment.git.

## Structure du dépôt
```text
mise-en-prod-3A/
├── app/
│   ├── api.py
│   └── run.sh
├── data/
│   └── raw/
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── streamlit-deployment.yaml
│   ├── streamlit-service.yaml
│   └── streamlit-ingress.yaml
├── models/
├── notebooks/
├── src/
│   ├── get_data.py
│   ├── data_prep.py
│   └── train.py
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   ├── Dockerfile
│   └── requirements.txt
├── .github/workflows/
├── Dockerfile
├── _quarto.yml
├── index.qmd
├── pyproject.toml
└── README.md
```

## Prérequis
- Python 3.9+
- `uv` (gestion de dépendances)
- accès S3/MinIO (pour data et modèle)

Installation de l'environnement :
```bash
uv sync
```

## Variables d'environnement
Exemples de variables utiles selon les étapes :

- `MY_BUCKET` : bucket S3 principal (obligatoire pour data prep/train)
- `MLFLOW_TRACKING_URI` : URI du serveur MLflow (optionnel, local sinon)
- `AWS_S3_ENDPOINT` : endpoint S3/MinIO (utilisé par l'API)
- `JETON_API` : token de protection de l'API FastAPI
- `API_URL` : URL de l'API consommée par Streamlit
- `API_TOKEN` : token transmis par Streamlit à l'API

## Pipeline données
1. Récupérer les données météo et les stocker en parquet sur S3 :
```bash
uv run python src/get_data.py
```

2. Construire le dataset final :
```bash
uv run python src/data_prep.py \
  --raw_dir s3://<bucket>/mise-en-prod \
  --processed_dir s3://<bucket>/mise-en-prod
```

Le script produit `dataset_final.parquet` sur S3.

## Entraînement des modèles
Script principal : `src/train.py`

Ce script :
- charge les données depuis S3,
- entraîne Logistic Regression, AdaBoost et XGBoost,
- compare plusieurs configurations d'hyperparamètres via arguments CLI,
- peut aussi lancer un mode GridSearchCV (optionnel),
- logge params/métriques/artefacts dans MLflow,
- exporte les modèles (`.joblib`) et un modèle production `model.skops`.

### Lancement minimal
```bash
uv run python src/train.py
```

### Exemple de comparaison multi-valeurs
```bash
uv run python src/train.py \
  --logreg_c_values 0.1,1,10 \
  --logreg_max_iter_values 500,1000 \
  --adaboost_n_estimators_values 10,50 \
  --adaboost_learning_rate_values 0.05,0.1 \
  --adaboost_tree_depth_values 2,3 \
  --xgb_n_estimators_values 100,200 \
  --xgb_learning_rate_values 0.05,0.1 \
  --xgb_max_depth_values 4,6
```

### Option Grid Search
```bash
uv run python src/train.py --use_grid_search
```

Sorties principales :
- modèles dans `models/`
- comparaison dans `models/model_comparison.csv`
- artefacts et runs dans MLflow

## API FastAPI
### Lancement local
```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /` : santé API
- `POST /predict` : prédiction binaire + probabilité

### Exemple de requête
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-token: <JETON_API>" \
  -d '{"dd":180,"ff":5,"t":25,"td":15,"precip":0,"hu":40}'
```

## Dashboard Streamlit
Depuis le dossier `streamlit_app/` :
```bash
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard interroge l'API via `API_URL` et `API_TOKEN`.

## Docker
### Image API (racine du repo)
```bash
docker build -t incendies-api:local .
```

### Image Streamlit
```bash
docker build -t incendies-streamlit:local streamlit_app
```

## Kubernetes
Les manifests sont dans `kubernetes/` :
- API : `deployment.yaml`, `service.yaml`, `ingress.yaml`
- Streamlit : `streamlit-deployment.yaml`, `streamlit-service.yaml`, `streamlit-ingress.yaml`

Application (exemple) :
```bash
kubectl apply -f kubernetes/
```

## CI/CD
- `test.yaml` : installation, lint (`pylint`) et exécution du training
- `prod.yml` : build/push Docker API + Streamlit
- `website.yaml` : publication Quarto sur GitHub Pages

## Site Quarto
La page de présentation est définie par :
- `_quarto.yml`
- `index.qmd`
- `styles.css`

Publication automatisée via le workflow `website.yaml`.

## Sources de données
- BDIFF : https://bdiff.agriculture.gouv.fr/incendies
- Meteonet : https://meteonet.umr-cnrm.fr/
- Communes françaises : https://www.data.gouv.fr/fr/datasets/communes-de-france-base-des-codes-postaux/
- GeoJSON communes : https://public.opendatasoft.com/explore/dataset/georef-france-commune/
- GeoJSON régions : https://france-geojson.gregoiredavid.fr/repo/regions.geojson

## Notes
Ce projet est pédagogique. Les prédictions ne doivent pas être utilisées seules pour des décisions opérationnelles de sécurité civile.
