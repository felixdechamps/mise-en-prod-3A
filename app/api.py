import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import skops.io as sio
import pandas as pd
import s3fs
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration
BUCKET = os.environ.get("MY_BUCKET")
JETON_API = os.environ.get("JETON_API")
MODEL_S3_PATH = f"{BUCKET}/mise-en-prod/model.skops"
MODEL_LOCAL_PATH = "model.skops"

# Télécharger le modèle depuis S3 au démarrage (si pas déjà en local)
if not os.path.exists(MODEL_LOCAL_PATH):
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )
    fs.get(MODEL_S3_PATH, MODEL_LOCAL_PATH)

# Chargement du modèle
trusted_types = sio.get_untrusted_types(file=MODEL_LOCAL_PATH)
model = sio.load(MODEL_LOCAL_PATH, trusted=trusted_types)


# Schéma des données d'entrée
class Features(BaseModel):
    dd: float    # direction du vent
    ff: float    # vitesse du vent
    t: float     # température
    td: float    # point de rosée
    precip: float
    hu: float    # humidité


@app.get("/")
def root():
    return {"message": "API incendies OK"}


@app.post("/predict")
def predict(features: Features, x_token: str = Header(default=None)):
    # Vérification du jeton API
    if JETON_API and x_token != JETON_API:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    X = pd.DataFrame([features.model_dump()])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"incendie": int(prediction), "probabilite": float(proba)}