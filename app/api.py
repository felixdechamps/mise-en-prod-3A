from fastapi import FastAPI, Header
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import mlflow
import logging


# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


load_dotenv()

app = FastAPI(
    title="Démonstration du modèle de prédiction d'incendies",
    description="<b>Application de prédiction d'incendies de forêt"
                "Prédiction basée sur les conditions météo"
)

# Configuration MLFlow

logging.info(
    "Getting model from MLFlow"
)

model_name = "incendies_mlops"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)


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
    X = pd.DataFrame([features.model_dump()])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"incendie": int(prediction), "probabilite": float(proba)}