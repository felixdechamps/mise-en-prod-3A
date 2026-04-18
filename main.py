from fastapi import FastAPI
from pydantic import BaseModel
import skops.io as sio
import pandas as pd

app = FastAPI()

# Chargement du modèle au démarrage
trusted_types = sio.get_untrusted_types(file="model.skops")
model = sio.load("model.skops", trusted=trusted_types)


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
def predict(features: Features):
    X = pd.DataFrame([features.model_dump()])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {"incendie": int(prediction), "probabilite": float(proba)}
