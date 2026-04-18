import argparse
import logging
import os
from dotenv import load_dotenv

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Définition des variables globales
FEATURES = ["dd", "ff", "t", "td", "precip", "hu"]
TARGET = "incendie"


def load_and_split_data(data_path):
    """Charge les données, filtre les colonnes utiles et sépare en train/test."""
    logger.info("Chargement des données")
    df = pd.read_parquet(
        data_path,
        storage_options={
            "client_kwargs": {'endpoint_url': 'https://minio.lab.sspcloud.fr'},
            "anon": True
        }
    )

    cols_to_keep = FEATURES + [TARGET]
    df_prediction = df[cols_to_keep].copy()

    shape_before = df_prediction.shape
    df_prediction = df_prediction.dropna()
    logger.info(
        "Lignes avec valeurs manquantes supprimées : %s -> %s",
        shape_before,
        df_prediction.shape,
    )

    logger.info("Distribution de la cible :\n%s", df_prediction[TARGET].value_counts())

    x_data = df_prediction[FEATURES]
    y_data = df_prediction[TARGET]

    return train_test_split(x_data, y_data, test_size=0.2, random_state=42)


def build_logistic_regression():
    """Crée le pipeline pour la Régression Logistique."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def build_adaboost():
    """Crée le pipeline pour le modèle AdaBoost."""
    base_tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                AdaBoostClassifier(
                    estimator=base_tree,
                    n_estimators=10,
                    algorithm="SAMME",
                    learning_rate=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost(scale_weight):
    """Crée le pipeline pour le modèle XGBoost."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    scale_pos_weight=scale_weight,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model, x_test, y_test, model_name):
    """Calcule et affiche les métriques de performance d'un modèle."""
    logger.info("--- Évaluation du modèle : %s ---", model_name)

    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    auc_roc = roc_auc_score(y_test, probabilities)
    logger.info("AUC-ROC : %.4f", auc_roc)

    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Matrice de confusion :\n%s", conf_matrix)

    report = classification_report(y_test, y_pred)
    logger.info("Rapport de classification :\n%s", report)

    return auc_roc


def save_model(model, models_dir, filename):
    """Sauvegarde le modèle entraîné"""
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    logger.info("Modèle sauvegardé : %s", filepath)


def main(data_path, models_dir):
    """Fonction principale orchestrant l'entraînement."""
    x_train, x_test, y_train, y_test = load_and_split_data(data_path)

    # Entraînement et sauvegarde de la régression Logistique
    logger.info("Démarrage de l'entraînement : Régression Logistique...")
    logreg_model = build_logistic_regression()
    logreg_model.fit(x_train, y_train)
    evaluate_model(logreg_model, x_test, y_test, "Régression Logistique")
    save_model(logreg_model, models_dir, "logistic_regression.joblib")

    # Entraînement et sauvegarde de l'AdaBoost
    logger.info("Démarrage de l'entraînement : AdaBoost...")
    adaboost_model = build_adaboost()
    adaboost_model.fit(x_train, y_train)
    evaluate_model(adaboost_model, x_test, y_test, "AdaBoost")
    save_model(adaboost_model, models_dir, "adaboost.joblib")

    # XGBoost
    # ratio pour l'équilibrage des classes
    ratio_desequilibre = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("Démarrage de l'entraînement : XGBoost...")
    xgboost_model = build_xgboost(scale_weight=ratio_desequilibre)
    xgboost_model.fit(x_train, y_train)
    evaluate_model(xgboost_model, x_test, y_test, "XGBoost")
    save_model(xgboost_model, models_dir, "xgboost.joblib")

    logger.info("Pipeline d'entraînement terminé avec succès.")


if __name__ == "__main__":

    load_dotenv()
    bucket = os.environ.get("MY_BUCKET")

    parser = argparse.ArgumentParser(description="Entraînement des modèles ML")
    parser.add_argument(
        "--data_path",
        type=str,
        default=f"s3://{bucket}/mise-en-prod/dataset_final.parquet",
        help="Chemin vers le fichier de données préparées",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Dossier où sauvegarder les modèles entraînés",
    )
    args = parser.parse_args()

    main(args.data_path, args.models_dir)
