import argparse
import logging
import os
from numbers import Number
from dotenv import load_dotenv

import joblib
import skops.io as sio
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Définition des variables globales
FEATURES = ["dd", "ff", "t", "td", "precip", "hu"]
TARGET = "incendie"


def configure_mlflow():
    """Configure le tracking MLflow à partir des variables d'environnement."""
    mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_server:
        mlflow.set_tracking_uri(mlflow_server)
        logger.info("Tracking MLflow configuré sur : %s", mlflow_server)
    else:
        logger.warning(
            "MLFLOW_TRACKING_URI non défini, MLflow utilisera le tracking local par défaut."
        )


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

    precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "auc_roc": float(auc_roc),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_class_1": float(precision_1),
        "recall_class_1": float(recall_1),
        "f1_class_1": float(f1_1),
    }

    return metrics, conf_matrix, report


def save_model(model, models_dir, filename):
    """Sauvegarde le modèle entraîné"""
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    logger.info("Modèle sauvegardé : %s", filepath)
    return filepath
    

def save_model_skops(model, filepath):
    sio.dump(model, filepath)
    logger.info("Modèle de production sauvegardé : %s", filepath)


def _log_model_params(params):
    """Logge seulement les paramètres compatibles avec MLflow."""
    for key, value in params.items():
        if value is None:
            mlflow.log_param(key, "None")
        elif isinstance(value, (str, bool, Number)):
            mlflow.log_param(key, value)


def train_and_log_model(
    model,
    model_name,
    model_filename,
    x_train,
    y_train,
    x_test,
    y_test,
    models_dir,
    extra_params=None,
):
    """Entraîne, évalue, sauvegarde et logge un modèle dans MLflow."""
    with mlflow.start_run(run_name=model_name, nested=True):
        logger.info("Démarrage de l'entraînement : %s...", model_name)
        model.fit(x_train, y_train)

        metrics, conf_matrix, report = evaluate_model(model, x_test, y_test, model_name)
        mlflow.log_metrics(metrics)

        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            classifier_params = model.named_steps["classifier"].get_params()
            _log_model_params(classifier_params)

        if extra_params:
            _log_model_params(extra_params)

        model_path = save_model(model, models_dir, model_filename)
        mlflow.log_artifact(model_path, artifact_path="joblib_models")
        mlflow.sklearn.log_model(model, artifact_path="model")

        report_path = os.path.join(
            models_dir, f"{os.path.splitext(model_filename)[0]}_report.txt"
        )
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(str(report))
        mlflow.log_artifact(report_path, artifact_path="reports")

        conf_matrix_path = os.path.join(
            models_dir, f"{os.path.splitext(model_filename)[0]}_confusion_matrix.csv"
        )
        pd.DataFrame(conf_matrix).to_csv(conf_matrix_path, index=False)
        mlflow.log_artifact(conf_matrix_path, artifact_path="reports")

    return model



def main(data_path, models_dir):
    """Fonction principale orchestrant l'entraînement."""
    x_train, x_test, y_train, y_test = load_and_split_data(data_path)

    with mlflow.start_run(run_name="train_all_models"):
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("train_size", len(x_train))
        mlflow.log_param("test_size", len(x_test))
        mlflow.log_param("features", ",".join(FEATURES))

        # Entraînement et sauvegarde de la régression Logistique
        logreg_model = build_logistic_regression()
        train_and_log_model(
            logreg_model,
            "LogisticRegression",
            "logistic_regression.joblib",
            x_train,
            y_train,
            x_test,
            y_test,
            models_dir,
        )

        # Entraînement et sauvegarde de l'AdaBoost
        adaboost_model = build_adaboost()
        train_and_log_model(
            adaboost_model,
            "AdaBoost",
            "adaboost.joblib",
            x_train,
            y_train,
            x_test,
            y_test,
            models_dir,
        )

        # XGBoost
        # ratio pour l'équilibrage des classes
        ratio_desequilibre = (y_train == 0).sum() / (y_train == 1).sum()
        xgboost_model = build_xgboost(scale_weight=ratio_desequilibre)
        xgboost_model = train_and_log_model(
            xgboost_model,
            "XGBoost",
            "xgboost.joblib",
            x_train,
            y_train,
            x_test,
            y_test,
            models_dir,
            extra_params={"scale_pos_weight": float(ratio_desequilibre)},
        )

        skops_path = os.path.join(models_dir, "model.skops")
        save_model_skops(xgboost_model, skops_path)
        mlflow.log_artifact(skops_path, artifact_path="production_model")

    logger.info("Pipeline d'entraînement terminé avec succès.")


if __name__ == "__main__":

    load_dotenv()
    configure_mlflow()
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
    parser.add_argument(
    "--experiment_name", 
    type=str, 
    default="incendies-tracking", 
    help="Expérience MLFlow"
    )
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    main(args.data_path, args.models_dir)
