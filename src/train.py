import argparse
import logging
import os
from numbers import Number
from itertools import product
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


def parse_int_list(value):
    """Convertit une liste de valeurs séparées par des virgules en entiers."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    """Convertit une liste de valeurs séparées par des virgules en flottants."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_logistic_regression(c_value=1.0, max_iter=1000):
    """Crée le pipeline pour la Régression Logistique."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    C=c_value,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_adaboost(n_estimators=10, learning_rate=0.1, tree_max_depth=3):
    """Crée le pipeline pour le modèle AdaBoost."""
    base_tree = DecisionTreeClassifier(max_depth=tree_max_depth, class_weight="balanced")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                AdaBoostClassifier(
                    estimator=base_tree,
                    n_estimators=n_estimators,
                    algorithm="SAMME",
                    learning_rate=learning_rate,
                    random_state=42,
                ),
            ),
        ]
    )


def build_xgboost(scale_weight, n_estimators=100, learning_rate=0.1, max_depth=6):
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
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
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


def main(data_path, models_dir, args):
    """Fonction principale orchestrant l'entraînement."""
    x_train, x_test, y_train, y_test = load_and_split_data(data_path)
    comparison_rows = []

    with mlflow.start_run(run_name="train_all_models"):
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("train_size", len(x_train))
        mlflow.log_param("test_size", len(x_test))
        mlflow.log_param("features", ",".join(FEATURES))

        # XGBoost ratio pour l'équilibrage des classes
        ratio_desequilibre = (y_train == 0).sum() / (y_train == 1).sum()

        # Entraînements multi-configurations pour comparaison
        for c_value, max_iter in product(
            args.logreg_c_values, args.logreg_max_iter_values
        ):
            config_name = f"C={c_value}_max_iter={max_iter}"
            logreg_model = build_logistic_regression(c_value=c_value, max_iter=max_iter)
            train_and_log_model(
                logreg_model,
                f"LogisticRegression_{config_name}",
                f"logistic_regression_C{c_value}_iter{max_iter}.joblib",
                x_train,
                y_train,
                x_test,
                y_test,
                models_dir,
                extra_params={"C": c_value, "max_iter": max_iter},
            )
            metrics, _, _ = evaluate_model(logreg_model, x_test, y_test, "LogReg compare")
            comparison_rows.append(
                {
                    "model": "LogisticRegression",
                    "config": config_name,
                    "mode": "manual_search",
                    "auc_roc": metrics["auc_roc"],
                    "accuracy": metrics["accuracy"],
                }
            )

        for n_estimators, learning_rate, tree_depth in product(
            args.adaboost_n_estimators_values,
            args.adaboost_learning_rate_values,
            args.adaboost_tree_depth_values,
        ):
            config_name = (
                f"n_estimators={n_estimators}_lr={learning_rate}_depth={tree_depth}"
            )
            adaboost_model = build_adaboost(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                tree_max_depth=tree_depth,
            )
            train_and_log_model(
                adaboost_model,
                f"AdaBoost_{config_name}",
                f"adaboost_n{n_estimators}_lr{learning_rate}_d{tree_depth}.joblib",
                x_train,
                y_train,
                x_test,
                y_test,
                models_dir,
                extra_params={
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "tree_max_depth": tree_depth,
                },
            )
            metrics, _, _ = evaluate_model(adaboost_model, x_test, y_test, "AdaBoost compare")
            comparison_rows.append(
                {
                    "model": "AdaBoost",
                    "config": config_name,
                    "mode": "manual_search",
                    "auc_roc": metrics["auc_roc"],
                    "accuracy": metrics["accuracy"],
                }
            )

        last_xgboost_model = None
        for n_estimators, learning_rate, max_depth in product(
            args.xgb_n_estimators_values,
            args.xgb_learning_rate_values,
            args.xgb_max_depth_values,
        ):
            config_name = f"n_estimators={n_estimators}_lr={learning_rate}_depth={max_depth}"
            xgboost_model = build_xgboost(
                scale_weight=ratio_desequilibre,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
            xgboost_model = train_and_log_model(
                xgboost_model,
                f"XGBoost_{config_name}",
                f"xgboost_n{n_estimators}_lr{learning_rate}_d{max_depth}.joblib",
                x_train,
                y_train,
                x_test,
                y_test,
                models_dir,
                extra_params={
                    "scale_pos_weight": float(ratio_desequilibre),
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                },
            )
            metrics, _, _ = evaluate_model(xgboost_model, x_test, y_test, "XGBoost compare")
            comparison_rows.append(
                {
                    "model": "XGBoost",
                    "config": config_name,
                    "mode": "manual_search",
                    "auc_roc": metrics["auc_roc"],
                    "accuracy": metrics["accuracy"],
                }
            )
            last_xgboost_model = xgboost_model

        if last_xgboost_model is not None:
            skops_path = os.path.join(models_dir, "model.skops")
            save_model_skops(last_xgboost_model, skops_path)
            mlflow.log_artifact(skops_path, artifact_path="production_model")

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df = comparison_df.sort_values(by="auc_roc", ascending=False)
        comparison_path = os.path.join(models_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("Comparaison des performances:\n%s", comparison_df.head(10))
        mlflow.log_artifact(comparison_path, artifact_path="comparison")

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
    parser.add_argument(
        "--logreg_c_values",
        type=parse_float_list,
        default=[0.1],
        help="Valeurs de C pour Logistic Regression (ex: 0.1,1,10)",
    )
    parser.add_argument(
        "--logreg_max_iter_values",
        type=parse_int_list,
        default=[1000],
        help="Valeurs de max_iter pour Logistic Regression (ex: 500,1000)",
    )

    parser.add_argument(
        "--adaboost_n_estimators_values",
        type=parse_int_list,
        default=[10],
        help="Valeurs de n_estimators pour AdaBoost (ex: 10,50)",
    )
    parser.add_argument(
        "--adaboost_learning_rate_values",
        type=parse_float_list,
        default=[0.1],
        help="Valeurs de learning_rate pour AdaBoost (ex: 0.05,0.1,0.2)",
    )
    parser.add_argument(
        "--adaboost_tree_depth_values",
        type=parse_int_list,
        default=[4],
        help="Valeurs de profondeur d'arbre pour AdaBoost (ex: 2,3,4)",
    )

    parser.add_argument(
        "--xgb_n_estimators_values",
        type=parse_int_list,
        default=[100],
        help="Valeurs de n_estimators pour XGBoost (ex: 100,200)",
    )
    parser.add_argument(
        "--xgb_learning_rate_values",
        type=parse_float_list,
        default=[0.1],
        help="Valeurs de learning_rate pour XGBoost (ex: 0.05,0.1)",
    )
    parser.add_argument(
        "--xgb_max_depth_values",
        type=parse_int_list,
        default=[6],
        help="Valeurs de max_depth pour XGBoost (ex: 4,6,8)",
    )

    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    main(args.data_path, args.models_dir, args)
