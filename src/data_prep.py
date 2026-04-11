import os
import requests
import tarfile
import logging
import pandas as pd
import numpy as np
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Url fixe pour les données météo
URL_METEO = (
    "https://meteonet.umr-cnrm.fr/dataset/data/SE/ground_stations/"
    "SE_ground_stations_2018.tar.gz"
)

# -----télécharger les données-----


def download_and_extract_data(url, raw_dir):
    """Télécharge et extrait les données (pour les données météo)"""
    os.makedirs(raw_dir, exist_ok=True)
    file_name = url.split("/")[-1]
    file_path = os.path.join(raw_dir, file_name)

    # Téléchargement
    if not os.path.exists(file_path):
        logger.info(f"Téléchargement des données depuis {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                file.write(response.content)
            logger.info("Téléchargement réussi.")
        else:
            logger.error(f"Échec : {response.status_code}")
            return
    else:
        logger.info(f"Le fichier {file_path} existe déjà")

    # Extraction
    logger.info(f"Extraction de {file_path}...")
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        logger.info("Extraction terminée.")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction : {e}")


# -----Préparation des données-----


def prep_communes(filepath):
    """Charge et nettoie les données des communes."""
    logger.info("Préparation des communes...")
    df = pd.read_csv(filepath)
    df = df[["code_commune_INSEE", "latitude", "longitude", "nom_region"]]
    df.columns = ["code_commune_INSEE", "lat", "lon", "region"]

    liste_regions = [
        "Corse",
        "Provence-Alpes-Côte d'Azur",
        "Occitanie",
        "Auvergne-Rhône-Alpes",
        "Nouvelle-Aquitaine",
    ]
    df = (
        df[df["region"].isin(liste_regions)]
        .drop_duplicates(subset="code_commune_INSEE")
        .reset_index(drop=True)
    )

    # Correction des codes INSEE en string pour garantir le 0 initial
    df["code_commune_INSEE"] = (
        df["code_commune_INSEE"]
        .astype(str)
        .apply(lambda x: "0" + x if len(x) == 4 else x)
    )
    return df


def prep_meteo(filepath):
    """Charge, agrège les données météo et extrait les stations uniques."""
    logger.info("Préparation des données météo...")
    df_meteo = pd.read_csv(filepath)
    df_meteo["date"] = pd.to_datetime(df_meteo["date"])

    # Agrégation journalière
    df_meteo_moy = (
        df_meteo.groupby(["number_sta", pd.Grouper(key="date", freq="D")])
        .mean()
        .reset_index()
    )
    df_meteo_moy = df_meteo_moy.rename(columns={"lat": "lat_sta", "lon": "lon_sta"})

    # Extraction des stations
    df_stations = (
        df_meteo[["number_sta", "lat", "lon"]]
        .drop_duplicates(subset="number_sta")
        .reset_index(drop=True)
    )

    return df_meteo_moy, df_stations


def prep_incendies(filepath, year=2018):
    """Charge et nettoie les données d'incendies."""
    logger.info(f"Préparation des données incendies pour l'année {year}...")
    df = pd.read_csv(filepath, sep=";")

    df["Date de première alerte"] = pd.to_datetime(df["Date de première alerte"])
    df["date"] = df["Date de première alerte"].dt.date
    df = df.rename(columns={"Code INSEE": "code_commune_INSEE"})

    # On filtre sur l'année ciblée
    df_filtered = df[df["Année"] == year].copy()
    return df_filtered


# -----Construction du dataframe final-----


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points GPS en km (vectorisé)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def find_closest_station(row, df_stations):
    """Trouve la station la plus proche pour une ligne donnée."""
    dists = haversine_distance(
        row["lat"], row["lon"], df_stations["lat"].values, df_stations["lon"].values
    )
    best_idx = np.argmin(dists)
    return df_stations.iloc[best_idx]["number_sta"]


def build_final_dataset(df_communes, df_stations, df_meteo_moy, df_incendies):
    """Assemble toutes les données et crée la variable cible."""
    logger.info(
        "Association des communes à leur station la plus proche (calcul de distance)..."
    )
    df_final = df_communes.copy()

    df_final["number_sta"] = df_final.apply(
        lambda r: find_closest_station(r, df_stations), axis=1
    )
    df_final["number_sta"] = df_final["number_sta"].astype(int)

    logger.info("Jointure avec les données météo...")
    df_final = df_final.merge(df_meteo_moy, on="number_sta", how="left")
    df_final["date"] = pd.to_datetime(df_final["date"]).dt.date

    logger.info("Jointure avec les données d'incendies...")
    df_final = pd.merge(
        df_final, df_incendies, how="left", on=["date", "code_commune_INSEE"]
    )

    # Création de la variable cible
    colonnes_a_verifier = [
        "Origine de l'alerte",
        "Moyens de première intervention",
        "Surface parcourue (m2)",
        "Surface forêt (m2)",
        "Surface maquis garrigues (m2)",
        "Autres surfaces naturelles hors forêt (m2)",
        "Surfaces agricoles (m2)",
        "Autres surfaces (m2)",
        "Surface autres terres boisées (m2)",
        "Surfaces non boisées naturelles (m2)",
        "Surfaces non boisées artificialisées (m2)",
        "Surfaces non boisées (m2)",
        "Précision des surfaces",
        "Surface de feu à l'arrivée des secours > 0,1 ha",
        "Voie carrossable la plus proche",
        "Activité ou habitation la plus proche",
        "Type de peuplement",
        "Connaissance",
        "Source de l'enquête",
        "Nature",
        "Intervention de l'équipe RCCI",
        "Décès ou bâtiments touchés",
        "Nombre de décès",
        "Nombre de bâtiments totalement détruits",
        "Nombre de bâtiments partiellement détruits",
        "Hygrométrie (%)",
        "Vitesse moyenne du vent (Km/h)",
        "Direction du vent",
        "Température (°C)",
        "Précision de la donnée",
        "Présence d'un contour valide",
    ]

    logger.info("Création de la variable cible 'incendie'...")
    # Si toutes les colonnes à vérifier sont NaN, c'est qu'il n'y a pas d'incendie (0), sinon 1
    df_final["incendie"] = 1 - df_final[colonnes_a_verifier].isna().all(axis=1).astype(
        int
    )
    df_final = df_final[
        df_final.columns[df_final.isna().sum() / df_final.shape[0] < 0.99]
    ].drop(["lat", "lon", "lat_sta", "lon_sta", "height_sta", "number_sta"], axis=1)

    return df_final


#-----main----


def main(raw_dir, processed_dir):
    """Fonction principale orchestrant le pipeline de préparation."""
    # Création du dossier d'output s'il n'existe pas
    os.makedirs(processed_dir, exist_ok=True)

    download_and_extract_data(URL_METEO, raw_dir)

    df_com = prep_communes(os.path.join(raw_dir, "communes.csv"))
    df_meteo_moy, df_sta = prep_meteo(os.path.join(raw_dir, "SE2018.csv"))
    df_inc = prep_incendies(os.path.join(raw_dir, "Incendies_18.csv"))

    df_final = build_final_dataset(df_com, df_sta, df_meteo_moy, df_inc)

    # Sauvegarde dans data/processed
    output_path = os.path.join(processed_dir, "dataset_final.csv")
    df_final.to_csv(output_path, index=False)
    print(f"Pipeline terminé. Données prêtes pour le ML dans : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation données Incendies/Météo")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Dossier contenant les données brutes",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Dossier où sauvegarder les données nettoyées",
    )
    args = parser.parse_args()

    main(args.raw_dir, args.processed_dir)
