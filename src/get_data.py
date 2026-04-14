import tarfile
import requests
import io
import pyarrow.csv as pv
import pyarrow.parquet as pq
import os
from dotenv import load_dotenv

load_dotenv()
MY_BUCKET = os.environ.get("MY_BUCKET")
if not MY_BUCKET:
        raise ValueError("La variable MY_BUCKET n'est pas définie dans .env")

def extract_tar_gz_to_s3(url, s3_path, separator=','):
    """
    télécharge les données et enregistre en parquet sur S3
    """
    
    response = requests.get(url, stream=True)
    if response.status_code != 200: return

    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith('.csv'):
                print(f"Fichier trouvé : {member.name}")
                
                raw_file = tar.extractfile(member)
                if raw_file is not None:
                    file_in_memory = io.BytesIO(raw_file.read())
                    parse_options = pv.ParseOptions(delimiter=separator)
                    
                    table = pv.read_csv(file_in_memory, parse_options=parse_options)
                    noms_colonnes = table.column_names

                    # Retirer un potentiel caractère invisible de la première colonne
                    if noms_colonnes[0].startswith('\ufeff'):
                        noms_colonnes[0] = noms_colonnes[0].replace('\ufeff', '')
                        table = table.rename_columns(noms_colonnes)
                    
                    # Écriture  sur S3
                    pq.write_table(table, s3_path)
                    
                    print("fichier stocké sur le S3")
                    return
        

# --- Utilisation ---
if __name__ == "__main__":
    URL_METEO = "https://meteonet.umr-cnrm.fr/dataset/data/SE/ground_stations/SE_ground_stations_2018.tar.gz"
    S3_DESTINATION = f"s3://{MY_BUCKET}/mise-en-prod/meteo_2018.parquet"
    
    extract_tar_gz_to_s3(URL_METEO, S3_DESTINATION)

