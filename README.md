# mise-en-prod-3A
## Projet Python pour la Data Science
 Auteurs : *Thomas Chen, Félix de Champs, Clément Destouesse, Thomas Roussaux*  

# Sujet :
<div align="justify">
En octobre 2022, le Président de la République avait annoncé la création d’une météo des forêts destinée à mieux informer les Français sur le risque de feux.  
Depuis le 1er juin 2023, tous les jours à 17h, Météo France diffuse donc ce nouveau dispositif pour indiquer le niveau de danger de feux en France métropolitaine. Cette information est établie à partir des prévisions de plusieurs paramètres météorologiques qui influencent fortement le départ et la propagation des feux : pluie, humidité de l’air, température, force du vent et état de sécheresse de la végétation.  
La météo des forêts n’informe pas sur les incendies en cours ou à venir, c’est un outil d’information et de prévention destiné au public. Son objectif est d’indiquer les zones dans lesquelles les conditions météorologiques peuvent aggraver le risque de feux et de rappeler les bons réflexes pour éviter les départs de feux.  
Nous nous sommes donc dit qu'il serait intéressant dans le cadre de ce projet de produire un outil de prévention se rapprochant de ce qui se fait dans le cadre de la météo des forêt. Ainsi, nous nous sommes fixé comme objectif d'étudier l'influence des paramètres climatiques sur la probabilité d'occurrence d’un incendie forestier. Essayer de prédire l'apparition d'un incendie dans la journée en fonction des paramètres météorologiques.

# Structure du projet : 
```text
mon_projet_ds/
├── data/                   
│   ├── raw/                <- Données brutes
│   └── processed/          <- Données nettoyées
├── models/                 <- Modèles entraînés (provisoire avant MLFlow)
├── src/                    
│   ├── data_prep.py        <- Pipeline de préparation
│   ├── train.py            <- Entraînement
│   └── predict.py          <- Script d'inférence (à venir)
├── .gitignore              
├── pyproject.toml          
├── uv.lock                 
└── README.md
```
## détail des scripts sources :
- data_prep.py : télécharge les données météo, associe chaque commune à la station météo la plus proche, génère dataset_final.csv avec la variable cible incendie.
- train.py : charge dataset_final.csv, entraîne 3 modèles (régression logistiquen, adaboost et xgboost), renvoie les métriques de performance et exporte les modèles.

# Mise en route : 

## prérequis : 
Si vous n'avez pas encore `uv` installé sur votre machine :
* Sur macOS / Linux : `curl -LsSf https://astral.sh/uv/install.sh | sh` ou `brew install uv`
* Sur Windows : `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Cloner le projet
```bash
git clone https://github.com/felixdechamps/mise-en-prod-3A
cd mise-en-prod-3A
```
## Installer l'environnement :
```bash
uv sync
```

# Données utilisées :
- [BDIFF](https://bdiff.agriculture.gouv.fr/incendies) (Base de Données sur les Incendies de Forêts en France), base de données sur les feux de forêts en france de 2006 à 2022.  
- [Meteonet](https://meteonet.umr-cnrm.fr/), données météo fournies par Météo France toutes les 6 minutes pour 532 stations dans le quart Sud-Est de la France.
- [Base de données sur les communes françaises](https://www.data.gouv.fr/fr/datasets/communes-de-france-base-des-codes-postaux/), contenant notamment leurs coordonnées GPS.
- [Base de données Geojson des forêts françaises](https://transcode.geo.data.gouv.fr/services/5e2a1f74fa4268bc255efbc3/feature-types/ms:PARC_PUBL_FR?format=GeoJSON&projection=WGS84)

- [Base de données Geojson des communes françaises](https://public.opendatasoft.com/explore/dataset/georef-france-commune/information/?disjunctive.reg_name&disjunctive.dep_name&disjunctive.arrdep_name&disjunctive.ze2020_name&disjunctive.bv2022_name&disjunctive.epci_name&disjunctive.ept_name&disjunctive.com_name&disjunctive.ze2010_name&disjunctive.com_is_mountain_area&sort=-com_name&refine.dep_name=Bouches-du-Rh%C3%B4ne) et sur les [régions françaises](https://france-geojson.gregoiredavid.fr/repo/regions.geojson) qui permettent de retracer sur un fond de carte les communes touchées par les incendies.

# Navigation au sein du projet : 
Il suffit d'exécuter successivement les cellules du notebook : [notebookincendie.ipynb](notebookincendies.ipynb)

# Remarque sur la reproductibilité : 
**Attention** à ce que le dossier **ensae-prog2A** ne soit pas dans le **work**, afin que les chemins utilisés pour accéder aux fichiers correspondent bien au code du notebook.

Autres sources : [Ministère de l'écologie](https://www.ecologie.gouv.fr/feux-foret-en-france)
