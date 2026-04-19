"""Page : Informations sur le projet et l'architecture."""
import streamlit as st
from utils import inject_custom_css, render_sidebar, API_URL

st.set_page_config(page_title="ℹ️ À propos", page_icon="ℹ️", layout="wide")
inject_custom_css()
render_sidebar()

st.title("ℹ️ À propos du projet")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Vue d'ensemble", "🏗️ Architecture", "🤖 Modèle", "🔌 Utiliser l'API"
])

with tab1:
    st.markdown("""
    ## 🎯 Vue d'ensemble
    
    Ce projet a été réalisé dans le cadre du cours **Mise en production des projets de data science** 
    à l'ENSAE Paris (3ème année).
    
    L'objectif principal n'est pas de construire le meilleur modèle de prédiction, mais de 
    mettre en place un **pipeline MLOps complet** : de l'entraînement au déploiement continu, 
    avec toutes les bonnes pratiques d'industrialisation d'un modèle de Machine Learning.
    
    ### 🎓 Compétences mobilisées
    
    - **Infrastructure as Code** : manifests Kubernetes, Dockerfiles
    - **CI/CD** : GitHub Actions pour le build, ArgoCD pour le déploiement GitOps
    - **Conteneurisation** : Docker, multi-stage builds
    - **Orchestration** : Kubernetes, Pods, Services, Ingress, Secrets
    - **Développement API** : FastAPI, Pydantic, authentification par token
    - **Frontend** : Streamlit multi-pages, composants interactifs
    - **Stockage distribué** : MinIO (S3-compatible)
    - **Gestion de versions** : Git tags sémantiques, branches
    """)

with tab2:
    st.markdown("## 🏗️ Architecture technique")
    
    st.markdown("""
    ### Vue d'ensemble
    """)
    
    st.graphviz_chart("""
    digraph {
        rankdir=TB;
        node [shape=box, style=rounded, fontname="Helvetica"];
        
        User [label="👤 Utilisateur", shape=ellipse, fillcolor="#FEF3C7", style="filled,rounded"];
        
        subgraph cluster_pub {
            label = "🌐 Interfaces publiques";
            style = filled;
            color = "#ffe8db";
            
            Site [label="Site Quarto\\n(GitHub Pages)"];
            Dash [label="Dashboard\\n(Streamlit)"];
            Swagger [label="Swagger Docs"];
        }
        
        subgraph cluster_k8s {
            label = "☸️ Kubernetes";
            style = filled;
            color = "#e3f2fd";
            
            API [label="API FastAPI"];
            Model [label="Modèle XGBoost\\n(chargé en RAM)", shape=cylinder];
        }
        
        S3 [label="💾 MinIO (S3)", shape=cylinder, fillcolor="#F3E5F5", style="filled"];
        
        User -> Site;
        User -> Dash;
        User -> Swagger;
        Dash -> API;
        Swagger -> API;
        API -> Model;
        Model -> S3 [label="loaded\\nat startup", style=dashed];
    }
    """)
    
    st.markdown("### 🔄 Pipeline CI/CD")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Continuous Integration
        1. `git push` sur GitHub
        2. **GitHub Actions** se déclenche
        3. Build de **2 images Docker** en parallèle :
           - `app-mise-en-prod` (API)
           - `streamlit-incendies` (Dashboard)
        4. Push sur **Docker Hub** avec tag = nom de branche
        """)
    
    with col2:
        st.markdown("""
        #### Continuous Deployment
        1. **ArgoCD** surveille le repo GitHub
        2. Détecte les changements dans `kubernetes/`
        3. Applique automatiquement les YAML
        4. Kubernetes tire les nouvelles images
        5. Rolling update sans interruption
        """)

with tab3:
    st.markdown("""
    ## 🤖 Le modèle
    
    ### Choix algorithmique
    
    Le modèle utilisé est un **XGBoost Classifier**, choisi pour :
    
    - 🎯 Ses **bonnes performances** sur des données tabulaires
    - ⚡ Sa **rapidité d'inférence** (~3 ms par prédiction)
    - 🔍 Son **interprétabilité** (feature importance native)
    - 📦 Sa **légèreté** (facilite le déploiement)
    
    ### Features d'entrée
    
    Le modèle prend en entrée 6 variables météorologiques :
    
    | Variable | Description | Unité |
    |----------|-------------|-------|
    | `dd` | Direction du vent | degrés (0-360) |
    | `ff` | Vitesse du vent | m/s |
    | `t` | Température | °C |
    | `td` | Point de rosée | °C |
    | `precip` | Précipitations | mm |
    | `hu` | Humidité relative | % |
    
    ### Sortie
    
    Le modèle renvoie :
    - `incendie` : classe binaire (0 = pas d'incendie, 1 = incendie)
    - `probabilite` : score de probabilité ∈ [0, 1]
    
    ### Sérialisation
    
    Le modèle est sérialisé avec **skops** (alternative sécurisée à `pickle`), stocké sur 
    **MinIO** et téléchargé par l'API au démarrage du Pod.
    """)
    
    st.warning("""
    ⚠️ **Avertissement honnête** : le modèle n'est pas ré-entraîné automatiquement dans 
    cette version. Un pipeline de retraining périodique (CronJob Kubernetes) + drift 
    detection (Evidently AI) seraient les briques suivantes à ajouter pour un MLOps 
    complet en production réelle.
    """)

with tab4:
    st.markdown("## 🔌 Utiliser l'API directement")
    
    st.markdown("""
    Cette application utilise une API publique que vous pouvez aussi consommer directement.
    
    ### Endpoint principal