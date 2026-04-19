"""Page d'accueil du dashboard Fire Risk."""
import streamlit as st
from utils import inject_custom_css, render_sidebar, API_URL

st.set_page_config(
    page_title="🔥 Fire Risk | Accueil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar()

# Hero
st.markdown("""
<div class="hero">
    <h1>🔥 Fire Risk</h1>
    <p>Évaluation en temps réel du risque d'incendie en France métropolitaine</p>
</div>
""", unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏙️ Villes surveillées", "12", "en temps réel")
col2.metric("🎯 AUC du modèle", "0.84", "XGBoost")
col3.metric("⚡ Latence API", "< 10ms", "P95")
col4.metric("☸️ Uptime", "99.9%", "Kubernetes")

st.divider()

# Présentation du projet
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("""
    ## 🎯 À propos du projet
    
    Cette application prédit le **risque d'incendie de forêt** en France à partir 
    de conditions météorologiques en temps réel.
    
    Le modèle de Machine Learning (**XGBoost**) a été entraîné sur des données 
    historiques d'incendies français et déployé via un pipeline MLOps complet :
    
    - 🤖 **API FastAPI** conteneurisée avec Docker
    - ☸️ **Orchestration** sur Kubernetes (SSP Cloud)
    - 🔄 **CI/CD** automatisé (GitHub Actions + ArgoCD GitOps)
    - 💾 **Stockage** du modèle sur MinIO (S3)
    - 📊 **Frontend** Streamlit interactif
    
    ### 🚀 Explorer l'application
    
    Utilisez le menu à gauche pour :
    - 🗺️ **Carte France** — Voir le risque en temps réel sur une carte interactive
    - 🧪 **Tester le modèle** — Faire une prédiction avec vos propres paramètres
    - 🔬 **Comparer scénarios** — Comparer l'impact de différentes conditions
    - ℹ️ **À propos** — Architecture technique détaillée
    """)

with col_right:
    st.info("""
    ### 💡 Cas d'usage
    
    Cette API pourrait être utilisée par :
    
    - **Services de protection civile** pour anticiper les déploiements
    - **Gestionnaires forestiers** pour planifier les patrouilles
    - **Applications grand public** d'alerte météo
    - **Recherche académique** sur le changement climatique
    """)
    
    st.warning("""
    ### ⚠️ Avertissement
    
    Ce projet est une démonstration académique. Les prédictions ne doivent 
    pas être utilisées pour des décisions opérationnelles réelles sans 
    validation par des experts.
    """)

st.divider()

# Architecture
st.markdown("## 🏗️ Architecture technique")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📦 Infrastructure
    - **Kubernetes** (SSP Cloud)
    - **Docker** + Docker Hub
    - **MinIO** (stockage S3)
    - **Nginx Ingress** + HTTPS
    """)

with col2:
    st.markdown("""
    ### 🤖 Machine Learning
    - **XGBoost** Classifier
    - **scikit-learn** pour le preprocessing
    - **skops** pour la sérialisation
    - **6 features** météo en entrée
    """)

with col3:
    st.markdown("""
    ### 🔄 CI/CD
    - **GitHub Actions** (build & push)
    - **ArgoCD** (déploiement GitOps)
    - **Versioning** sémantique
    - **Secrets** Kubernetes
    """)

st.divider()
st.caption("Développé dans le cadre du cours de Mise en production à l'ENSAE Paris — 2026")