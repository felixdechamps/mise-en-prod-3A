"""Page : Informations sur le projet et l'architecture."""
import streamlit as st
from utils import inject_custom_css, render_sidebar, API_URL

st.set_page_config(page_title="À propos", page_icon="ℹ️", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">À propos du projet</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Prédiction du risque d\'incendie de forêt — pipeline MLOps complet</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["Vue d'ensemble", "Architecture", "Modèle", "API"])

# ========== TAB 1 ==========
with tab1:
    st.markdown("""
    <div class='card' style='margin-top:1rem;'>
      <div style='font-size:1.05rem; line-height:1.7; color:#1a2b3c;'>
        Ce projet s'inscrit dans le cours <b>Mise en production des projets de data science</b> à l'ENSAE Paris.
        L'objectif n'est pas de construire le meilleur modèle, mais de mettre en place un <b>pipeline MLOps complet</b>
        — de l'entraînement au déploiement continu, avec les bonnes pratiques d'industrialisation.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Compétences mobilisées</div>', unsafe_allow_html=True)

    skills = [
        ("Infrastructure as Code", "Manifests Kubernetes, Dockerfiles"),
        ("CI/CD",                  "GitHub Actions pour le build, ArgoCD pour le déploiement GitOps"),
        ("Conteneurisation",       "Docker, multi-stage builds"),
        ("Orchestration",          "Kubernetes : Pods, Services, Ingress, Secrets"),
        ("Développement API",      "FastAPI, Pydantic, authentification par token"),
        ("Frontend",               "Streamlit multi-pages, composants interactifs"),
        ("Stockage distribué",     "MinIO (compatible S3)"),
        ("Versioning",             "Git, tags sémantiques, branches"),
    ]

    # Grille 2 colonnes
    for i in range(0, len(skills), 2):
        c1, c2 = st.columns(2, gap="medium")
        for col, (name, desc) in zip((c1, c2), skills[i:i+2]):
            col.markdown(f"""
            <div class='card' style='padding:1rem 1.2rem; margin-bottom:0.8rem;'>
              <div style='font-weight:600; color:#1a2b3c; font-size:0.95rem; margin-bottom:4px;'>{name}</div>
              <div style='color:#6b7a8c; font-size:0.87rem; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ========== TAB 2 ==========
with tab2:
    st.markdown('<div class="section-title">Architecture globale</div>', unsafe_allow_html=True)

    st.graphviz_chart("""
    digraph {
        rankdir=TB;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11,
              color="#1a2b3c", fillcolor="white", penwidth=1];
        edge [color="#6b7a8c", fontname="Inter", fontsize=9];

        User [label="Utilisateur", shape=ellipse, fillcolor="#fef3e8"];

        subgraph cluster_pub {
            label="Interfaces publiques";
            style="rounded,filled";
            color="#e8f1fb";
            fontname="Inter";
            fontsize=11;
            Site [label="Site Quarto\\n(GitHub Pages)"];
            Dash [label="Dashboard\\n(Streamlit)"];
            Swagger [label="Swagger Docs"];
        }

        subgraph cluster_k8s {
            label="Cluster Kubernetes";
            style="rounded,filled";
            color="#f0f4f8";
            fontname="Inter";
            fontsize=11;
            API [label="API FastAPI"];
            Model [label="Modèle XGBoost\\n(en mémoire)", shape=cylinder];
        }

        S3 [label="MinIO (S3)", shape=cylinder, fillcolor="#fef3e8"];

        User -> Site;
        User -> Dash;
        User -> Swagger;
        Dash -> API;
        Swagger -> API;
        API -> Model;
        Model -> S3 [label="chargé au démarrage", style=dashed];
    }
    """)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pipeline CI/CD</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    col1.markdown("""
    <div class='card'>
      <div style='font-size:0.78rem; color:#6b7a8c; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; margin-bottom:12px;'>
        Continuous Integration
      </div>
      <ol style='padding-left:1.2rem; color:#1a2b3c; line-height:1.8; margin:0; font-size:0.92rem;'>
        <li>Push sur GitHub</li>
        <li>GitHub Actions se déclenche</li>
        <li>Build parallèle de 2 images Docker<br><span style='color:#6b7a8c; font-size:0.85rem;'>app-mise-en-prod (API) et streamlit-incendies (Dashboard)</span></li>
        <li>Push sur Docker Hub, tag = nom de branche</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class='card'>
      <div style='font-size:0.78rem; color:#6b7a8c; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; margin-bottom:12px;'>
        Continuous Deployment
      </div>
      <ol style='padding-left:1.2rem; color:#1a2b3c; line-height:1.8; margin:0; font-size:0.92rem;'>
        <li>ArgoCD surveille le repo GitHub</li>
        <li>Détection des changements dans <code>kubernetes/</code></li>
        <li>Application automatique des manifests YAML</li>
        <li>Kubernetes tire les nouvelles images</li>
        <li>Rolling update sans interruption</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

# ========== TAB 3 ==========
with tab3:
    st.markdown('<div class="section-title">Choix algorithmique</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
      <div style='color:#1a2b3c; line-height:1.7; font-size:0.95rem;'>
        Le modèle est un <b>XGBoost Classifier</b>, retenu pour ses bonnes performances sur données tabulaires,
        sa rapidité d'inférence (~3 ms par prédiction), son interprétabilité (<i>feature importance</i> native)
        et sa légèreté qui facilite le déploiement conteneurisé.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Features d\'entrée</div>', unsafe_allow_html=True)

    features_data = [
        ("dd",     "Direction du vent",   "degrés (0–360)"),
        ("ff",     "Vitesse du vent",     "m/s"),
        ("t",      "Température",         "°C"),
        ("td",     "Point de rosée",      "°C"),
        ("precip", "Précipitations",      "mm"),
        ("hu",     "Humidité relative",   "%"),
    ]

    rows_html = ""
    for code, desc, unit in features_data:
        rows_html += f"""
        <div style='display:grid; grid-template-columns: 100px 1fr 140px;
                    padding:10px 14px; border-bottom:1px solid rgba(107,122,140,0.1);
                    align-items:center;'>
          <code style='background:#eef2f7; color:#1a2b3c; padding:2px 8px; border-radius:6px;
                       font-size:0.82rem; width:fit-content;'>{code}</code>
          <span style='color:#1a2b3c; font-size:0.92rem;'>{desc}</span>
          <span style='color:#6b7a8c; font-size:0.85rem; text-align:right;'>{unit}</span>
        </div>
        """

    st.markdown(f"""
    <div class='card' style='padding:0.5rem 0;'>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sortie du modèle</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
      <div style='color:#1a2b3c; line-height:1.8; font-size:0.92rem;'>
        <div><code style='background:#eef2f7; padding:2px 8px; border-radius:6px;'>incendie</code> — classe binaire (0 ou 1)</div>
        <div><code style='background:#eef2f7; padding:2px 8px; border-radius:6px;'>probabilite</code> — score ∈ [0, 1]</div>
      </div>
      <div style='margin-top:12px; color:#6b7a8c; font-size:0.87rem; line-height:1.6;'>
        Sérialisation via <b>skops</b> (alternative sécurisée à pickle), stockage sur MinIO,
        chargement à l'initialisation du Pod.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='border-left:3px solid #F59E0B; background:rgba(255,255,255,0.9);'>
      <div style='font-size:0.78rem; color:#F59E0B; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:8px;'>
        Limitation assumée
      </div>
      <div style='color:#1a2b3c; font-size:0.92rem; line-height:1.6;'>
        Le modèle n'est pas ré-entraîné automatiquement dans cette version.
        Un pipeline de retraining périodique (CronJob Kubernetes) couplé à du drift detection
        (Evidently AI) seraient les prochaines briques pour un MLOps complet en production réelle.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ========== TAB 4 ==========
with tab4:
    st.markdown('<div class="section-title">Utiliser l\'API directement</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#6b7a8c; font-size:0.92rem; margin-bottom:1rem;">Endpoint : <code>{API_URL}</code></div>',
        unsafe_allow_html=True,
    )
    # ... reste du tab à compléter selon ce que tu avais