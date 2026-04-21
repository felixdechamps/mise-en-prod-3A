"""Page d'accueil du dashboard Fire Risk."""
import streamlit as st
from utils import inject_custom_css, render_sidebar, VILLES

st.set_page_config(
    page_title="Risque incendies",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div style='padding: 2rem 0 3rem 0;'>
  <div style='font-size: 0.82rem; color: #6b7a8c; text-transform: uppercase;
              letter-spacing: 0.12em; font-weight: 500; margin-bottom: 1rem;'>
    Projet MLOps · ENSAE Paris
  </div>
  <div style='font-size: 3.2rem; font-weight: 700; color: #1a2b3c;
              letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 1rem;'>
    Le risque d'incendie en France,<br>prédit en temps réel.
  </div>
  <div style='font-size: 1.15rem; color: #6b7a8c; line-height: 1.6; max-width: 680px;'>
    Un modèle XGBoost interrogé via une API FastAPI, déployé sur Kubernetes,
    alimenté par les conditions météo actuelles des principales villes françaises.
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cards de navigation — 3 entrées principales
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="medium")

def nav_card(col, label, title, desc):
    col.markdown(f"""
    <div class='card' style='padding:1.6rem 1.4rem; height:100%; min-height:180px;
                              display:flex; flex-direction:column; justify-content:space-between;'>
      <div>
        <div style='font-size:0.72rem; color:#6b7a8c; text-transform:uppercase;
                    letter-spacing:0.1em; font-weight:500; margin-bottom:10px;'>
          {label}
        </div>
        <div style='font-size:1.25rem; font-weight:600; color:#1a2b3c;
                    letter-spacing:-0.01em; margin-bottom:8px;'>
          {title}
        </div>
        <div style='font-size:0.9rem; color:#6b7a8c; line-height:1.5;'>
          {desc}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

nav_card(col1, "01 · Visualiser",
         "Carte de France",
         f"Risque en temps réel pour {len(VILLES)} villes, mis à jour toutes les 10 minutes.")
nav_card(col2, "02 · Simuler",
         "Simulateur",
         "Testez vos propres paramètres météo ou comparez plusieurs scénarios côte à côte.")
nav_card(col3, "03 · Comprendre",
         "À propos",
         "Architecture technique, choix du modèle et pipeline MLOps détaillé.")

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stack technique — en une ligne sobre
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Stack technique</div>', unsafe_allow_html=True)

stack_items = [
    ("Modèle",         "XGBoost"),
    ("API",            "FastAPI"),
    ("Orchestration",  "Kubernetes"),
    ("CI/CD",          "GitHub Actions + ArgoCD"),
    ("Stockage",       "MinIO (S3)"),
    ("Frontend",       "Streamlit"),
]

cols = st.columns(len(stack_items))
for col, (label, value) in zip(cols, stack_items):
    col.markdown(f"""
    <div style='padding:0.9rem 0.8rem; text-align:center;
                background:rgba(255,255,255,0.5); border-radius:12px;
                border:1px solid rgba(255,255,255,0.7);'>
      <div style='font-size:0.68rem; color:#6b7a8c; text-transform:uppercase;
                  letter-spacing:0.08em; margin-bottom:4px;'>{label}</div>
      <div style='font-size:0.92rem; color:#1a2b3c; font-weight:600;'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Avertissement — sobre
# ---------------------------------------------------------------------------
st.markdown("""
<div class='card' style='border-left:3px solid #F59E0B;
                          background:rgba(255,255,255,0.7); padding:1rem 1.3rem;'>
  <div style='font-size:0.72rem; color:#F59E0B; text-transform:uppercase;
              letter-spacing:0.08em; font-weight:600; margin-bottom:6px;'>
    Projet de démonstration
  </div>
  <div style='color:#1a2b3c; font-size:0.9rem; line-height:1.6;'>
    Les prédictions ne doivent pas être utilisées pour des décisions opérationnelles
    sans validation par des experts. L'accent a été mis sur la chaîne de mise en production,
    pas sur l'optimisation du modèle.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.caption("Cours « Mise en production des projets de data science » · ENSAE Paris · 2026")
