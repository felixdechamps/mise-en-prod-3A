"""Utilitaires partagés entre les pages du dashboard."""
import os
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "https://api-incendies-troussaux.lab.sspcloud.fr")
API_TOKEN = os.environ.get("API_TOKEN", "incendies2026")

VILLES = {
    "Paris": (48.8566, 2.3522),
    "Marseille": (43.2965, 5.3698),
    "Lyon": (45.7578, 4.8320),
    "Toulouse": (43.6047, 1.4442),
    "Nice": (43.7102, 7.2620),
    "Nantes": (47.2184, -1.5536),
    "Strasbourg": (48.5734, 7.7521),
    "Montpellier": (43.6108, 3.8767),
    "Bordeaux": (44.8378, -0.5792),
    "Lille": (50.6292, 3.0573),
    "Ajaccio": (41.9192, 8.7386),
    "Perpignan": (42.6886, 2.8949),
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def get_weather(lat: float, lon: float) -> dict:
    """Récupère la météo actuelle depuis Open-Meteo (sans clé API)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&hourly=relativehumidity_2m,dewpoint_2m,precipitation"
        f"&timezone=auto"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data["current_weather"]
    hourly = data["hourly"]

    return {
        "dd": float(current["winddirection"]),
        "ff": float(current["windspeed"]) / 3.6,
        "t": float(current["temperature"]),
        "td": float(hourly["dewpoint_2m"][0]),
        "precip": float(hourly["precipitation"][0]),
        "hu": float(hourly["relativehumidity_2m"][0]),
    }


def predict_risk(features: dict) -> dict:
    """Appelle l'API de prédiction."""
    response = requests.post(
        f"{API_URL}/predict",
        headers={"x-token": API_TOKEN},
        json=features,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------
def get_risk_color(proba: float) -> str:
    if proba >= 0.5:
        return "#DC2626"
    elif proba >= 0.2:
        return "#F59E0B"
    else:
        return "#16A34A"


def get_risk_label(proba: float) -> str:
    """Label sans émoji (les couleurs portent déjà l'info)."""
    if proba >= 0.5:
        return "Élevé"
    elif proba >= 0.2:
        return "Modéré"
    else:
        return "Faible"


# ---------------------------------------------------------------------------
# UI : CSS global (palette app météo)
# ---------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* BASE */
        .stApp {
            background: linear-gradient(160deg, #e8f1fb 0%, #f5f7fa 45%, #fef3e8 100%);
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        header[data-testid="stHeader"] { background: transparent; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }

        /* TYPOGRAPHIE */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1a2b3c;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #6b7a8c;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1a2b3c;
            margin: 0.5rem 0 1rem 0;
            letter-spacing: -0.01em;
        }

        /* CARDS */
        .card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(30, 50, 80, 0.06);
        }
        .kpi-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 4px 20px rgba(30, 50, 80, 0.06);
            transition: transform 0.2s ease;
        }
        .kpi-card:hover { transform: translateY(-2px); }
        .kpi-label {
            font-size: 0.78rem;
            color: #6b7a8c;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 600;
            color: #1a2b3c;
            line-height: 1;
        }
        .kpi-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }

        /* RANKING */
        .rank-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 14px;
            margin: 6px 0;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            border-left: 3px solid var(--accent, #ccc);
            transition: all 0.2s ease;
        }
        .rank-row:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateX(2px);
        }
        .rank-city { font-size: 0.95rem; color: #1a2b3c; font-weight: 500; }
        .rank-num { color: #9aa7b8; font-size: 0.8rem; font-weight: 500; margin-right: 6px; }
        .rank-value { font-size: 1rem; font-weight: 600; }

        /* METRIC NATIF */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 4px 20px rgba(30, 50, 80, 0.06);
        }

        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7fa 0%, #e8f1fb 100%);
            border-right: 1px solid rgba(107, 122, 140, 0.1);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        .sidebar-brand {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1a2b3c;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .sidebar-tagline {
            font-size: 0.82rem;
            color: #6b7a8c;
            margin-top: 2px;
            margin-bottom: 1.5rem;
        }
        .sidebar-label {
            font-size: 0.72rem;
            color: #6b7a8c;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
            margin: 1rem 0 0.5rem 0;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(107, 122, 140, 0.15);
            border-radius: 999px;
            font-size: 0.85rem;
            color: #1a2b3c;
            font-weight: 500;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.ok { background: #16A34A; box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.15); }
        .status-dot.warn { background: #F59E0B; box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.15); }
        .status-dot.ko { background: #DC2626; box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.15); }

        .sidebar-link {
            display: block;
            padding: 8px 12px;
            margin: 4px 0;
            color: #1a2b3c !important;
            text-decoration: none !important;
            border-radius: 8px;
            font-size: 0.88rem;
            transition: all 0.15s;
        }
        .sidebar-link:hover {
            background: rgba(255, 255, 255, 0.7);
            transform: translateX(2px);
        }
        .sidebar-footer {
            font-size: 0.78rem;
            color: #9aa7b8;
            margin-top: 0.5rem;
            line-height: 1.5;
        }

        /* WIDGETS */
        .stButton > button {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(26, 43, 60, 0.1);
            color: #1a2b3c;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s;
            padding: 0.5rem 1rem;
            box-shadow: none;
        }
        .stButton > button:hover {
            background: #1a2b3c;
            color: white;
            border-color: #1a2b3c;
            transform: translateY(-1px);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 0.5rem 1.2rem;
            font-weight: 500;
            color: #1a2b3c;
        }
        .stTabs [aria-selected="true"] {
            background: #1a2b3c !important;
            color: white !important;
        }

        hr { border: none; border-top: 1px solid rgba(107, 122, 140, 0.15); margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI : Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Barre latérale partagée entre toutes les pages."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Fire Risk</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-tagline">Prédiction des risques d\'incendies de forêt</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">État du système</div>', unsafe_allow_html=True)
        try:
            r = requests.get(f"{API_URL}/", timeout=5)
            if r.status_code == 200:
                st.markdown(
                    '<div class="status-pill"><span class="status-dot ok"></span>API opérationnelle</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-pill"><span class="status-dot warn"></span>API dégradée</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                '<div class="status-pill"><span class="status-dot ko"></span>API injoignable</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-label">Ressources</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <a class="sidebar-link" href="{API_URL}/docs" target="_blank">Documentation API</a>
        <a class="sidebar-link" href="https://open-meteo.com" target="_blank">Source météo</a>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="sidebar-footer" style="margin-top:2rem;">Projet MLOps — ENSAE 3A</div>',
            unsafe_allow_html=True,
        )
