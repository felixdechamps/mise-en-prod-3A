"""Utilitaires partagés entre les pages du dashboard."""
import os
import requests
import streamlit as st


# Configuration
API_URL = os.environ.get("API_URL", "https://api-incendies-troussaux.lab.sspcloud.fr")
API_TOKEN = os.environ.get("API_TOKEN", "incendies2026")

# Villes françaises surveillées
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
        "ff": float(current["windspeed"]) / 3.6,  # km/h -> m/s
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


def get_risk_color(proba: float) -> str:
    """Retourne une couleur selon le niveau de risque."""
    if proba >= 0.5:
        return "#DC2626"  # rouge
    elif proba >= 0.2:
        return "#F59E0B"  # orange
    else:
        return "#16A34A"  # vert


def get_risk_label(proba: float) -> str:
    """Retourne un label textuel selon le niveau de risque."""
    if proba >= 0.5:
        return "🔴 Élevé"
    elif proba >= 0.2:
        return "🟠 Modéré"
    else:
        return "🟢 Faible"


def inject_custom_css():
    """Applique du CSS custom pour un look plus pro."""
    st.markdown("""
    <style>
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #FF4B1F 0%, #FF9068 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 75, 31, 0.25);
    }
    .hero h1 {
        font-size: 2.8rem;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    /* KPI cards */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #FF4B1F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FAFAF7 0%, #F4E9D8 100%);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 75, 31, 0.3);
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Barre latérale partagée entre toutes les pages."""
    with st.sidebar:
        st.markdown("## 🔥 Fire Risk")
        st.markdown("*Prédiction des risques d'incendies de forêt*")
        st.divider()
        
        st.markdown("### 📊 Status du système")
        try:
            r = requests.get(f"{API_URL}/", timeout=5)
            if r.status_code == 200:
                st.success("🟢 API opérationnelle")
            else:
                st.warning("🟠 API dégradée")
        except Exception:
            st.error("🔴 API injoignable")
        
        st.divider()
        
        st.markdown("### Ressources")
        st.markdown("- [API Docs](https://api-incendies-troussaux.lab.sspcloud.fr/docs)")
        st.markdown("- [Code source](https://github.com/felixdechamps/mise-en-prod-3A)")
        st.markdown("- [Site projet](https://felixdechamps.github.io/mise-en-prod-3A/)")
        
        st.divider()
        
        st.caption("Projet MLOps - ENSAE 3A")
        st.caption("Météo : [Open-Meteo](https://open-meteo.com)")
