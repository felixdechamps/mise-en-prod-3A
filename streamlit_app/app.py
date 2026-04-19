import streamlit as st
import folium
from streamlit_folium import st_folium
import requests

st.set_page_config(
    page_title="🔥 Risque d'incendie en France",
    layout="wide"
)

st.title("🔥 Prédiction du risque d'incendie en France")
st.markdown("Cliquez sur une ville pour évaluer le risque actuel.")

# Liste de villes françaises avec coordonnées
VILLES = {
    "Paris": (48.8566, 2.3522),
    "Marseille": (43.2965, 5.3698),
    "Lyon": (45.7578, 4.8320),
    "Toulouse": (43.6047, 1.4442),
    "Bordeaux": (44.8378, -0.5792),
    "Nice": (43.7102, 7.2620),
    "Nantes": (47.2184, -1.5536),
    "Strasbourg": (48.5734, 7.7521),
    "Montpellier": (43.6108, 3.8767),
    "Lille": (50.6292, 3.0573),
}

# Configuration
API_URL = "https://api-incendies-troussaux.lab.sspcloud.fr"
API_TOKEN = "incendies2026"

def get_weather(lat, lon):
    """Récupère la météo actuelle via Open-Meteo (gratuit, sans clé API)."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m,dewpoint_2m,precipitation&timezone=auto"
    response = requests.get(url)
    data = response.json()
    
    current = data["current_weather"]
    hourly = data["hourly"]
    now_idx = 0  # Simplification
    
    return {
        "dd": current["winddirection"],
        "ff": current["windspeed"] / 3.6,  # km/h → m/s
        "t": current["temperature"],
        "td": hourly["dewpoint_2m"][now_idx],
        "precip": hourly["precipitation"][now_idx],
        "hu": hourly["relativehumidity_2m"][now_idx],
    }

def predict_risk(features):
    """Appelle ton API de prédiction."""
    response = requests.post(
        f"{API_URL}/predict",
        headers={"x-token": API_TOKEN},
        json=features
    )
    return response.json()

def risk_color(proba):
    if proba > 0.5:
        return "red"
    elif proba > 0.2:
        return "orange"
    else:
        return "green"

# Préparer la carte avec toutes les villes
@st.cache_data(ttl=600)  # Cache 10 min
def compute_all_risks():
    results = {}
    for ville, (lat, lon) in VILLES.items():
        try:
            weather = get_weather(lat, lon)
            pred = predict_risk(weather)
            results[ville] = {
                "lat": lat,
                "lon": lon,
                "weather": weather,
                "prediction": pred,
            }
        except Exception as e:
            st.error(f"Erreur pour {ville}: {e}")
    return results

with st.spinner("Récupération des risques en temps réel..."):
    risques = compute_all_risks()

# Créer la carte
m = folium.Map(location=[46.6, 2.5], zoom_start=6)

for ville, data in risques.items():
    proba = data["prediction"]["probabilite"]
    color = risk_color(proba)
    
    popup_html = f"""
    <b>{ville}</b><br>
    🌡️ {data['weather']['t']:.1f}°C<br>
    💧 {data['weather']['hu']:.0f}%<br>
    💨 {data['weather']['ff']:.1f} m/s<br>
    <hr>
    <b>Risque : {proba*100:.1f}%</b>
    """
    
    folium.CircleMarker(
        location=[data["lat"], data["lon"]],
        radius=10 + proba * 20,
        popup=popup_html,
        color=color,
        fill=True,
        fillOpacity=0.7,
    ).add_to(m)

# Afficher
col1, col2 = st.columns([2, 1])

with col1:
    st_folium(m, width=800, height=600)

with col2:
    st.subheader("📊 Classement")
    sorted_villes = sorted(
        risques.items(),
        key=lambda x: x[1]["prediction"]["probabilite"],
        reverse=True
    )
    
    for ville, data in sorted_villes:
        proba = data["prediction"]["probabilite"] * 100
        emoji = "🔴" if proba > 50 else "🟠" if proba > 20 else "🟢"
        st.markdown(f"{emoji} **{ville}** — {proba:.1f}%")

st.caption("Données météo : Open-Meteo • Prédictions : API déployée sur Kubernetes")