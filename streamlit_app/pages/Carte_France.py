"""Page : Carte interactive de France avec les risques en temps réel."""
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px
from utils import (
    inject_custom_css, render_sidebar, VILLES,
    get_weather, predict_risk, get_risk_color, get_risk_label
)

st.set_page_config(page_title="🗺️ Carte France", page_icon="🗺️", layout="wide")
inject_custom_css()
render_sidebar()

st.title("🗺️ Carte du risque en France")
st.markdown("Risque d'incendie pour les principales villes, calculé à partir de la **météo actuelle**.")


@st.cache_data(ttl=600, show_spinner=False)
def compute_all_risks():
    results = {}
    errors = []
    for ville, (lat, lon) in VILLES.items():
        try:
            weather = get_weather(lat, lon)
            pred = predict_risk(weather)
            results[ville] = {
                "lat": lat, "lon": lon,
                "weather": weather,
                "prediction": pred,
            }
        except Exception as e:
            errors.append(f"{ville} : {str(e)[:100]}")
    return results, errors


# Sidebar controls
with st.sidebar:
    st.divider()
    st.markdown("### 🎨 Options de la carte")
    map_style = st.selectbox(
        "Style",
        ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"],
        index=1,
    )
    show_heatmap = st.checkbox("Afficher la heatmap", value=True)
    show_markers = st.checkbox("Afficher les marqueurs", value=True)

# Load data
with st.spinner("🌡️ Récupération de la météo en temps réel..."):
    risques, errors = compute_all_risks()

if errors:
    with st.expander(f"⚠️ {len(errors)} erreur(s) rencontrée(s)"):
        for e in errors:
            st.warning(e)

if not risques:
    st.error("Aucune donnée disponible. Vérifiez la connexion à l'API.")
    st.stop()

# KPI summary
col1, col2, col3, col4 = st.columns(4)
probas = [d["prediction"]["probabilite"] for d in risques.values()]
nb_eleves = sum(1 for p in probas if p >= 0.5)
nb_moderes = sum(1 for p in probas if 0.2 <= p < 0.5)
nb_faibles = sum(1 for p in probas if p < 0.2)

col1.metric("📊 Villes analysées", len(risques))
col2.metric("🔴 Risque élevé", nb_eleves)
col3.metric("🟠 Risque modéré", nb_moderes)
col4.metric("🟢 Risque faible", nb_faibles)

st.divider()

# Map + ranking
col_map, col_rank = st.columns([2, 1])

with col_map:
    st.markdown("### 🗺️ Carte interactive")
    
    m = folium.Map(
        location=[46.6, 2.5],
        zoom_start=6,
        tiles=map_style,
    )
    
    # Heatmap
    if show_heatmap:
        heat_data = [
            [d["lat"], d["lon"], d["prediction"]["probabilite"]]
            for d in risques.values()
        ]
        HeatMap(heat_data, radius=45, blur=30, min_opacity=0.3).add_to(m)
    
    # Markers
    if show_markers:
        for ville, d in risques.items():
            proba = d["prediction"]["probabilite"]
            color = get_risk_color(proba)
            w = d["weather"]
            
            popup_html = f"""
            <div style='font-family: sans-serif; min-width: 200px;'>
              <h4 style='margin: 0 0 8px 0; color: #FF4B1F;'>📍 {ville}</h4>
              <table style='width: 100%; font-size: 13px;'>
                <tr><td>🌡️ Température</td><td><b>{w['t']:.1f}°C</b></td></tr>
                <tr><td>💧 Humidité</td><td><b>{w['hu']:.0f}%</b></td></tr>
                <tr><td>💨 Vent</td><td><b>{w['ff']:.1f} m/s</b></td></tr>
                <tr><td>🌧️ Précip.</td><td><b>{w['precip']:.1f} mm</b></td></tr>
              </table>
              <hr style='margin: 8px 0;'>
              <p style='margin: 0; font-size: 16px;'>
                <b>Risque : {proba*100:.1f}%</b><br>
                <span style='color: {color};'>{get_risk_label(proba)}</span>
              </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[d["lat"], d["lon"]],
                radius=8 + proba * 20,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{ville} — {proba*100:.0f}%",
                color=color,
                fill=True,
                fillOpacity=0.7,
                weight=2,
            ).add_to(m)
    
    st_folium(m, use_container_width=True, height=550)

with col_rank:
    st.markdown("### 📊 Classement")
    sorted_villes = sorted(
        risques.items(),
        key=lambda x: x[1]["prediction"]["probabilite"],
        reverse=True,
    )
    
    for rang, (ville, d) in enumerate(sorted_villes, start=1):
        proba = d["prediction"]["probabilite"] * 100
        color = get_risk_color(d["prediction"]["probabilite"])
        
        st.markdown(f"""
        <div style='padding: 10px; margin: 6px 0; background: white; 
                    border-radius: 8px; border-left: 4px solid {color};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
          <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
              <span style='color: #999; font-size: 12px;'>#{rang}</span>
              <b style='font-size: 14px;'> {ville}</b>
            </div>
            <b style='color: {color}; font-size: 16px;'>{proba:.1f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Bar chart
st.markdown("### 📈 Comparaison des risques")

df = pd.DataFrame([
    {
        "Ville": ville,
        "Risque (%)": d["prediction"]["probabilite"] * 100,
        "Température (°C)": d["weather"]["t"],
        "Humidité (%)": d["weather"]["hu"],
    }
    for ville, d in risques.items()
]).sort_values("Risque (%)", ascending=True)

fig = px.bar(
    df, y="Ville", x="Risque (%)", orientation="h",
    color="Risque (%)",
    color_continuous_scale=["#16A34A", "#F59E0B", "#DC2626"],
    title="Niveau de risque par ville",
    hover_data=["Température (°C)", "Humidité (%)"],
)
fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Caption
import datetime
st.caption(f"🔄 Dernière mise à jour : {datetime.datetime.now().strftime('%H:%M:%S')} — Cache 10 min")