"""Page : Tester le modèle avec des paramètres personnalisés."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from utils import (
    inject_custom_css, render_sidebar,
    predict_risk, get_risk_color, get_risk_label
)

st.set_page_config(page_title="🧪 Tester le modèle", page_icon="🧪", layout="wide")
inject_custom_css()
render_sidebar()

st.title("🧪 Tester le modèle")
st.markdown("Ajustez les paramètres météo ci-dessous pour obtenir une prédiction en temps réel.")

# Session state pour l'historique
if "history" not in st.session_state:
    st.session_state.history = []

# Presets
st.markdown("### 🎯 Presets")
col1, col2, col3, col4 = st.columns(4)
preset = None
if col1.button("☀️ Canicule sèche", use_container_width=True):
    preset = {"dd": 180, "ff": 15.0, "t": 38.0, "td": 8.0, "precip": 0.0, "hu": 18}
if col2.button("🌧️ Pluie d'été", use_container_width=True):
    preset = {"dd": 270, "ff": 5.0, "t": 22.0, "td": 19.0, "precip": 15.0, "hu": 85}
if col3.button("❄️ Hiver froid", use_container_width=True):
    preset = {"dd": 90, "ff": 3.0, "t": 5.0, "td": 2.0, "precip": 2.0, "hu": 75}
if col4.button("🍂 Automne doux", use_container_width=True):
    preset = {"dd": 180, "ff": 4.0, "t": 15.0, "td": 10.0, "precip": 1.0, "hu": 65}

st.divider()

# Form
col_form, col_result = st.columns([1, 1])

with col_form:
    st.markdown("### 🎛️ Paramètres météo")
    
    dd = st.slider("🧭 Direction du vent (°)", 0, 360, preset["dd"] if preset else 180,
                   help="0 = Nord, 90 = Est, 180 = Sud, 270 = Ouest")
    ff = st.slider("💨 Vitesse du vent (m/s)", 0.0, 30.0, preset["ff"] if preset else 5.0, step=0.5)
    t = st.slider("🌡️ Température (°C)", -10.0, 45.0, preset["t"] if preset else 25.0, step=0.5)
    td = st.slider("💧 Point de rosée (°C)", -10.0, 45.0, preset["td"] if preset else 15.0, step=0.5)
    precip = st.slider("🌧️ Précipitations (mm)", 0.0, 50.0, preset["precip"] if preset else 0.0, step=0.5)
    hu = st.slider("💦 Humidité (%)", 0, 100, preset["hu"] if preset else 40)
    
    if td > t:
        st.warning("⚠️ Le point de rosée est normalement inférieur à la température.")

with col_result:
    st.markdown("### 🎯 Résultat")
    
    try:
        features = {"dd": dd, "ff": ff, "t": t, "td": td, "precip": precip, "hu": hu}
        result = predict_risk(features)
        proba = result["probabilite"]
        
        # Gauge plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={'text': "Probabilité d'incendie (%)", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': get_risk_color(proba), 'thickness': 0.7},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ddd",
                'steps': [
                    {'range': [0, 20], 'color': '#D1FAE5'},
                    {'range': [20, 50], 'color': '#FEF3C7'},
                    {'range': [50, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Verdict
        label = get_risk_label(proba)
        color = get_risk_color(proba)
        st.markdown(f"""
        <div style='padding: 1rem; background: white; border-radius: 12px; 
                    border-left: 6px solid {color}; margin-top: 1rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);'>
          <h2 style='margin: 0; color: {color};'>{label}</h2>
          <p style='margin: 0.5rem 0 0 0; color: #666;'>
            Probabilité d'incendie : <b>{proba*100:.1f}%</b>
          </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sauvegarder dans l'historique
        if st.button("💾 Enregistrer cette prédiction", use_container_width=True):
            st.session_state.history.append({
                "timestamp": datetime.now(),
                "features": features,
                "proba": proba,
            })
            st.success("✅ Enregistré dans l'historique")
    
    except Exception as e:
        st.error(f"❌ Erreur API : {e}")

st.divider()

# Historique
st.markdown("### 📋 Historique de votre session")

if not st.session_state.history:
    st.info("Aucune prédiction enregistrée. Utilisez le bouton 💾 ci-dessus pour en sauvegarder.")
else:
    df_hist = pd.DataFrame([
        {
            "Heure": h["timestamp"].strftime("%H:%M:%S"),
            "T (°C)": h["features"]["t"],
            "Humidité (%)": h["features"]["hu"],
            "Vent (m/s)": h["features"]["ff"],
            "Précip (mm)": h["features"]["precip"],
            "Risque (%)": f"{h['proba']*100:.1f}",
            "Niveau": get_risk_label(h["proba"]),
        }
        for h in reversed(st.session_state.history)
    ])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Vider l'historique"):
        st.session_state.history = []
        st.rerun()