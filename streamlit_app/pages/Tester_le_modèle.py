"""Page : Tester le modèle avec des paramètres personnalisés."""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from utils import (
    inject_custom_css, render_sidebar,
    predict_risk, get_risk_color, get_risk_label
)

st.set_page_config(page_title="Simulateur", page_icon="🧪", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">Simulateur de prédiction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ajustez les paramètres météorologiques pour évaluer le risque d\'incendie</div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Presets ----------
PRESETS = {
    "Canicule sèche":      {"dd": 180, "ff": 15.0, "t": 38.0, "td": 8.0,  "precip": 0.0,  "hu": 18},
    "Pluie d'été":         {"dd": 270, "ff": 5.0,  "t": 22.0, "td": 19.0, "precip": 15.0, "hu": 85},
    "Hiver froid":         {"dd": 90,  "ff": 3.0,  "t": 5.0,  "td": 2.0,  "precip": 2.0,  "hu": 75},
    "Automne doux":        {"dd": 180, "ff": 4.0,  "t": 15.0, "td": 10.0, "precip": 1.0,  "hu": 65},
}

st.markdown('<div class="section-title">Scénarios types</div>', unsafe_allow_html=True)
preset_cols = st.columns(len(PRESETS))
preset = None
for col, (name, values) in zip(preset_cols, PRESETS.items()):
    if col.button(name, use_container_width=True, key=f"preset_{name}"):
        preset = values

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Form + Result ----------
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="section-title">Paramètres</div>', unsafe_allow_html=True)

    dd = st.slider("Direction du vent (°)", 0, 360, preset["dd"] if preset else 180,
                   help="0 = Nord, 90 = Est, 180 = Sud, 270 = Ouest")
    ff = st.slider("Vitesse du vent (m/s)", 0.0, 30.0, preset["ff"] if preset else 5.0, step=0.5)
    t = st.slider("Température (°C)", -10.0, 45.0, preset["t"] if preset else 25.0, step=0.5)
    td = st.slider("Point de rosée (°C)", -10.0, 45.0, preset["td"] if preset else 15.0, step=0.5)
    precip = st.slider("Précipitations (mm)", 0.0, 50.0, preset["precip"] if preset else 0.0, step=0.5)
    hu = st.slider("Humidité relative (%)", 0, 100, preset["hu"] if preset else 40)

    if td > t:
        st.warning("Le point de rosée dépasse la température — valeur inhabituelle.")

with col_result:
    st.markdown('<div class="section-title">Prédiction</div>', unsafe_allow_html=True)

    try:
        features = {"dd": dd, "ff": ff, "t": t, "td": td, "precip": precip, "hu": hu}
        result = predict_risk(features)
        proba = result["probabilite"]
        color = get_risk_color(proba)
        label = get_risk_label(proba)

        # Gauge épurée
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%", 'font': {'size': 48, 'color': '#1a2b3c', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'transparent',
                         'tickfont': {'color': '#9aa7b8', 'size': 11}},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20],   'color': 'rgba(22, 163, 74, 0.12)'},
                    {'range': [20, 50],  'color': 'rgba(245, 158, 11, 0.12)'},
                    {'range': [50, 100], 'color': 'rgba(220, 38, 38, 0.12)'},
                ],
            }
        ))
        fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Verdict sobre
        st.markdown(f"""
        <div style='text-align:center; margin-top:-1rem;'>
          <div style='font-size:0.78rem; color:#6b7a8c; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; margin-bottom:4px;'>Niveau de risque</div>
          <div style='font-size:1.3rem; color:{color}; font-weight:600;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

        if st.button("Enregistrer ce scénario", use_container_width=True):
            st.session_state.history.append({
                "timestamp": datetime.now(),
                "features": features,
                "proba": proba,
            })
            st.toast("Scénario enregistré", icon="✓")

    except Exception as e:
        st.error(f"Erreur API : {e}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- Historique ----------
st.markdown('<div class="section-title">Historique de session</div>', unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(
        '<div style="color:#6b7a8c; font-size:0.9rem; padding:1rem 0;">Aucun scénario enregistré pour le moment.</div>',
        unsafe_allow_html=True,
    )
else:
    df_hist = pd.DataFrame([
        {
            "Heure": h["timestamp"].strftime("%H:%M:%S"),
            "T (°C)": h["features"]["t"],
            "Humidité (%)": h["features"]["hu"],
            "Vent (m/s)": h["features"]["ff"],
            "Précip. (mm)": h["features"]["precip"],
            "Risque (%)": f"{h['proba']*100:.1f}",
            "Niveau": get_risk_label(h["proba"]),
        }
        for h in reversed(st.session_state.history)
    ])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if st.button("Vider l'historique"):
        st.session_state.history = []
        st.rerun()