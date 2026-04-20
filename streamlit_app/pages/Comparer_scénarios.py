"""Page : Comparer plusieurs scénarios côte à côte."""
import streamlit as st
import plotly.graph_objects as go
from utils import (
    inject_custom_css, render_sidebar,
    predict_risk, get_risk_color, get_risk_label
)

st.set_page_config(page_title="Comparaison", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar()

st.markdown('<div class="hero-title">Comparer trois scénarios</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Évaluez en parallèle l\'impact de conditions météorologiques contrastées</div>',
    unsafe_allow_html=True,
)

SCENARIOS = [
    ("Canicule sèche",       {"dd": 180, "ff": 15.0, "t": 38.0, "td": 8.0,  "precip": 0.0,  "hu": 18}),
    ("Conditions normales",  {"dd": 180, "ff": 5.0,  "t": 25.0, "td": 15.0, "precip": 2.0,  "hu": 50}),
    ("Pluie et fraîcheur",   {"dd": 270, "ff": 3.0,  "t": 18.0, "td": 17.0, "precip": 15.0, "hu": 90}),
]

cols = st.columns(3, gap="large")
results = []

for i, (col, (title, defaults)) in enumerate(zip(cols, SCENARIOS)):
    with col:
        features = {}
        with st.expander(f"{title} — paramètres", expanded=False):
            features["dd"]     = st.slider("Direction vent (°)",     0, 360,  defaults["dd"],     key=f"dd_{i}")
            features["ff"]     = st.slider("Vitesse vent (m/s)",     0.0, 30.0, defaults["ff"],   step=0.5, key=f"ff_{i}")
            features["t"]      = st.slider("Température (°C)",       -10.0, 45.0, defaults["t"],  step=0.5, key=f"t_{i}")
            features["td"]     = st.slider("Point rosée (°C)",       -10.0, 45.0, defaults["td"], step=0.5, key=f"td_{i}")
            features["precip"] = st.slider("Précipitations (mm)",    0.0, 50.0, defaults["precip"], step=0.5, key=f"precip_{i}")
            features["hu"]     = st.slider("Humidité (%)",           0, 100,  defaults["hu"],     key=f"hu_{i}")

        try:
            result = predict_risk(features)
            proba = result["probabilite"]
            color = get_risk_color(proba)
            label = get_risk_label(proba)

            st.markdown(f"""
            <div class='kpi-card' style='text-align:center; padding:1.8rem 1.4rem;'>
              <div style='font-size:0.78rem; color:#6b7a8c; text-transform:uppercase;
                          letter-spacing:0.08em; font-weight:500; margin-bottom:12px;'>
                {title}
              </div>
              <div style='font-size:2.8rem; font-weight:600; color:{color}; line-height:1; letter-spacing:-0.02em;'>
                {proba*100:.1f}<span style='font-size:1.4rem; color:#9aa7b8;'>%</span>
              </div>
              <div style='margin-top:10px; font-size:0.88rem; color:#1a2b3c; font-weight:500;'>
                {label}
              </div>
              <div style='margin-top:14px; padding-top:12px; border-top:1px solid rgba(107,122,140,0.12);
                          display:flex; justify-content:space-between; font-size:0.82rem; color:#6b7a8c;'>
                <span>{features['t']:.0f}°C</span>
                <span>{features['hu']}% HR</span>
                <span>{features['ff']:.1f} m/s</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            results.append({"title": title, "proba": proba, "color": color, "features": features})

        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("<hr>", unsafe_allow_html=True)

if len(results) == 3:
    st.markdown('<div class="section-title">Comparaison graphique</div>', unsafe_allow_html=True)

    fig = go.Figure(data=[
        go.Bar(
            x=[r["title"] for r in results],
            y=[r["proba"] * 100 for r in results],
            marker_color=[r["color"] for r in results],
            text=[f"{r['proba']*100:.1f}%" for r in results],
            textposition="outside",
            textfont=dict(size=14, color="#1a2b3c", family="Inter"),
            marker_line_width=0,
            width=0.5,
        )
    ])
    fig.update_layout(
        yaxis=dict(title=None, range=[0, 110], gridcolor="rgba(107,122,140,0.12)", zeroline=False),
        xaxis=dict(title=None, tickfont=dict(size=12, color="#6b7a8c")),
        showlegend=False,
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insight
    probas = [r["proba"] for r in results]
    max_i, min_i = probas.index(max(probas)), probas.index(min(probas))
    diff = (max(probas) - min(probas)) * 100

    st.markdown(f"""
    <div class='card' style='padding:1rem 1.3rem; font-size:0.92rem; color:#1a2b3c; line-height:1.6;'>
      Le scénario <b style='color:{results[max_i]["color"]};'>{results[max_i]['title']}</b> présente le risque le plus élevé
      ({max(probas)*100:.1f}%), contre <b style='color:{results[min_i]["color"]};'>{min(probas)*100:.1f}%</b>
      pour <b>{results[min_i]['title']}</b>. Écart : <b>{diff:.1f} points</b>.
    </div>
    """, unsafe_allow_html=True)