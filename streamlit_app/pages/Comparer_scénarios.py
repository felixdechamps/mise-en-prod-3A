"""Page : Comparer plusieurs scénarios côte à côte."""
import streamlit as st
import plotly.graph_objects as go
from utils import (
    inject_custom_css, render_sidebar,
    predict_risk, get_risk_color, get_risk_label
)

st.set_page_config(page_title="🔬 Comparer scénarios", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar()

st.title("🔬 Comparer des scénarios")
st.markdown("Testez simultanément **3 scénarios météo** pour comprendre l'impact des différents facteurs.")

SCENARIOS = [
    ("☀️ Canicule sèche", {"dd": 180, "ff": 15.0, "t": 38.0, "td": 8.0, "precip": 0.0, "hu": 18}),
    ("🌿 Conditions normales", {"dd": 180, "ff": 5.0, "t": 25.0, "td": 15.0, "precip": 2.0, "hu": 50}),
    ("🌧️ Pluie et fraîcheur", {"dd": 270, "ff": 3.0, "t": 18.0, "td": 17.0, "precip": 15.0, "hu": 90}),
]

cols = st.columns(3)
results = []

for i, (col, (title, defaults)) in enumerate(zip(cols, SCENARIOS)):
    with col:
        st.markdown(f"### {title}")
        
        with st.expander("⚙️ Paramètres", expanded=False):
            dd = st.slider(f"Direction vent (°)", 0, 360, defaults["dd"], key=f"dd_{i}")
            ff = st.slider(f"Vitesse vent (m/s)", 0.0, 30.0, defaults["ff"], step=0.5, key=f"ff_{i}")
            t = st.slider(f"Température (°C)", -10.0, 45.0, defaults["t"], step=0.5, key=f"t_{i}")
            td = st.slider(f"Point rosée (°C)", -10.0, 45.0, defaults["td"], step=0.5, key=f"td_{i}")
            precip = st.slider(f"Précipitations (mm)", 0.0, 50.0, defaults["precip"], step=0.5, key=f"precip_{i}")
            hu = st.slider(f"Humidité (%)", 0, 100, defaults["hu"], key=f"hu_{i}")
        
        features = {"dd": dd, "ff": ff, "t": t, "td": td, "precip": precip, "hu": hu}
        
        try:
            result = predict_risk(features)
            proba = result["probabilite"]
            color = get_risk_color(proba)
            label = get_risk_label(proba)
            
            st.markdown(f"""
            <div style='padding: 1.5rem; background: white; border-radius: 12px;
                        border-top: 6px solid {color}; text-align: center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
              <p style='margin: 0; color: #666; font-size: 14px;'>Probabilité</p>
              <h1 style='margin: 0.5rem 0; color: {color}; font-size: 3rem;'>{proba*100:.1f}%</h1>
              <p style='margin: 0; font-size: 18px;'>{label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            results.append({"title": title, "proba": proba, "color": color, "features": features})
        
        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()

# Graphique de comparaison
if len(results) == 3:
    st.markdown("### 📊 Comparaison visuelle")
    
    fig = go.Figure(data=[
        go.Bar(
            x=[r["title"] for r in results],
            y=[r["proba"] * 100 for r in results],
            marker_color=[r["color"] for r in results],
            text=[f"{r['proba']*100:.1f}%" for r in results],
            textposition="outside",
        )
    ])
    fig.update_layout(
        yaxis_title="Probabilité d'incendie (%)",
        yaxis_range=[0, 100],
        showlegend=False,
        height=400,
        plot_bgcolor="#FAFAF7",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight automatique
    probas = [r["proba"] for r in results]
    max_i = probas.index(max(probas))
    min_i = probas.index(min(probas))
    diff = (max(probas) - min(probas)) * 100
    
    st.info(f"""
    💡 **Analyse** : le scénario **{results[max_i]['title']}** présente le risque le plus élevé 
    ({max(probas)*100:.1f}%), tandis que **{results[min_i]['title']}** présente le risque le plus faible 
    ({min(probas)*100:.1f}%). La différence est de **{diff:.1f} points** de pourcentage.
    """)