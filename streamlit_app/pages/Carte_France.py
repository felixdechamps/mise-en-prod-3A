"""Page : Carte interactive de France avec les risques en temps réel."""
import datetime
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

st.set_page_config(page_title="Carte France", page_icon="🗺️", layout="wide")
inject_custom_css()
render_sidebar()

# ---------------------------------------------------------------------------
# Style : palette "app météo", dégradé ciel, cards glassmorphism
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #e8f1fb 0%, #f5f7fa 45%, #fef3e8 100%);
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
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
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a2b3c;
        margin: 0.5rem 0 1rem 0;
        letter-spacing: -0.01em;
    }
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
    hr { border: none; border-top: 1px solid rgba(107, 122, 140, 0.15); margin: 2rem 0; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Risque incendie — France</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Prévision temps réel basée sur les conditions météorologiques actuelles</div>',
    unsafe_allow_html=True,
)


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


with st.sidebar:
    st.divider()
    st.markdown("### Affichage")
    map_style = st.selectbox(
        "Style de carte",
        ["CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"],
        index=0,
    )
    show_heatmap = st.checkbox("Heatmap", value=True)
    show_markers = st.checkbox("Marqueurs", value=True)

with st.spinner("Récupération des conditions météo..."):
    risques, errors = compute_all_risks()

if errors:
    with st.expander(f"{len(errors)} erreur(s) rencontrée(s)"):
        for e in errors:
            st.warning(e)

if not risques:
    st.error("Aucune donnée disponible. Vérifiez la connexion à l'API.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
probas = [d["prediction"]["probabilite"] for d in risques.values()]
nb_eleves = sum(1 for p in probas if p >= 0.5)
nb_moderes = sum(1 for p in probas if 0.2 <= p < 0.5)
nb_faibles = sum(1 for p in probas if p < 0.2)

col1, col2, col3, col4 = st.columns(4)

def kpi_card(col, label, value, dot_color=None):
    dot = f'<span class="kpi-dot" style="background:{dot_color};"></span>' if dot_color else ""
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{dot}{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(col1, "Villes suivies", len(risques))
kpi_card(col2, "Risque élevé", nb_eleves, "#DC2626")
kpi_card(col3, "Risque modéré", nb_moderes, "#F59E0B")
kpi_card(col4, "Risque faible", nb_faibles, "#16A34A")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Map + ranking
# ---------------------------------------------------------------------------
col_map, col_rank = st.columns([2, 1])

with col_map:
    st.markdown('<div class="section-title">Carte interactive</div>', unsafe_allow_html=True)

    m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles=map_style)

    if show_heatmap:
        heat_data = [
            [d["lat"], d["lon"], d["prediction"]["probabilite"]]
            for d in risques.values()
        ]
        HeatMap(heat_data, radius=45, blur=30, min_opacity=0.3).add_to(m)

    if show_markers:
        for ville, d in risques.items():
            proba = d["prediction"]["probabilite"]
            color = get_risk_color(proba)
            w = d["weather"]

            popup_html = f"""
            <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; min-width: 210px; color:#1a2b3c;'>
              <div style='font-size:15px; font-weight:600; margin-bottom:10px;'>{ville}</div>
              <table style='width: 100%; font-size: 13px; color:#4a5a6c;'>
                <tr><td style='padding:3px 0;'>Température</td><td style='text-align:right;'><b style='color:#1a2b3c;'>{w['t']:.1f}°C</b></td></tr>
                <tr><td style='padding:3px 0;'>Humidité</td><td style='text-align:right;'><b style='color:#1a2b3c;'>{w['hu']:.0f}%</b></td></tr>
                <tr><td style='padding:3px 0;'>Vent</td><td style='text-align:right;'><b style='color:#1a2b3c;'>{w['ff']:.1f} m/s</b></td></tr>
                <tr><td style='padding:3px 0;'>Précipitations</td><td style='text-align:right;'><b style='color:#1a2b3c;'>{w['precip']:.1f} mm</b></td></tr>
              </table>
              <div style='margin-top:10px; padding-top:10px; border-top:1px solid #e5e9ef;'>
                <div style='font-size:22px; font-weight:600; color:{color};'>{proba*100:.1f}%</div>
                <div style='font-size:12px; color:#6b7a8c; text-transform:uppercase; letter-spacing:0.05em;'>{get_risk_label(proba)}</div>
              </div>
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
    st.markdown('<div class="section-title">Classement</div>', unsafe_allow_html=True)

    sorted_villes = sorted(
        risques.items(),
        key=lambda x: x[1]["prediction"]["probabilite"],
        reverse=True,
    )

    ranking_html = ""
    for rang, (ville, d) in enumerate(sorted_villes, start=1):
        proba = d["prediction"]["probabilite"] * 100
        color = get_risk_color(d["prediction"]["probabilite"])
        ranking_html += f"""
        <div class="rank-row" style="--accent:{color};">
          <div>
            <span class="rank-num">{rang:02d}</span>
            <span class="rank-city">{ville}</span>
          </div>
          <span class="rank-value" style="color:{color};">{proba:.1f}%</span>
        </div>
        """
    st.markdown(ranking_html, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Comparaison par ville</div>', unsafe_allow_html=True)

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
    hover_data=["Température (°C)", "Humidité (%)"],
)
fig.update_layout(
    height=500,
    showlegend=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#1a2b3c"),
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(gridcolor="rgba(107,122,140,0.12)", zeroline=False, title=None),
    yaxis=dict(gridcolor="rgba(107,122,140,0)", title=None),
    coloraxis_showscale=False,
)
fig.update_traces(
    marker_line_width=0,
    hovertemplate="<b>%{y}</b><br>Risque : %{x:.1f}%<extra></extra>",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Dernière mise à jour : {datetime.datetime.now().strftime('%H:%M:%S')}  ·  Mise en cache 10 min"
)