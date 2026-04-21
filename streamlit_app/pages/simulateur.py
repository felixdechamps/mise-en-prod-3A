"""Page : Simulateur — tester un ou plusieurs scénarios météo."""
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
    '<div class="hero-subtitle">Testez un scénario météo ou comparez-en plusieurs côte à côte</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Mode : simple ou comparaison
# ---------------------------------------------------------------------------
mode = st.radio(
    "Mode",
    ["Scénario unique", "Comparer plusieurs scénarios"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Presets (partagés)
# ---------------------------------------------------------------------------
PRESETS = {
    "Canicule sèche":      {"dd": 180, "ff": 15.0, "t": 38.0, "td": 8.0,  "precip": 0.0,  "hu": 18},
    "Conditions normales": {"dd": 180, "ff": 5.0,  "t": 25.0, "td": 15.0, "precip": 2.0,  "hu": 50},
    "Pluie d'été":         {"dd": 270, "ff": 5.0,  "t": 22.0, "td": 19.0, "precip": 15.0, "hu": 85},
    "Hiver froid":         {"dd": 90,  "ff": 3.0,  "t": 5.0,  "td": 2.0,  "precip": 2.0,  "hu": 75},
}

DEFAULT = {"dd": 180, "ff": 5.0, "t": 25.0, "td": 15.0, "precip": 0.0, "hu": 40}


def render_sliders(key_suffix: str, defaults: dict) -> dict:
    """Affiche le bloc de 6 sliders et retourne les features."""
    return {
        "dd":     st.slider("Direction du vent (°)",  0,      360,  defaults["dd"],
                           help="0 = Nord · 90 = Est · 180 = Sud · 270 = Ouest",
                           key=f"dd_{key_suffix}"),
        "ff":     st.slider("Vitesse du vent (m/s)",  0.0,    30.0, defaults["ff"],   step=0.5, key=f"ff_{key_suffix}"),
        "t":      st.slider("Température (°C)",       -10.0,  45.0, defaults["t"],    step=0.5, key=f"t_{key_suffix}"),
        "td":     st.slider("Point de rosée (°C)",    -10.0,  45.0, defaults["td"],   step=0.5, key=f"td_{key_suffix}"),
        "precip": st.slider("Précipitations (mm)",    0.0,    50.0, defaults["precip"], step=0.5, key=f"precip_{key_suffix}"),
        "hu":     st.slider("Humidité relative (%)",  0,      100,  defaults["hu"],   key=f"hu_{key_suffix}"),
    }


def render_result_card(title: str, features: dict, proba: float):
    """Carte de résultat compacte (pour le mode comparaison)."""
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


# ===========================================================================
# MODE 1 : SCÉNARIO UNIQUE
# ===========================================================================
if mode == "Scénario unique":

    # --- Presets ---
    st.markdown('<div class="section-title">Scénarios types</div>', unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESETS))
    preset = None
    for col, (name, values) in zip(preset_cols, PRESETS.items()):
        if col.button(name, use_container_width=True, key=f"preset_single_{name}"):
            preset = values

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Form + Résultat ---
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Paramètres</div>', unsafe_allow_html=True)
        features = render_sliders("single", preset if preset else DEFAULT)
        if features["td"] > features["t"]:
            st.warning("Le point de rosée dépasse la température — valeur inhabituelle.")

    with col_result:
        st.markdown('<div class="section-title">Prédiction</div>', unsafe_allow_html=True)
        try:
            result = predict_risk(features)
            proba = result["probabilite"]
            color = get_risk_color(proba)
            label = get_risk_label(proba)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%", 'font': {'size': 48, 'color': '#1a2b3c', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 0,
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

            # Verdict
            st.markdown(f"""
            <div style='text-align:center; margin-top:-1rem;'>
              <div style='font-size:0.78rem; color:#6b7a8c; text-transform:uppercase;
                          letter-spacing:0.08em; font-weight:500; margin-bottom:4px;'>
                Niveau de risque
              </div>
              <div style='font-size:1.3rem; color:{color}; font-weight:600;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

            if st.button("Enregistrer ce scénario", use_container_width=True, key="save_single"):
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "features": features,
                    "proba": proba,
                })
                st.toast("Scénario enregistré", icon="✓")

        except Exception as e:
            st.error(f"Erreur API : {e}")

    # --- Historique ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Historique de session</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div style="color:#6b7a8c; font-size:0.9rem; padding:0.5rem 0;">'
            'Aucun scénario enregistré. Utilisez le bouton ci-dessus pour en sauvegarder.</div>',
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

        c1, _ = st.columns([1, 4])
        if c1.button("Vider l'historique"):
            st.session_state.history = []
            st.rerun()


# ===========================================================================
# MODE 2 : COMPARAISON
# ===========================================================================
else:
    # Nombre de scénarios
    n_scenarios = st.slider("Nombre de scénarios à comparer", 2, 4, 3, key="n_comp")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Presets par défaut pour chaque colonne
    default_presets = list(PRESETS.items())

    cols = st.columns(n_scenarios, gap="large")
    results = []

    for i, col in enumerate(cols):
        with col:
            preset_name, preset_values = default_presets[i % len(default_presets)]

            with st.expander(f"{preset_name} — paramètres", expanded=False):
                features = render_sliders(f"comp_{i}", preset_values)

            try:
                result = predict_risk(features)
                proba = result["probabilite"]
                render_result_card(preset_name, features, proba)
                results.append({
                    "title": preset_name,
                    "proba": proba,
                    "color": get_risk_color(proba),
                    "features": features,
                })
            except Exception as e:
                st.error(f"Erreur : {e}")

    # --- Graphique comparatif ---
    if len(results) >= 2:
        st.markdown("<hr>", unsafe_allow_html=True)
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
