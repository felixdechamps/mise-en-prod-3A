"""Page : Enjeux et contexte — pourquoi surveiller les feux de forêt."""
import streamlit as st
import plotly.graph_objects as go
from utils import inject_custom_css, render_sidebar

st.set_page_config(page_title="Enjeux", page_icon="🔥", layout="wide")
inject_custom_css()
render_sidebar()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Pourquoi anticiper les feux ?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Le risque incendie devient un enjeu majeur en France — '
    'anticiper quelques heures à l\'avance peut faire la différence.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chiffres clés
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Quelques chiffres</div>', unsafe_allow_html=True)

def big_stat(col, value, label, source=""):
    col.markdown(f"""
    <div class='kpi-card' style='padding:1.6rem 1.4rem;'>
      <div style='font-size:2.4rem; font-weight:700; color:#1a2b3c;
                  letter-spacing:-0.02em; line-height:1; margin-bottom:8px;'>
        {value}
      </div>
      <div style='font-size:0.92rem; color:#1a2b3c; font-weight:500; line-height:1.4;'>
        {label}
      </div>
      <div style='font-size:0.75rem; color:#9aa7b8; margin-top:8px;'>
        {source}
      </div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
big_stat(c1, "66 000 ha",
         "Surface brûlée en France en 2022, année record",
         "Source : Ministère de l'Intérieur")
big_stat(c2, "+80 %",
         "Hausse prévue des surfaces brûlées d'ici 2050",
         "Source : vie-publique.fr")
big_stat(c3, "×4",
         "Fréquence prévue des mégafeux (>2500 ha) d'ici 2100 en Europe",
         "Source : Oxfam France")
big_stat(c4, "~500 M€",
         "Coût estimé des feux pour les collectivités (année 2024)",
         "Source : EFFIS / collectivités")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Évolution des surfaces brûlées
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Une tendance à la hausse</div>', unsafe_allow_html=True)

col_chart, col_text = st.columns([3, 2], gap="large")

with col_chart:
    # Données simplifiées issues des sources citées (moyenne approximative)
    annees = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    hectares = [5800, 16700, 11000, 28000, 66000, 22000, 12300]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annees,
        y=hectares,
        marker_color=["#c8d4e3" if h < 30000 else "#DC2626" for h in hectares],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{y:,} hectares<extra></extra>",
        text=[f"{h/1000:.0f}k" for h in hectares],
        textposition="outside",
        textfont=dict(size=12, color="#1a2b3c"),
        width=0.55,
    ))
    fig.update_layout(
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Inter, sans-serif", color="#1a2b3c"),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(
            title="Hectares brûlés",
            gridcolor="rgba(107,122,140,0.12)",
            zeroline=False,
            tickformat=",d",
        ),
        xaxis=dict(title=None, tickmode="linear"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_text:
    st.markdown("""
    <div class='card' style='height:100%;'>
      <div style='color:#1a2b3c; line-height:1.7; font-size:0.95rem;'>
        La surface brûlée varie fortement d'une année à l'autre selon les conditions
        météo, mais <b>2022 a battu tous les records</b> avec près de 66 000 hectares
        partis en fumée — six fois la moyenne historique.
      </div>
      <div style='margin-top:1rem; color:#6b7a8c; line-height:1.6; font-size:0.88rem;'>
        Au-delà des pics, trois tendances de fond sont documentées par Météo-France
        et l'ONF : extension géographique (Jura, Bretagne, Normandie désormais
        concernés), allongement de la saison à risque (mai à octobre au lieu de juin
        à septembre), et intensification des événements extrêmes.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Ce qui est en jeu
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Ce qui est en jeu</div>', unsafe_allow_html=True)

enjeux = [
    ("Vies humaines",
     "Évacuations, pertes, intoxications. En 2024, 3 000 vacanciers évacués d'un camping "
     "à Canet-en-Roussillon en une nuit. Les mégafeux tuent — 123 morts au Chili en 2024, "
     "29 à Los Angeles en 2025."),
    ("Écosystèmes",
     "Destruction d'habitats, perte de biodiversité, déstabilisation des sols. "
     "Une forêt brûlée met plusieurs décennies à retrouver sa maturité écologique, "
     "quand elle y parvient."),
    ("Climat",
     "Les incendies relâchent le CO₂ stocké par les forêts et réduisent leur capacité "
     "d'absorption future. C'est une boucle de rétroaction qui aggrave le réchauffement."),
    ("Économie locale",
     "Exploitation du bois, tourisme, agriculture. Les petites communes rurales sont "
     "les plus touchées — certaines régions supportent 50 à 100 M€ de coûts par saison."),
]

for i in range(0, len(enjeux), 2):
    c1, c2 = st.columns(2, gap="medium")
    for col, (title, desc) in zip((c1, c2), enjeux[i:i+2]):
        col.markdown(f"""
        <div class='card' style='margin-bottom:1rem; height:calc(100% - 1rem);'>
          <div style='font-size:1.05rem; font-weight:600; color:#1a2b3c;
                      letter-spacing:-0.01em; margin-bottom:10px;'>
            {title}
          </div>
          <div style='color:#4a5a6c; line-height:1.6; font-size:0.9rem;'>
            {desc}
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pourquoi un modèle ML ?
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Pourquoi un modèle de prédiction ?</div>', unsafe_allow_html=True)

st.markdown("""
<div class='card'>
  <div style='color:#1a2b3c; line-height:1.7; font-size:0.95rem; margin-bottom:1rem;'>
    Les pompiers, l'ONF et Météo-France utilisent déjà des indices de risque
    (IFM — Indice Forêt Météo). Un modèle de machine learning apporte trois choses
    complémentaires :
  </div>
  <div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:1.5rem; margin-top:1rem;'>
    <div>
      <div style='font-size:0.75rem; color:#6b7a8c; text-transform:uppercase;
                  letter-spacing:0.08em; font-weight:600; margin-bottom:6px;'>Granularité</div>
      <div style='color:#4a5a6c; line-height:1.55; font-size:0.88rem;'>
        Une prédiction par ville, par heure, plutôt qu'une carte départementale journalière.
      </div>
    </div>
    <div>
      <div style='font-size:0.75rem; color:#6b7a8c; text-transform:uppercase;
                  letter-spacing:0.08em; font-weight:600; margin-bottom:6px;'>Actionabilité</div>
      <div style='color:#4a5a6c; line-height:1.55; font-size:0.88rem;'>
        Une probabilité chiffrée (0 à 100%) plus directement exploitable qu'un code couleur.
      </div>
    </div>
    <div>
      <div style='font-size:0.75rem; color:#6b7a8c; text-transform:uppercase;
                  letter-spacing:0.08em; font-weight:600; margin-bottom:6px;'>Adaptabilité</div>
      <div style='color:#4a5a6c; line-height:1.55; font-size:0.88rem;'>
        Un modèle se réentraîne au fil des saisons, intégrant les nouvelles dynamiques climatiques.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# Avertissement final
st.markdown("""
<div class='card' style='border-left:3px solid #F59E0B; padding:1rem 1.3rem;'>
  <div style='font-size:0.72rem; color:#F59E0B; text-transform:uppercase;
              letter-spacing:0.08em; font-weight:600; margin-bottom:6px;'>
    Honnêteté scientifique
  </div>
  <div style='color:#1a2b3c; font-size:0.9rem; line-height:1.6;'>
    Le modèle présenté ici est une démonstration pédagogique. Les indices officiels
    utilisés par la sécurité civile (IFM, BMS feux) intègrent bien plus de variables
    (état de la végétation, historique, topographie) et restent la référence pour
    toute décision opérationnelle.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.caption(
    "Sources : Ministère de l'Intérieur · EFFIS (European Forest Fire Information System) · "
    "vie-publique.fr · Oxfam France · Météo-France"
)
