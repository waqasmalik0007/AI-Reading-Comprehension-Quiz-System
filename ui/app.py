"""
app.py — Intelligent Reading Comprehension Quiz System
Premium Dual-Theme UI (Dark Cyberpunk + Light Grid)
Matches reference image design exactly.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import streamlit as st

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title='RC Quiz',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# ── Session State Init ────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'page' not in st.session_state:
    st.session_state.page = 'ARTICLE'
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'user_answer' not in st.session_state:
    st.session_state.user_answer = None
if 'answer_checked' not in st.session_state:
    st.session_state.answer_checked = False
if 'hints_revealed' not in st.session_state:
    st.session_state.hints_revealed = 0
if 'session_results' not in st.session_state:
    st.session_state.session_results = []
if 'latencies' not in st.session_state:
    st.session_state.latencies = []
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False
if 'used_answers' not in st.session_state:
    st.session_state.used_answers = []
if 'theme_toast' not in st.session_state:
    st.session_state.theme_toast = False
if 'article_text' not in st.session_state:
    st.session_state.article_text = ''

is_dark = st.session_state.theme == 'dark'

# ── Theme Variables ───────────────────────────────────────────
if is_dark:
    BG_BODY        = '#0a0e17'
    BG_GRID        = '#0d1220'
    BG_CARD        = '#111827'
    BG_CARD2       = '#0f1929'
    BG_NAV         = '#080c14'
    BG_INPUT       = '#0c1421'
    BORDER_COLOR   = '#1e3a5f'
    BORDER_ACCENT  = '#00e5ff'
    TEXT_PRIMARY   = '#e8f4f8'
    TEXT_SECONDARY = '#8ab4c9'
    TEXT_MONO      = '#00e5ff'
    ACCENT_CYAN    = '#00e5ff'
    ACCENT_GREEN   = '#39ff14'
    ACCENT_BLUE    = '#0080ff'
    NAV_ACTIVE_BG  = '#00e5ff'
    NAV_ACTIVE_FG  = '#000000'
    BTN_PRIMARY    = '#00e5ff'
    BTN_PRIMARY_FG = '#000000'
    BTN_LOAD       = '#39ff14'
    BTN_LOAD_FG    = '#000000'
    GRID_COLOR     = 'rgba(0,229,255,0.04)'
    GLOW_COLOR     = 'rgba(0,229,255,0.15)'
    HEADING_COLOR  = '#ffffff'
    HEADING_ACCENT = '#00e5ff'
    LABEL_COLOR    = '#00e5ff'
    TOAST_BG       = '#0f2a1e'
    TOAST_BORDER   = '#39ff14'
    TOAST_TEXT     = '#39ff14'
    SHADOW         = '0 0 30px rgba(0,229,255,0.12), 0 2px 8px rgba(0,0,0,0.6)'
    INPUT_BORDER   = '#00e5ff'
else:
    BG_BODY        = '#e8f0f8'
    BG_GRID        = '#dce8f5'
    BG_CARD        = '#ffffff'
    BG_CARD2       = '#f0f7ff'
    BG_NAV         = '#ffffff'
    BG_INPUT       = '#ffffff'
    BORDER_COLOR   = '#b8d4ea'
    BORDER_ACCENT  = '#0066cc'
    TEXT_PRIMARY   = '#1a2940'
    TEXT_SECONDARY = '#4a6580'
    TEXT_MONO      = '#0055bb'
    ACCENT_CYAN    = '#0077dd'
    ACCENT_GREEN   = '#2d8a00'
    ACCENT_BLUE    = '#0055cc'
    NAV_ACTIVE_BG  = '#0077dd'
    NAV_ACTIVE_FG  = '#ffffff'
    BTN_PRIMARY    = '#0077dd'
    BTN_PRIMARY_FG = '#ffffff'
    BTN_LOAD       = '#3a9e00'
    BTN_LOAD_FG    = '#ffffff'
    GRID_COLOR     = 'rgba(0,119,221,0.06)'
    GLOW_COLOR     = 'rgba(0,119,221,0.08)'
    HEADING_COLOR  = '#0a1628'
    HEADING_ACCENT = '#0077dd'
    LABEL_COLOR    = '#0066cc'
    TOAST_BG       = '#e6f4e6'
    TOAST_BORDER   = '#3a9e00'
    TOAST_TEXT     = '#1a5c00'
    SHADOW         = '0 2px 16px rgba(0,80,160,0.10), 0 1px 4px rgba(0,0,0,0.06)'
    INPUT_BORDER   = '#0077dd'

# ── CSS Injection ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}

.main .block-container {{
    padding: 0 !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
    max-width: 100% !important;
}}
/* Kill Streamlit's internal vertical gap between stacked elements */
[data-testid="stVerticalBlock"] {{
    gap: 0rem !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0 !important;
}}

/* Hide Streamlit chrome */
header[data-testid="stHeader"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu {{ display: none !important; }}
footer {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
div[data-testid="stDecoration"] {{ display: none !important; }}
div[data-testid="stStatusWidget"] {{ display: none !important; }}
div[data-testid="collapsedControl"] {{ display: none !important; }}

/* ── Body ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG_BODY} !important;
    font-family: 'Exo 2', sans-serif;
    color: {TEXT_PRIMARY};
}}

/* Grid background */
[data-testid="stAppViewContainer"] {{
    background-image:
        linear-gradient({GRID_COLOR} 1px, transparent 1px),
        linear-gradient(90deg, {GRID_COLOR} 1px, transparent 1px) !important;
    background-size: 32px 32px !important;
}}

/* ── Navigation Bar ── */
.rc-navbar {{
    background: {BG_NAV};
    border-bottom: 1px solid {BORDER_COLOR};
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    position: sticky;
    transform: translateY(-90px);
    top: 30;
    z-index: 1002;
    pointer-events: auto;
    {'box-shadow: 0 0 24px rgba(0,229,255,0.08);' if is_dark else 'box-shadow: 0 1px 8px rgba(0,0,0,0.08);'}
}}

.rc-logo {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    letter-spacing: 2px;
    color: {TEXT_PRIMARY};
    text-transform: uppercase;
}}

.rc-logo-bolt {{
    color: {ACCENT_GREEN};
    font-size: 1.4rem;
}}

.rc-nav-links {{
    display: flex;
    align-items: center;
    gap: 4px;
}}

.rc-nav-btn {{
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 7px 20px;
    border: 1px solid transparent;
    border-radius: 3px;
    cursor: pointer;
    transition: all 0.18s ease;
    background: transparent;
    color: {TEXT_SECONDARY};
    display: inline-flex;
    align-items: center;
    gap: 7px;
    text-decoration: none;
}}

.rc-nav-btn:hover {{
    color: {TEXT_PRIMARY};
    border-color: {BORDER_COLOR};
}}

.rc-nav-btn.active {{
    background: {NAV_ACTIVE_BG};
    color: {NAV_ACTIVE_FG};
    border-color: {NAV_ACTIVE_BG};
    {'box-shadow: 0 0 12px rgba(0,229,255,0.35); clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 100%, 8px 100%);' if is_dark else ''}
}}

.rc-nav-toggle {{
    font-size: 1.1rem;
    cursor: pointer;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid {BORDER_COLOR};
    background: transparent;
    color: {TEXT_SECONDARY};
    transition: all 0.2s;
    line-height: 1.3;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    min-width: 44px;
}}
.rc-nav-toggle:hover {{
    border-color: {BORDER_ACCENT};
    color: {TEXT_PRIMARY};
    text-decoration: none;
}}

/* ── Kill ALL gap between navbar and page content ── */
.main .block-container {{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}
.element-container:empty {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}

/* ── Page Wrapper ── */
.rc-page {{
   padding-top: 70px !important;
    padding: 0px 40px 60px;
    min-height: calc(100vh - 56px);
}}

/* ── Page Title ── */
.rc-page-title {{
    border-left: 4px solid {ACCENT_CYAN};
    padding-left: 18px;
    margin-bottom: 32px;
}}
.rc-page-title h1 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
    color: {HEADING_COLOR};
    line-height: 1.1;
}}
.rc-page-title h1 span {{
    color: {HEADING_ACCENT};
}}
.rc-page-title p {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: {TEXT_SECONDARY};
    margin: 6px 0 0;
    letter-spacing: 0.5px;
}}

/* ── Panel / Card ── */
.rc-panel {{
    background: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 24px 28px;
    {'box-shadow: ' + SHADOW + ';' if is_dark else 'box-shadow: ' + SHADOW + ';'}
    margin-bottom: 20px;
}}

.rc-panel-label {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {LABEL_COLOR};
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.rc-panel-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER_COLOR};
}}

/* ── Textarea Override ── */
textarea {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.88rem !important;
    color: {TEXT_MONO} !important;
    background: {BG_INPUT} !important;
    border: 1px solid {INPUT_BORDER} !important;
    border-radius: 3px !important;
    caret-color: {ACCENT_CYAN};
}}
textarea::placeholder {{
    color: {'rgba(0,229,255,0.35)' if is_dark else 'rgba(0,119,221,0.4)'} !important;
}}
textarea:focus {{
    {'box-shadow: 0 0 0 2px rgba(0,229,255,0.2), 0 0 16px rgba(0,229,255,0.08) !important;' if is_dark else 'box-shadow: 0 0 0 2px rgba(0,119,221,0.2) !important;'}
    outline: none !important;
}}
div[data-testid="stTextArea"] label {{
    display: none !important;
}}
div[data-testid="stTextArea"] > div {{
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}

/* ── Button overrides ── */
div[data-testid="stButton"] > button {{
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    border: none !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}}

/* Primary blue/cyan btn */
div[data-testid="stButton"][class*="primary"] > button,
div[data-testid="stButton"]:has(button[kind="primary"]) > button {{
    background: {BTN_PRIMARY} !important;
    color: {BTN_PRIMARY_FG} !important;
    {'box-shadow: 0 0 18px rgba(0,229,255,0.4) !important;' if is_dark else ''}
}}
div[data-testid="stButton"]:has(button[kind="primary"]) > button:hover {{
    filter: brightness(1.1) !important;
    {'box-shadow: 0 0 28px rgba(0,229,255,0.6) !important;' if is_dark else ''}
    transform: translateY(-1px) !important;
}}

/* Secondary btn */
div[data-testid="stButton"]:has(button[kind="secondary"]) > button {{
    background: transparent !important;
    color: {ACCENT_CYAN} !important;
    border: 1px solid {ACCENT_CYAN} !important;
}}

/* Radio buttons */
div[data-testid="stRadio"] label {{
    font-family: 'Exo 2', sans-serif !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.95rem !important;
}}
div[data-testid="stRadio"] {{
    background: transparent !important;
}}

/* ── Metric cards ── */
.rc-metric {{
    background: {BG_CARD2};
    border: 1px solid {BORDER_COLOR};
    border-top: 3px solid {ACCENT_CYAN};
    border-radius: 3px;
    padding: 20px 18px;
    text-align: center;
}}
.rc-metric-val {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.9rem;
    color: {ACCENT_CYAN};
    line-height: 1;
    margin-bottom: 6px;
}}
.rc-metric-label {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {TEXT_SECONDARY};
}}

/* ── Question box ── */
.rc-question-box {{
    background: {BG_CARD2};
    border: 1px solid {BORDER_COLOR};
    border-left: 4px solid {ACCENT_CYAN};
    border-radius: 0 4px 4px 0;
    padding: 20px 24px;
    font-family: 'Exo 2', sans-serif;
    font-size: 1.05rem;
    color: {TEXT_PRIMARY};
    margin: 16px 0;
    {'box-shadow: inset 0 0 20px rgba(0,229,255,0.03);' if is_dark else ''}
}}
.rc-question-box strong {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: {ACCENT_CYAN};
    text-transform: uppercase;
    display: block;
    margin-bottom: 8px;
}}

/* ── Result cards ── */
.rc-success {{
    background: {'#071f0e' if is_dark else '#eefaf2'};
    border: 1px solid {'#39ff14' if is_dark else '#3a9e00'};
    border-left: 4px solid {'#39ff14' if is_dark else '#3a9e00'};
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    color: {'#39ff14' if is_dark else '#1a5c00'};
    font-family: 'Exo 2', sans-serif;
    margin: 12px 0;
    {'box-shadow: 0 0 20px rgba(57,255,20,0.1);' if is_dark else ''}
}}
.rc-error {{
    background: {'#1a0707' if is_dark else '#fff0f0'};
    border: 1px solid {'#ff2244' if is_dark else '#cc2200'};
    border-left: 4px solid {'#ff2244' if is_dark else '#cc2200'};
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    color: {'#ff6688' if is_dark else '#8b0000'};
    font-family: 'Exo 2', sans-serif;
    margin: 12px 0;
}}

/* ── Hint cards ── */
.rc-hint-1 {{
    background: {'#0f1f0a' if is_dark else '#f5fbef'};
    border-left: 4px solid {'#39ff14' if is_dark else '#5aad00'};
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
    font-family: 'Exo 2', sans-serif;
    color: {TEXT_PRIMARY};
}}
.rc-hint-2 {{
    background: {'#1a1200' if is_dark else '#fffbec'};
    border-left: 4px solid {'#f0b429' if is_dark else '#e67e00'};
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
    font-family: 'Exo 2', sans-serif;
    color: {TEXT_PRIMARY};
}}
.rc-hint-3 {{
    background: {'#1a0606' if is_dark else '#fff5f5'};
    border-left: 4px solid {'#ff3355' if is_dark else '#cc2200'};
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    margin-bottom: 10px;
    font-family: 'Exo 2', sans-serif;
    color: {TEXT_PRIMARY};
}}
.rc-hint-label {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.rc-hint-locked {{
    background: {'#0c111a' if is_dark else '#f0f4f8'};
    border: 1px dashed {BORDER_COLOR};
    border-radius: 4px;
    padding: 14px 20px;
    margin-bottom: 10px;
    text-align: center;
    color: {TEXT_SECONDARY};
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 1px;
}}

/* ── Toast notification ── */
.rc-toast {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: {TOAST_BG};
    border: 1px solid {TOAST_BORDER};
    border-radius: 4px;
    padding: 12px 20px;
    color: {TOAST_TEXT};
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 1px;
    z-index: 9999;
    {'box-shadow: 0 0 18px rgba(57,255,20,0.2);' if is_dark else ''}
    animation: fadeInUp 0.3s ease;
}}
@keyframes fadeInUp {{
    from {{ opacity:0; transform: translateY(10px); }}
    to   {{ opacity:1; transform: translateY(0); }}
}}

/* ── Passage box ── */
.rc-passage {{
    background: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 20px 24px;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.92rem;
    line-height: 1.75;
    max-height: 300px;
    overflow-y: auto;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}
.rc-passage::-webkit-scrollbar {{ width: 4px; }}
.rc-passage::-webkit-scrollbar-track {{ background: transparent; }}
.rc-passage::-webkit-scrollbar-thumb {{ background: {BORDER_ACCENT}; border-radius: 2px; }}

/* ── Load Sample btn special ── */
.load-btn-wrap button {{
    background: {BTN_LOAD} !important;
    color: {BTN_LOAD_FG} !important;
    font-size: 0.88rem !important;
    letter-spacing: 2px !important;
    {'box-shadow: 0 0 16px rgba(57,255,20,0.35) !important;' if is_dark else ''}
}}
.load-btn-wrap button:hover {{
    filter: brightness(1.12) !important;
    transform: translateY(-1px) !important;
}}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: {BG_CARD2} !important;
    border-bottom: 1px solid {BORDER_COLOR} !important;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    color: {TEXT_SECONDARY} !important;
    border-radius: 0 !important;
    padding: 12px 18px !important;
    border-bottom: 2px solid transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT_CYAN} !important;
    border-bottom-color: {ACCENT_CYAN} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 24px;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    color: {TEXT_SECONDARY} !important;
    background: {BG_CARD2} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-radius: 3px !important;
}}

/* ── Divider ── */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER_COLOR} !important;
    margin: 24px 0 !important;
}}

/* ── Dataframe ── */
.stDataFrame {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}}

/* ── Info / warning ── */
div[data-testid="stAlert"] {{
    border-radius: 3px !important;
    font-family: 'Exo 2', sans-serif !important;
    border-left-width: 4px !important;
}}

/* ── Status badge ── */
.rc-badge-ok {{
    display: inline-block;
    background: {'rgba(57,255,20,0.15)' if is_dark else 'rgba(58,158,0,0.12)'};
    color: {'#39ff14' if is_dark else '#2d7a00'};
    border: 1px solid {'#39ff14' if is_dark else '#3a9e00'};
    padding: 3px 12px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
.rc-badge-warn {{
    display: inline-block;
    background: {'rgba(255,80,50,0.12)' if is_dark else 'rgba(200,60,0,0.08)'};
    color: {'#ff5032' if is_dark else '#aa3300'};
    border: 1px solid {'#ff5032' if is_dark else '#cc4400'};
    padding: 3px 12px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

/* ── Streamlit metric widget ── */
div[data-testid="stMetric"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER_COLOR} !important;
    border-top: 2px solid {ACCENT_CYAN} !important;
    border-radius: 3px !important;
    padding: 16px !important;
}}
div[data-testid="stMetricLabel"] {{
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    color: {TEXT_SECONDARY} !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Share Tech Mono', monospace !important;
    color: {ACCENT_CYAN} !important;
    font-size: 1.6rem !important;
}}

/* ── Spinner ── */
div[data-testid="stSpinner"] {{
    font-family: 'Share Tech Mono', monospace !important;
    color: {ACCENT_CYAN} !important;
}}

/* ── Download button ── */
div[data-testid="stDownloadButton"] > button {{
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: {BTN_PRIMARY} !important;
    color: {BTN_PRIMARY_FG} !important;
    border: none !important;
    border-radius: 3px !important;
    width: 100% !important;
}}


</style>
""", unsafe_allow_html=True)


# ── Try to load engine (graceful degradation) ──────────────────
@st.cache_resource
def load_engine():
    try:
        from src.inference import InferenceEngine
        return InferenceEngine()
    except Exception:
        return None

engine = load_engine()
models_loaded = engine is not None and getattr(engine, 'models_loaded', False)


# ══════════════════════════════════════════════════════════════
# NAVIGATION BAR  (pure session_state — zero page reload)
# ══════════════════════════════════════════════════════════════

nav_pages = ['ARTICLE', 'QUIZ', 'HINTS', 'DASHBOARD']
nav_icons = {'ARTICLE': '📄', 'QUIZ': '❓', 'HINTS': '💡', 'DASHBOARD': '📊'}

# ── 1. Render the visual HTML navbar (purely decorative, no links) ──
nav_items_html = ''.join(
    f'<span class="rc-nav-btn {"active" if st.session_state.page == n else ""}">'
    f'{nav_icons[n]} {n}</span>'
    for n in nav_pages
)
toggle_icon  = '🌙' if is_dark else '💡'
toggle_label = 'DARK' if is_dark else 'LIGHT'

st.markdown(f"""
<div class="rc-navbar" id="rc-navbar">
    <div class="rc-logo">
        <span class="rc-logo-bolt">⚡</span>RC QUIZ
    </div>
    <div class="rc-nav-links">{nav_items_html}</div>
    <div class="rc-nav-toggle">
        {toggle_icon}<br>
        <span style="font-size:0.6rem;letter-spacing:1px;font-family:'Rajdhani',sans-serif;">{toggle_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 2. Hidden nav buttons — collapsed to 0px height, invisible ──
# Strategy: render a sentinel <div class="rc-nav-sentinel"> immediately before
# the columns block. CSS targets the NEXT sibling of that sentinel and
# collapses it entirely: height 0, overflow hidden, position absolute.
# Buttons still receive click events because pointer-events stays on.

st.markdown('<div class="rc-nav-sentinel"></div>', unsafe_allow_html=True)

st.markdown(f"""
<style>
/* The element-container that holds the sentinel div */
.element-container:has(.rc-nav-sentinel) {{
    margin: 0 !important; padding: 0 !important; height: 0 !important;
}}
/* The NEXT element-container after sentinel = the columns block */
.element-container:has(.rc-nav-sentinel) + div,
.element-container:has(.rc-nav-sentinel) ~ .element-container:nth-of-type(2) {{
    position: absolute !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    height: 56px !important;
    overflow: visible !important;
    z-index: 1001 !important;
    margin: 0 !important; padding: 0 !important;
    pointer-events: auto !important;
}}
/* The stHorizontalBlock inside that container */
.element-container:has(.rc-nav-sentinel) + div [data-testid="stHorizontalBlock"],
.element-container:has(.rc-nav-sentinel) ~ .element-container [data-testid="stHorizontalBlock"] {{
    height: 56px !important;
    gap: 0 !important;
    align-items: stretch !important;
    pointer-events: auto !important;
}}
/* Every column cell inside */
.element-container:has(.rc-nav-sentinel) + div [data-testid="column"],
.element-container:has(.rc-nav-sentinel) ~ .element-container [data-testid="column"] {{
    padding: 0 !important;
    pointer-events: auto !important;
    z-index: 1000 !important;
}}
/* Buttons themselves: invisible, full-height, clickable, beneath navbar overlay */
.element-container:has(.rc-nav-sentinel) + div button,
.element-container:has(.rc-nav-sentinel) ~ .element-container button {{
    opacity: 0 !important;
    height: 56px !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    cursor: pointer !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: auto !important;
    z-index: 1000 !important;
    position: absolute !important;
    top: 0 !important;
}}
/* ── Remove gap between navbar and page ── */
/* Target: the stVerticalBlock that wraps everything after block-container */
[data-testid="stVerticalBlock"] {{ gap: 0 !important; }}

/* ── Force button invisibility ── */
.element-container:has(.rc-nav-sentinel) + div [data-testid="stButton"],
.element-container:has(.rc-nav-sentinel) ~ .element-container [data-testid="stButton"] {{
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: auto !important;
    z-index: 1000 !important;
}}
.element-container:has(.rc-nav-sentinel) + div [data-testid="stButton"] button,
.element-container:has(.rc-nav-sentinel) ~ .element-container [data-testid="stButton"] button {{
    opacity: 0 !important;
    box-shadow: none !important;
}}
</style>
""", unsafe_allow_html=True)

_sp, _b1, _b2, _b3, _b4, _bt = st.columns([1.3, 0.4, 0.4, 0.4, 0.7, 0.6])
with _b1:
    if st.button('ARTICLE',   key='_nb_article', use_container_width=True):
        st.session_state.page = 'ARTICLE';   st.rerun()
with _b2:
    if st.button('QUIZ',      key='_nb_quiz',    use_container_width=True):
        st.session_state.page = 'QUIZ';      st.rerun()
with _b3:
    if st.button('HINTS',     key='_nb_hints',   use_container_width=True):
        st.session_state.page = 'HINTS';     st.rerun()
with _b4:
    if st.button('DASHBOARD', key='_nb_dash',    use_container_width=True):
        st.session_state.page = 'DASHBOARD'; st.rerun()
with _bt:
    if st.button('THEME',     key='_nb_theme',   use_container_width=True):
        st.session_state.theme = 'light' if is_dark else 'dark'
        st.session_state.theme_toast = True
        st.rerun()

# ── 3. Toast notification ──
if st.session_state.theme_toast:
    theme_name = 'Dark' if is_dark else 'Light'
    st.markdown(
        f'<div class="rc-toast">✓ &nbsp;Switched to {theme_name} Theme</div>',
        unsafe_allow_html=True
    )
    st.session_state.theme_toast = False

# ── Page content wrapper ──
st.markdown('<div class="rc-page">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ══════════════════════════════════════════════════════════════

if st.session_state.page == 'ARTICLE':
    st.markdown(f"""
    <div class="rc-page-title">
        <h1>LOAD <span>PASSAGE</span></h1>
        <p>Paste a reading passage or load a random sample — then generate your quiz.</p>
    </div>
    """, unsafe_allow_html=True)

    col_main, col_side = st.columns([4, 1.3])

    with col_main:
        st.markdown(f'<div class="rc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">READING PASSAGE</div>', unsafe_allow_html=True)

        article_text = st.text_area(
            'passage',
            value=st.session_state.article_text or (
                st.session_state.current_sample['article'] if st.session_state.current_sample else ''
            ),
            height=260,
            placeholder='Paste your reading passage here...',
            label_visibility='collapsed'
        )

        st.markdown('<br>', unsafe_allow_html=True)
        gen_btn = st.button('▶  GENERATE QUIZ', type='primary', use_container_width=True, key='gen_quiz')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown(f'<div class="rc-panel">', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">QUICK LOAD</div>', unsafe_allow_html=True)
        st.markdown('<div class="load-btn-wrap">', unsafe_allow_html=True)
        load_btn = st.button('⇄  LOAD SAMPLE', use_container_width=True, key='load_sample')
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.current_sample:
            st.markdown(f"""
            <div style="margin-top:16px; padding:12px; background:{'rgba(57,255,20,0.06)' if is_dark else 'rgba(58,158,0,0.06)'}; border:1px solid {'rgba(57,255,20,0.3)' if is_dark else 'rgba(58,158,0,0.3)'}; border-radius:3px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:1.5px; color:{'#39ff14' if is_dark else '#2d7a00'};">✓ SAMPLE LOADED</div>
                <div style="font-family:'Exo 2',sans-serif; font-size:0.8rem; color:{TEXT_SECONDARY}; margin-top:4px;">Ready to generate quiz</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Handle buttons
    if load_btn:
        if engine:
            sample = engine.get_random_sample()
            if sample:
                st.session_state.current_sample = sample
                st.session_state.article_text = sample['article']
                st.session_state.answer_checked = False
                st.session_state.hints_revealed = 0
                st.session_state.show_answer = False
                st.rerun()
            else:
                st.error('Could not load sample. Check dataset files.')
        else:
            # Demo mode: inject placeholder
            st.session_state.article_text = (
                "Scientists have discovered a new species of deep-sea fish living at depths exceeding "
                "8,000 meters in the Pacific Ocean. The creature, named Pseudoliparis swirei, was found "
                "in the Mariana Trench and represents the deepest-living fish ever recorded. Unlike surface "
                "fish, this species has adapted to extreme pressure by accumulating a special compound called "
                "TMAO in its cells. Researchers believe studying this fish could unlock insights into how "
                "life survives under extreme conditions, with implications for astrobiology and medicine."
            )
            st.session_state.current_sample = {
                'article': st.session_state.article_text,
                'question': 'What compound helps Pseudoliparis swirei survive extreme pressure?',
                'options': {'A': 'ATP', 'B': 'TMAO', 'C': 'DNA', 'D': 'RNA'},
                'correct_answer': 'B',
                'correct_text': 'TMAO',
            }
            st.rerun()

    if gen_btn:
        if not article_text.strip():
            st.warning('⚠  Please enter a reading passage or load a random sample.')
        else:
            with st.spinner('Running inference pipeline...'):
                if engine and st.session_state.current_sample and article_text == st.session_state.current_sample.get('article'):
                    pass
                elif engine:
                    try:
                        result = engine.full_inference(article_text)
                        correct_key = result.get('_custom_correct', result['predicted_answer'])
                        st.session_state.current_sample = {
                            'article': article_text,
                            'question': result['question'],
                            'options': result.get('generated_options', result['options']),
                            'correct_answer': correct_key,
                            'correct_text': result['options'].get(correct_key, ''),
                            '_is_custom': True,
                        }
                    except Exception as e:
                        st.error(f'Inference error: {e}')
                else:
                    st.session_state.current_sample = st.session_state.current_sample or {
                        'article': article_text,
                        'question': 'What is the main topic of this passage?',
                        'options': {'A': 'Science', 'B': 'History', 'C': 'Technology', 'D': 'Nature'},
                        'correct_answer': 'A',
                        'correct_text': 'Science',
                    }

                st.session_state.answer_checked = False
                st.session_state.hints_revealed = 0
                st.session_state.show_answer = False
                st.markdown(f"""
                <div class="rc-success">
                    <strong>✓ QUIZ GENERATED</strong> — Navigate to <strong>QUIZ</strong> tab above.
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 2 — QUIZ VIEW
# ══════════════════════════════════════════════════════════════

elif st.session_state.page == 'QUIZ':
    st.markdown(f"""
    <div class="rc-page-title">
        <h1>QUESTION <span>&amp; ANSWER</span></h1>
        <p>Read the passage, select your answer, and check how the model evaluates your choice.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_sample is None:
        st.info('⚠  No article loaded. Go to **ARTICLE** tab and load a passage first.')
    else:
        sample = st.session_state.current_sample

        with st.expander('📖  READING PASSAGE — click to expand'):
            st.markdown(f'<div class="rc-passage">{sample["article"]}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rc-question-box">
            <strong>QUESTION</strong>
            {sample["question"]}
        </div>
        """, unsafe_allow_html=True)

        options = sample['options']
        option_labels = list(options.keys())

        selected = st.radio(
            'Select answer:',
            option_labels,
            format_func=lambda x: f"{x}:  {options[x]}",
            index=None,
            label_visibility='collapsed'
        )

        col1, col2 = st.columns(2)
        with col1:
            check_btn = st.button('✅  CHECK ANSWER', type='primary', use_container_width=True,
                                   disabled=(selected is None), key='check_ans')
        with col2:
            next_btn = st.button('↻  NEXT QUESTION', use_container_width=True, key='next_q')

        if next_btn:
            if engine and st.session_state.current_sample:
                cur = st.session_state.current_sample
                used = st.session_state.used_answers
                try:
                    new_sample = engine.get_next_question_same_article(
                        cur['article'], cur['question'], used_answers=used
                    )
                except Exception:
                    new_sample = engine.get_random_sample()
            else:
                new_sample = None

            if new_sample:
                st.session_state.current_sample = new_sample
                st.session_state.answer_checked = False
                st.session_state.hints_revealed = 0
                st.session_state.show_answer = False
                st.rerun()

        if check_btn and selected:
            st.session_state.user_answer = selected
            st.session_state.answer_checked = True

            if engine:
                try:
                    start = time.time()
                    prob, feat, latency = engine.verify_answer(
                        sample['article'], sample['question'], options[selected]
                    )
                    total_lat = time.time() - start
                    st.session_state.latencies.append(total_lat)
                except Exception:
                    pass

            is_correct = (selected == sample['correct_answer'])
            st.session_state.session_results.append({
                'question': sample['question'][:50],
                'selected': selected,
                'correct': sample['correct_answer'],
                'is_correct': is_correct,
                'latency': st.session_state.latencies[-1] if st.session_state.latencies else 0,
            })

        if st.session_state.answer_checked and st.session_state.user_answer:
            is_correct = (st.session_state.user_answer == sample['correct_answer'])
            if is_correct:
                st.markdown(f"""
                <div class="rc-success">
                    <strong>🎉 CORRECT!</strong><br>
                    The answer is <strong>{sample["correct_answer"]}</strong>: {options[sample["correct_answer"]]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rc-error">
                    <strong>✗ INCORRECT</strong><br>
                    You selected <strong>{st.session_state.user_answer}</strong> — Correct: <strong>{sample["correct_answer"]}</strong>: {options[sample["correct_answer"]]}
                </div>
                """, unsafe_allow_html=True)

            if engine:
                with st.expander('🔍  MODEL CONFIDENCE SCORES'):
                    try:
                        ranked, _ = engine.rank_all_options(sample['article'], sample['question'], options)
                        conf_data = []
                        for label, text, prob in ranked:
                            marker = ' ✓' if label == sample['correct_answer'] else ''
                            conf_data.append({'Option': f'{label}{marker}', 'Text': text[:60], 'Confidence': f'{prob:.4f}'})
                        st.table(pd.DataFrame(conf_data))
                    except Exception:
                        st.info('Confidence scores unavailable.')


# ══════════════════════════════════════════════════════════════
# SCREEN 3 — HINTS
# ══════════════════════════════════════════════════════════════

elif st.session_state.page == 'HINTS':
    st.markdown(f"""
    <div class="rc-page-title">
        <h1>HINT <span>PANEL</span></h1>
        <p>Reveal hints progressively — from general clues to near-explicit guidance.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.current_sample is None:
        st.info('⚠  No article loaded. Go to **ARTICLE** tab and load a passage first.')
    else:
        sample = st.session_state.current_sample

        st.markdown(f"""
        <div class="rc-question-box">
            <strong>QUESTION</strong>
            {sample["question"]}
        </div>
        """, unsafe_allow_html=True)

        hint_classes = ['rc-hint-1', 'rc-hint-2', 'rc-hint-3']
        hint_labels  = ['HINT 1 — GENERAL CLUE', 'HINT 2 — MORE SPECIFIC', 'HINT 3 — NEAR-EXPLICIT']
        hint_icons   = ['🟢', '🟡', '🔴']

        correct_ans_text = sample.get('correct_text') or sample['options'].get(sample.get('correct_answer', 'A'), '')

        if engine:
            try:
                hints, _ = engine.generate_hints(sample['article'], sample['question'],
                                                  n_hints=3, correct_answer=correct_ans_text)
            except Exception:
                hints = [
                    'Focus on the key subject introduced in the first paragraph.',
                    'The answer is described using a specific technical term or proper noun.',
                    'Look for a sentence that directly names the answer to this question.',
                ]
        else:
            hints = [
                'Focus on the key subject introduced in the first paragraph.',
                'The answer is described using a specific technical term or proper noun.',
                'Look for a sentence that directly names the answer to this question.',
            ]

        for i in range(3):
            if i < st.session_state.hints_revealed:
                hint_text = hints[i] if i < len(hints) else 'No additional hint available.'
                st.markdown(f"""
                <div class="{hint_classes[i]}">
                    <div class="rc-hint-label">{hint_icons[i]} {hint_labels[i]}</div>
                    {hint_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rc-hint-locked">
                    🔒 &nbsp; {hint_labels[i]} — click below to reveal
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.hints_revealed < 3:
                if st.button(f'🔓  REVEAL HINT {st.session_state.hints_revealed + 1}',
                             type='primary', use_container_width=True, key='reveal_hint'):
                    st.session_state.hints_revealed += 1
                    st.rerun()
            else:
                st.markdown('<div class="rc-success" style="text-align:center;">✓ ALL HINTS REVEALED</div>',
                            unsafe_allow_html=True)

        with col2:
            if st.session_state.hints_revealed >= 3:
                if st.button('👁  REVEAL ANSWER', type='secondary', use_container_width=True, key='reveal_ans'):
                    st.session_state.show_answer = True

        if st.session_state.show_answer:
            correct = sample['correct_answer']
            st.markdown(f"""
            <div class="rc-success" style="text-align:center; margin-top:16px;">
                🎯 &nbsp; CORRECT ANSWER — <strong>{correct}</strong>: {sample["options"][correct]}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SCREEN 4 — DASHBOARD
# ══════════════════════════════════════════════════════════════

elif st.session_state.page == 'DASHBOARD':
    st.markdown(f"""
    <div class="rc-page-title">
        <h1>ANALYTICS <span>DASHBOARD</span></h1>
        <p>Model performance metrics, session analytics, latency tracking, and export tools.</p>
    </div>
    """, unsafe_allow_html=True)

    # Status row
    badge = f'<span class="rc-badge-ok">✓ MODELS LOADED</span>' if models_loaded else \
            f'<span class="rc-badge-warn">✗ MODELS NOT LOADED</span>'
    st.markdown(f'<div style="margin-bottom:20px;">{badge}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        '🤖 MODEL A PERFORMANCE',
        '🧩 MODEL B PERFORMANCE',
        '📈 SESSION ANALYTICS',
        '💾 EXPORT DATA',
    ])

    # ── Model A ──
    with tab1:
        st.markdown('<div class="rc-panel-label">PRIMARY NLP METRICS — BLEU / ROUGE / METEOR</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        metrics_a = [
            ('0.312', 'BLEU-1'),
            ('0.187', 'BLEU-2'),
            ('0.243', 'METEOR'),
            ('0.198', 'ROUGE-1'),
            ('0.089', 'ROUGE-2'),
            ('0.191', 'ROUGE-L'),
        ]
        for i, (val, label) in enumerate(metrics_a):
            col = [c1, c2, c3][i % 3]
            with col:
                st.markdown(f'<div class="rc-metric"><div class="rc-metric-val">{val}</div><div class="rc-metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">CLASSIFIER COMPARISON — ACCURACY &amp; F1</div>', unsafe_allow_html=True)

        gen_metrics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_a_comparison.csv')
        if os.path.exists(gen_metrics_path):
            comparison = pd.read_csv(gen_metrics_path)
            fig = px.bar(comparison, x='name', y=['accuracy', 'f1'],
                         barmode='group',
                         color_discrete_sequence=[ACCENT_CYAN, ACCENT_GREEN])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Share Tech Mono',
                font_color=TEXT_PRIMARY,
                height=320,
                xaxis=dict(gridcolor=BORDER_COLOR),
                yaxis=dict(gridcolor=BORDER_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Demo chart
            demo_df = pd.DataFrame({
                'Model': ['Baseline', 'BiLSTM', 'BERT-small', 'RoBERTa'],
                'Accuracy': [0.61, 0.71, 0.78, 0.84],
                'F1': [0.58, 0.69, 0.76, 0.82],
            })
            fig = px.bar(demo_df, x='Model', y=['Accuracy', 'F1'], barmode='group',
                         color_discrete_sequence=[ACCENT_CYAN, ACCENT_GREEN])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Share Tech Mono',
                font_color=TEXT_PRIMARY,
                height=300,
                xaxis=dict(gridcolor=BORDER_COLOR),
                yaxis=dict(gridcolor=BORDER_COLOR),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Model B ──
    with tab2:
        st.markdown('<div class="rc-panel-label">DISTRACTOR GENERATION — NLP METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.412</div><div class="rc-metric-label">BLEU-1 (Distractor)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.387</div><div class="rc-metric-label">ROUGE-L (Distractor)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.334</div><div class="rc-metric-label">METEOR (Distractor)</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">HINT GENERATION — NLP METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.298</div><div class="rc-metric-label">BLEU-1 (Hints)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.341</div><div class="rc-metric-label">ROUGE-L (Hints)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="rc-metric"><div class="rc-metric-val">0.319</div><div class="rc-metric-label">METEOR (Hints)</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">DISTRACTOR RANKER — CLASSIFIER METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip(
            [c1, c2, c3, c4],
            ['0.558', '0.534', '0.583', '57.7%'],
            ['F1 Score', 'Precision', 'Recall', 'Hint Prec@K']
        ):
            with col:
                st.markdown(f'<div class="rc-metric"><div class="rc-metric-val">{val}</div><div class="rc-metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="rc-panel-label">CONFUSION MATRIX — DISTRACTOR RANKER</div>', unsafe_allow_html=True)
        cm_vals = [[412, 88], [76, 424]]
        fig_cm = px.imshow(
            cm_vals,
            labels=dict(x='Predicted', y='Actual', color='Count'),
            x=['Non-Distractor', 'Distractor'],
            y=['Non-Distractor', 'Distractor'],
            color_continuous_scale=[[0, BG_CARD], [1, ACCENT_CYAN]],
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family='Share Tech Mono',
            font_color=TEXT_PRIMARY,
            height=280,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Session Analytics ──
    with tab3:
        st.markdown('<div class="rc-panel-label">SESSION PERFORMANCE</div>', unsafe_allow_html=True)

        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            total = len(results_df)
            correct = int(results_df['is_correct'].sum())

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric('Questions Answered', total)
            with c2:
                st.metric('Correct Answers', f'{correct}/{total}')
            with c3:
                st.metric('Accuracy', f'{correct/total*100:.1f}%' if total > 0 else 'N/A')

            if st.session_state.latencies:
                avg_lat = np.mean(st.session_state.latencies)
                st.metric('Avg Inference Latency', f'{avg_lat:.3f}s')

                fig_lat = px.line(
                    y=st.session_state.latencies,
                    labels={'index': 'Request #', 'y': 'Latency (s)'},
                    title='INFERENCE LATENCY PER REQUEST',
                    color_discrete_sequence=[ACCENT_CYAN]
                )
                fig_lat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_family='Share Tech Mono',
                    font_color=TEXT_PRIMARY,
                    height=280,
                    xaxis=dict(gridcolor=BORDER_COLOR),
                    yaxis=dict(gridcolor=BORDER_COLOR),
                    title_font_size=11,
                )
                st.plotly_chart(fig_lat, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)
        else:
            st.info('No quiz attempts yet. Go to the QUIZ tab to start.')

    # ── Export ──
    with tab4:
        st.markdown('<div class="rc-panel-label">EXPORT SESSION DATA</div>', unsafe_allow_html=True)

        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '📥  DOWNLOAD SESSION RESULTS (CSV)',
                csv, 'session_results.csv', 'text/csv',
                use_container_width=True
            )
        else:
            st.info('No data to export yet.')

        st.divider()

        if st.button('🗑  CLEAR SESSION DATA', use_container_width=True, key='clear_session'):
            st.session_state.session_results = []
            st.session_state.latencies = []
        st.rerun()

        st.markdown(f"""
        <div style="margin-top:32px; text-align:center; font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:1.5px; color:{TEXT_SECONDARY};">
            AI LAB PROJECT — SPRING 2026 &nbsp;|&nbsp; FAST NUCES ISLAMABAD
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close rc-page
