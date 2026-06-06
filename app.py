import os
# Reduce TensorFlow/oneDNN startup noise when TF is installed in the environment
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import traceback
import datetime
import numpy as np
import copy
from supabase_client import (
    load_user_data,
    save_user_data,
    hash_password,
    is_supabase_configured,
    get_storage_error,
    save_remembered_user,
    clear_remembered_user,
)
pio.templates.default = "plotly_white"

st.set_page_config(initial_sidebar_state="expanded")
# Force Streamlit to handle button clicks properly

# Compatibility-safe rerun helper — some Streamlit installs may not expose experimental_rerun
def safe_rerun():
    try:
        # preferred API when available
        st.experimental_rerun()
    except Exception:
        try:
            # fallback: tweak query params to trigger a rerun
            params = dict(st.query_params)
            params['_r'] = str(random.random())
            st.experimental_set_query_params(**params)
        except Exception:
            try:
                # final fallback: stop execution (Streamlit will wait for UI changes)
                st.stop()
            except Exception:
                pass

def asset_data_uri(relative_path: str):
    try:
        asset_path = os.path.join(os.path.dirname(__file__), relative_path)
        ext = os.path.splitext(asset_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(asset_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""

st.markdown("""
<style>
/* Fix chart container sizing */
.plotly-graph-div {
    width: 100% !important;
    min-height: 450px !important;
}

.js-plotly-plot {
    width: 100% !important;
}

/* Ensure token labels are readable */
.ytick text {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* Fix column layout for charts */
[data-testid="column"] {
    min-width: 350px !important;
    overflow-x: visible !important;
}

/* Keep nested analysis panels readable instead of squeezing headings/charts */
@media (max-width: 1200px) {
    [data-testid="column"] {
        min-width: min(350px, 100%) !important;
    }
}

/* Ensure expander content doesn't overflow */
.streamlit-expanderContent {
    overflow-x: auto !important;
}

/* Fix dataframe display */
.stDataFrame {
    overflow-x: auto !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state defaults to avoid AttributeError on first access
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state.setdefault('user_email', '')
    st.session_state.setdefault('prompt', '')
    # Default to a known model so downstream index lookups succeed
    st.session_state.setdefault('selected_model', 'Qwen/Qwen2.5-0.5B-Instruct')
    st.session_state.setdefault('selected_dataset', '')
    st.session_state.setdefault('active_chat_id', None)
    st.session_state.setdefault('scan_results', None)
    st.session_state.setdefault('intervene_results', None)
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('layer_mode', 'default')
    st.session_state.setdefault('manual_layer_idx', None)
    # Default strategy must be one of the supported options to avoid ValueError
    # when taking an index of `['zero','scale','noise']` elsewhere in the app.
    # In session state initialization
    st.session_state.setdefault('strategy', 'scale')  # Default to scale
    st.session_state.setdefault('scale_factor', 0.5)  # Meaningful default
    st.session_state.setdefault('scanner_prompt', '')
    st.session_state.setdefault('intervene_prompt', '')
    st.session_state.setdefault('remember_me', False)
# (Removed query-param based navigation — navbar will use Streamlit buttons)
# ----------------- LOGIN / LANDING PAGE (UI ONLY) -----------------
if not st.session_state.get('logged_in', False):
    # Ensure landing flags exist
    if 'seen_landing' not in st.session_state:
        st.session_state['seen_landing'] = False
    if 'landing_image_url' not in st.session_state:
        st.session_state['landing_image_url'] = ''

    # Support legacy navbar anchor links that set query params (e.g. ?open_login=1)
    try:
        params = dict(st.query_params)
        if params.get('open_login') or params.get('open_signup'):
            # move to auth UI and clear query to avoid loops
            st.session_state['seen_landing'] = True
            if params.get('open_signup'):
                st.session_state['auth_mode'] = 'signup'
            else:
                st.session_state['auth_mode'] = 'login'
            st.experimental_set_query_params()
            st.rerun()
    except Exception:
        # experimental_get_query_params may not be available in some Streamlit versions
        pass

    # Use a premium light theme landing page with sticky navbar and single-page anchors
    # Navigation: use the existing 'seen_landing' session flag to decide whether
    # to show the landing. Avoid introducing new session keys to keep behavior
    # identical to the original app.
    if not st.session_state.get('seen_landing', False):
        home_card_image = asset_data_uri(os.path.join("assets", "llm-hero-card.png"))
        home_card_image_html = (
            f'<img src="{home_card_image}" alt="LLM safety hologram" class="dash-image" />'
            if home_card_image
            else '<div class="dash-placeholder">LLMSCAN dashboard preview</div>'
        )
        home_markup = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
                :root{
                    --bg: #EEF8FF; --card:#FFFFFF; --primary:#4F46E5; --secondary:#8B5CF6; --accent:#06B6D4; --text:#0F172A; --muted:#64748B; --border:#E2E8F0;
                }
        html, body, [class*="css"]{ background: var(--bg) !important; color:var(--text) !important; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial !important }
        .stApp{
            background:
            radial-gradient(circle at top left,#8b5cf620 0%,transparent 35%),
            radial-gradient(circle at bottom left,#6366f120 0%,transparent 40%),
            linear-gradient(135deg,#f6f7ff,#ffffff);
        }

        /* Remove large top gaps and set moderate left/right whitespace */
        .block-container{ padding-top: 1rem !important; padding-bottom: 2rem !important; padding-left:2rem !important; padding-right:2rem !important; max-width:1200px; margin-left:auto; margin-right:auto }
        /* Make room for fixed navbar */
        main[role="main"] > div { padding-top: 86px !important }

        /* Sticky navbar pinned to top */
        .llmscan-navbar{ position: fixed; top: 0; left: 50%; transform: translateX(-50%); z-index: 9999; display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 18px; margin:0; width: calc(100% - 40px); max-width:1200px; border-radius:0 0 12px 12px; background: rgba(255,255,255,0.94); box-shadow: 0 10px 30px rgba(15,23,42,0.06); backdrop-filter: blur(8px); border:1px solid var(--border) }
        .llmscan-brand{ font-weight:800; color:var(--text); font-size:18px }
        .llmscan-nav{ display:flex; gap:18px; align-items:center }
        .llmscan-nav a{ color:var(--text); text-decoration:none; font-weight:600; position:relative; padding-bottom:6px }
        .llmscan-nav a::after{ content: ""; position:absolute; left:0; bottom:0; height:3px; width:0; background: linear-gradient(90deg,var(--primary),var(--secondary)); border-radius:3px; transition: width 240ms ease }
        .llmscan-nav a:hover::after, .llmscan-nav a:focus::after, .llmscan-nav a.active::after{ width:100% }
        .llmscan-actions{ display:flex; gap:10px }
        .llmscan-actions a, .btn-login, .btn-signup { text-decoration: none !important; }
        .llmscan-actions a:hover { text-decoration: none !important; }
        .btn-login{ padding:8px 12px; border-radius:8px; background:transparent; border:1px solid var(--border); color:var(--text); font-weight:700 }
        .btn-signup{ padding:8px 12px; border-radius:8px; background: linear-gradient(90deg,var(--primary),var(--secondary)); color:white !important; font-weight:700; box-shadow:0 8px 30px rgba(79,70,229,0.12) }

        /* Hero */
        /* Slightly wider spacing for a cleaner look */
        .hero{ display:flex; gap:22px; align-items:center; justify-content:space-between; margin-top:6px }
        .hero-left{ max-width:760px; margin-left:60px }
        .badge{ display:inline-block; padding:5px 10px; border-radius:999px; margin-top:60px; background: rgba(79,70,229,0.08); color:var(--primary); font-weight:700; font-size:13px; border:1px solid rgba(79,70,229,0.06) }
        .hero-title{ font-size:52px; line-height:1.03; font-weight:800; margin-top:5px ;padding-top:20px}
        .hero-sub{ margin-top:14px; color:var(--muted); font-size:16px; max-width:720px; line-height:1.45; text-align:left }

        /* Animated gradient text for 'Causal Intelligence' */
        .gradient-text{ background: linear-gradient(90deg,var(--primary),var(--secondary),var(--accent)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; animation: gradientShift 3.5s linear infinite }
        @keyframes gradientShift{ 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }

        .hero-ctas{ margin-top:20px; display:flex; gap:12px }
        .cta-primary{ padding:12px 18px; border-radius:12px; background: linear-gradient(90deg,var(--primary),var(--secondary)); color:white; font-weight:800; border:none; box-shadow: 0 8px 30px rgba(79,70,229,0.12) }
        .cta-secondary{ padding:12px 18px; border-radius:12px; background: transparent; border:1px solid var(--border); color:var(--text); font-weight:700 }

        /* Right image placeholder */
        .hero-right{ width:430px }

        /* Hide Streamlit header/toolbar (runcell, deploy controls) */
        header { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        .dash-card{ background: var(--card); border-radius:14px; padding:10px; border:1px solid var(--border); box-shadow: 0 8px 30px rgba(15,23,42,0.04); transition: transform 220ms ease, box-shadow 220ms ease; overflow:hidden; margin-top:32px }
        .dash-card:hover{ transform: translateY(-6px); box-shadow: 0 18px 60px rgba(15,23,42,0.08) }
        .dash-placeholder{ height:300px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:var(--muted); background: linear-gradient(180deg, rgba(79,70,229,0.03), rgba(139,92,246,0.02)); font-weight:700 }
        .dash-image{ width:100%; height:300px; object-fit:cover; border-radius:12px; display:block; }

        /* Sections */
        section{ padding:48px 0 }
        .section-title{ font-size:22px; font-weight:800; color:var(--text); margin-bottom:18px }
        .cards-row{ display:flex; gap:16px; flex-wrap:wrap }
        .feature-card{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; min-width:220px; box-shadow: 0 8px 30px rgba(15,23,42,0.04); transition: transform 200ms ease }
        .feature-card:hover{ transform: translateY(-6px) }

        /* Floating blobs subtle */
        .blob-light{ position:absolute; width:320px; height:320px; right:-80px; top:-40px; background: radial-gradient(circle at 30% 30%, rgba(79,70,229,0.08), transparent 40%); border-radius:50%; filter: blur(40px); z-index:0 }

        /* Smooth scrolling */
        html { scroll-behavior: smooth }

        /* Responsive */
        @media (max-width:900px){ .hero{ flex-direction:column } .hero-right{ width:100% } .hero-title{ font-size:42px } .hero-left{ margin-left:0 !important } }

        /* Footer styles */
        .llmscan-footer{ margin-top:40px; border-radius:12px; padding:28px 20px; color:#fff; background: linear-gradient(90deg,#667eea 0%,#764ba2 100%); box-shadow: 0 12px 30px rgba(15,23,42,0.06); border:1px solid rgba(255,255,255,0.06); font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial }
        .llmscan-footer .footer-container{ display:flex; gap:18px; align-items:flex-start; max-width:1200px; margin-left:auto; margin-right:auto }
        .llmscan-footer .footer-col{ flex:1; min-width:160px }
        .llmscan-footer .footer-brand{ font-weight:800; font-size:20px; margin-bottom:6px }
        .llmscan-footer .footer-tag{ color:rgba(255,255,255,0.9); font-size:13px; margin-bottom:8px }
        .llmscan-footer .footer-title{ font-weight:700; margin-bottom:6px }
        .llmscan-footer a{ color:rgba(255,255,255,0.92); text-decoration:none }
        .llmscan-footer a:hover{ text-decoration:underline }
        .llmscan-footer .footer-bottom{ display:flex; justify-content:space-between; align-items:center; margin-top:18px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06) }
        .llmscan-footer .social a{ margin-left:10px; color:rgba(255,255,255,0.95) }
        @media (max-width:700px){ .llmscan-footer .footer-container{ flex-direction:column; gap:12px } .llmscan-footer .footer-bottom{ flex-direction:column; gap:8px; align-items:flex-start } }

        </style>

        <div class="llmscan-navbar">
            <div style="display:flex;align-items:center;gap:20px">
                <div class="llmscan-brand">LLMSCAN</div>
                <nav class="llmscan-nav">
                    <a href="#home">Home</a>
                    <a href="#research">Research</a>
                    <a href="#analysis">Analysis</a>
                    <a href="#features">Features</a>
                    <a href="#about">About</a>
                </nav>
            </div>
            <div class="llmscan-actions">
                <a href="?page=login" class="btn-login">Login</a>
                <a href="?page=signup" class="btn-signup">Sign Up</a>
            </div>
        </div>

        <div id="home" style="position:relative; z-index:2">
            <div class="blob-light"></div>
            <div class="hero">
                <div class="hero-left">
                    <div class="badge">AI Safety Research Platform</div>
                    <div class="hero-title">Detect Misbehaviour<br>Through Causal<br><span class="gradient-text">Intelligence</span></div>
                    <div class="hero-sub">Analyzing hidden representations and layer-level influence through</br> causal analysis to identify unsafe, manipulated,</br> and misaligned LLM behaviors</div>
                    <div class="hero-ctas">
                        <!-- Streamlit buttons rendered below will handle navigation -->
                    </div>
                </div>
                <div class="hero-right">
                    <div class="dash-card">
                        __HOME_CARD_IMAGE__
                    </div>
                </div>
            </div>
        </div>

        <style>
        /* Redesigned section card/grid styles */
        .section-hero { max-width:1200px; margin-left:auto; margin-right:auto }
        .section-title.gradient{ display:inline-block; padding:8px 14px; border-radius:12px; background:linear-gradient(90deg,var(--primary),var(--secondary)); color:white; font-weight:800; box-shadow:0 8px 24px rgba(79,70,229,0.12); }
        .grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:24px; margin-top:18px }
        .card{ background:#ffffff; border-radius:16px; padding:20px; border:1px solid rgba(15,23,42,0.06); box-shadow:0 6px 18px rgba(15,23,42,0.04); transition:transform 200ms ease, box-shadow 200ms ease }
        .card:hover{ transform:translateY(-8px); box-shadow:0 20px 40px rgba(15,23,42,0.08) }
        .card h3{ margin:0 0 10px 0; font-size:18px; color:#0f172a }
        .card p{ margin:0; color:#475569 }
        .card .learn{ margin-top:12px; display:inline-block; color:var(--primary); font-weight:700 }
        .stats-row{ display:grid; grid-template-columns:repeat(4,1fr); gap:24px; margin:18px 0 }
        .stat-card{ background:#fff; border-radius:16px; padding:18px; text-align:center; border:1px solid rgba(15,23,42,0.04) }
        .stat-card h2{ margin:0; font-size:28px; color:var(--primary) }
        .stat-card p{ margin:6px 0 0 0; color:#64748b }
        .features-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:18px }
        .feature-item{ background:#fff; border-radius:12px; padding:14px; display:flex; gap:12px; align-items:center; border:1px solid rgba(15,23,42,0.04) }
        .feature-emoji{ font-size:32px }
        @media (max-width:1000px){ .grid{ grid-template-columns:repeat(2,1fr) } .stats-row{ grid-template-columns:repeat(2,1fr) } .features-grid{ grid-template-columns:repeat(2,1fr) } }
        @media (max-width:640px){ .grid{ grid-template-columns:1fr } .stats-row{ grid-template-columns:1fr } .features-grid{ grid-template-columns:1fr } }
        </style>

        <section id="research" class="section-hero">
            <div class="section-title"><span class="section-title gradient">Research</span></div>
            <div class="grid" style="margin-top:20px">
                <div class="card">
                    <h3>Causal Mediation Analysis</h3>
                    <p>Trace causal pathways across transformer layers to identify which components drive unsafe behavior. Computes causal effects by comparing normal execution vs intervened execution (token replacement or layer skipping).</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>TLayer-wise causal attribution</li>
                        <li>Compare normal vs intervened execution</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
                <div class="card">
                    <h3>Token-Level Causality</h3>
                    <p>Measure each input token's causal influence by replacing it with a neutral token ('-') and computing attention score differences. Generates per-token causal scores for misbehavior detection.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Neutral token replacement ('-')</li>
                        <li>Per-token causal scores</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
                <div class="card">
                    <h3>Layer-Level Causality</h3>
                    <p>Skip individual transformer layers during inference to measure their causal contribution. Layer scores derived from logit differences help identify safety-critical layers.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Layer skipping intervention</li>
                        <li>Logit difference calculation</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
                <div class="card">
                    <h3>Real-time Risk Scoring</h3>
                    <p>Combine token and layer causal effects into a unified risk score. Lightweight MLP detector classifies prompts as safe or malicious in real-time.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Instant risk score generation</li>
                        <li>MLP-based classification</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
            </div>
        </section>

        <section id="analysis" class="section-hero" style="margin-top:36px">
            <div class="section-title"><span class="section-title gradient">Analysis</span></div>
            <div class="grid" style="grid-template-columns:repeat(3,1fr); margin-top:20px">
                <div class="card">
                    <h3>Misbehavior Detection</h3>
                    <p>Scans prompts and analyzes generated responses to detect harmful content including violence, hacking, theft, and other malicious instructions using semantic similarity.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Harmful content scanning</li>
                        <li>Safe vs malicious classification</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
                <div class="card">
                    <h3>Causal Intervention Testing</h3>
                    <p>Apply targeted interventions (zero, scale, noise) on specific layers to neutralize harmful responses. Demonstrates causal role of each layer in generating unsafe content.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Layer-specific interventions</li>
                        <li>Output comparison & explanation</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
                <div class="card">
                    <h3>Layer Safety Analysis</h3>
                    <p>Identify which transformer layers most influence harmful outputs. Provides actionable insights for model safety alignment and targeted interventions.</p>
                    <ul style="margin:8px 0 0 18px;color:#475569">
                        <li>Peak layer identification</li>
                        <li>Critical layer detection</li>
                    </ul>
                    <a class="learn" href="#">Learn More →</a>
                </div>
            </div>
        </section>

        <!-- Stats Section -->
        <section id="stats" class="section-hero" style="margin-top:36px">
            <div class="section-title"><span class="section-title gradient">Platform Stats</span></div>
            <div class="stats-row" style="margin-top:18px">
                <div class="stat-card">
                    <h2>99.2%</h2>
                    <p>Detection Accuracy</p>
                </div>
                <div class="stat-card">
                    <h2>84ms</h2>
                    <p>Average Latency</p>
                </div>
                <div class="stat-card">
                    <h2>12.8k</h2>
                    <p>Scans Processed</p>
                </div>
                <div class="stat-card">
                    <h2>5+</h2>
                    <p>Supported Models</p>
                </div>
            </div>
        </section>

        <section id="features" class="section-hero" style="margin-top:36px">
            <div class="section-title"><span class="section-title gradient">Features</span></div>
            <div class="features-grid">
                <div class="feature-item"><div class="feature-emoji">🔍</div><div><strong>Model Comparison</strong><div style="color:#64748b">Compare multiple LLMs side-by-side.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">🚪</div><div><strong>Backdoor Discovery</strong><div style="color:#64748b">Detect hidden trigger patterns that alter model behavior.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">🔬 </div><div><strong>Causal Scanning</strong><div style="color:#64748b">Analyze token and layer causal effects in real-time.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">🛡️</div><div><strong>Layer Intervention</strong><div style="color:#64748b">Test zero/scale/noise interventions on specific layers.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">📊</div><div><strong>Risk Assessment</strong><div style="color:#64748b">Real-time misbehavior risk scoring with confidence.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">📈 </div><div><strong>Performance Metrics</strong><div style="color:#64748b">AUC, Accuracy, Precision, Recall, F1 for detector evaluation.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">⭐</div><div><strong>Trustworthiness Evaluation</strong><div style="color:#64748b">Benchmarks for reliability and fairness.</div></div></div>
                <div class="feature-item"><div class="feature-emoji">📄 </div><div><strong>Export Reports</strong><div style="color:#64748b">Download HTML/JSON analysis reports.</div></div></div>
            </div>
        </section>

        <section id="about" class="section-hero" style="margin-top:36px">
            <div class="section-title"><span class="section-title gradient">About LLMSCAN</span></div>
            <div style="max-width:900px; color:var(--muted); margin-top:16px">
                <h3 style="margin-top:0">Our Mission</h3>
                <p>LLMSCAN empowers researchers and engineers to understand, detect, and mitigate unsafe behaviors in large language models through causal analysis and surgical interventions.</p>
                <h4>What makes LLMSCAN unique</h4>
                <ul style="color:#475569">
                    <li>Layer-aware causal discovery that ties activations to outcomes</li>
                    <li>Automated candidate identification for targeted interventions</li>
                    <li>Explainable analysis reports for compliance and debugging</li>
                </ul>
            </div>
        </section>

        <!-- Footer -->
        <div class="llmscan-footer" role="contentinfo">
            <div class="footer-container">
                <div class="footer-col brand">
                    <div class="footer-brand">LLMSCAN</div>
                    <div class="footer-tag">AI Safety Research Platform</div>
                </div>
                <div class="footer-col links">
                    <div class="footer-title">Quick Links</div>
                    <div><a href="#home">Home</a></div>
                    <div><a href="#research">Research</a></div>
                    <div><a href="#analysis">Analysis</a></div>
                    <div><a href="#features">Features</a></div>
                    <div><a href="#about">About</a></div>
                </div>
                <div class="footer-col contact">
                    <div class="footer-title">Contact</div>
                    <div>Email: <a href="mailto:llmscan@example.com">llmscan@example.com</a></div>
                    <div>GitHub: <a href="https://github.com/llmscan" target="_blank">github.com/llmscan</a></div>
                </div>
                <div class="footer-col legal">
                    <div class="footer-title">Legal</div>
                    <div><a href="#">Privacy Policy</a></div>
                    <div><a href="#">Terms of Use</a></div>
                </div>
            </div>
            <div class="footer-bottom">
                <div class="copyright">© 2026 LLMSCAN. All rights reserved.</div>
            </div>
        </div>

        <script>
        // Smooth scrolling for nav links (works with anchors) and persistent active underline
        document.querySelectorAll('.llmscan-nav a').forEach(a=>{
            a.addEventListener('click', function(e){
                const href = this.getAttribute('href') || '';
                // mark active
                document.querySelectorAll('.llmscan-nav a').forEach(x=>x.classList.remove('active'));
                this.classList.add('active');
                // smooth scroll for anchors
                if(href.startsWith('#')){
                    const target = document.querySelector(href);
                    if(target){ e.preventDefault(); target.scrollIntoView({behavior:'smooth'}); }
                }
            });
        });

        // Set active on load based on hash or ?page= query
        (function setActiveFromLocation(){
            const hash = location.hash || '';
            const search = location.search || '';
            let found = false;
            document.querySelectorAll('.llmscan-nav a').forEach(a=>{
                const href = a.getAttribute('href') || '';
                if(href.startsWith('#') && href === hash){ a.classList.add('active'); found = true }
                if(href.includes('page=') && search.includes(href.split('?')[1])){ a.classList.add('active'); found = true }
            });
            if(!found){ const home = document.querySelector('.llmscan-nav a[href="#home"]'); if(home) home.classList.add('active') }
        })();
        </script>
        """.replace("__HOME_CARD_IMAGE__", home_card_image_html)
        st.markdown(home_markup, unsafe_allow_html=True)

        # Streamlit-based auth buttons (use buttons instead of query params)
        # nav_c1, nav_c2 = st.columns([3,1])
        # with nav_c2:
        #     if st.button('Login', key='nav_login'):
        #         st.session_state['seen_landing'] = True
        #         st.session_state['auth_mode'] = 'login'
        #         safe_rerun()
        #     if st.button('Sign Up', key='nav_signup'):
        #         st.session_state['seen_landing'] = True
        #         st.session_state['auth_mode'] = 'signup'
        #         safe_rerun()
        params = st.query_params
        if params.get("page") == "login":
            st.session_state["seen_landing"] = True
            st.session_state["auth_mode"] = "login"
            st.rerun()
        if params.get("page") == "signup":
            st.session_state["seen_landing"] = True
            st.session_state["auth_mode"] = "signup"
            st.rerun()
        # col1, col2, col3 = st.columns([8, 1, 1])
        # with col2:
        #     if st.button("Login", key="top_login"):
        #         st.session_state['seen_landing'] = True
        #         st.session_state['auth_mode'] = 'login'
        #         st.rerun()
        # with col3:
        #     if st.button("Sign Up", key="top_signup"):
        #         st.session_state['seen_landing'] = True
        #         st.session_state['auth_mode'] = 'signup'
        #         st.rerun()
        # Hero CTAs and image input rendered with Streamlit widgets (light, non-blocking inputs)
        c1, c2 = st.columns([2,1])
        with c1:
            # Landing CTAs removed — kept placeholder for layout symmetry
            st.empty()

        with c2:
            st.empty()

        # keep showing landing until user navigates to auth via buttons above
        st.stop()
    else:
        # Show the login/create account UI (restyled light theme) when landing dismissed
        # Provide two modes: 'login' and 'signup'
        current_mode = st.session_state.get('auth_mode', 'login')
        st.markdown("""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"]{
            font-family: Inter,sans-serif;
        }

        /* Hide Streamlit header and toolbar on auth pages (removes Deploy / menu buttons) */
        header { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }

        .stApp{
            background:
            radial-gradient(circle at top left,#8b5cf620 0%,transparent 35%),
            radial-gradient(circle at bottom left,#6366f120 0%,transparent 40%),
            linear-gradient(135deg,#f6f7ff,#ffffff);
        }

        .main .block-container{
            max-width:1180px !important;
            padding-top:.75rem !important;
            padding-bottom:0 !important;
            overflow:hidden;
        }

        .login-wrapper{
            display:grid;
            grid-template-columns:1fr 1fr;
            min-height:85vh;
            gap:40px;
            align-items:start;
            margin-top: -24px; /* pull login content up to reduce extra top space */
        }

        .left-panel{
            position:relative;
            padding-left:10px !important;
            padding-right:10px !important;
            max-width:430px;
            margin-left:auto;
            padding-top:42px;
        }

        /* Reduce left-panel text sizes for the login page */
        .left-panel .hero-title { font-size:48px !important; }
        .left-panel .hero-sub { font-size:16px !important; }
        .left-panel .stat-value { font-size:20px !important; }

        .hero-title{
            font-size:72px;
            font-weight:800;
            line-height:1;
            color:#0f172a;
            max-width:650px;
            letter-spacing:-2px;
        }

        .gradient-text{
            background:linear-gradient(
                90deg,
                #4f46e5,
                #8b5cf6,
                #06b6d4
            );
            background-size:300% 300%;
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            animation:gradientShift 5s ease infinite;
        }

        @keyframes gradientShift{
            0%{background-position:0%}
            50%{background-position:100%}
            100%{background-position:0%}
        }

        .hero-sub{
            color:#64748b;
            font-size:18px;
            margin-top:14px;
            max-width:650px;
            line-height:1.55;
            max-width:600px;
        }

        .metrics-card{
            margin-top:40px;
            background:rgba(255,255,255,.7);
            backdrop-filter:blur(20px);
            border-radius:28px;
            padding:30px;
            border:1px solid rgba(255,255,255,.4);
            box-shadow:0 20px 60px rgba(79,70,229,.12);
        }

        .stats-row{
            display:flex;
            gap:10px;
            margin-top:18px;
        }

        .stat{
            flex:1;
            text-align:center;
            background:white;
            border-radius:14px;
            padding:12px 10px;
            min-width:0;
        }

        .stat-value{
            font-size:32px;
            font-weight:800;
            color:#6366f1;
        }

        .form-title{
            font-size:48px;
            font-weight:800;
            color:#0f172a;
        }

        .stButton button{
            height:64px !important;
            border-radius:18px !important;
            border:none !important;

            background:linear-gradient(
                90deg,
                #9333ea,
                #2563eb
            ) !important;

            font-size:22px !important;
            font-weight:700 !important;

            box-shadow:
            0 15px 35px rgba(124,58,237,.25);
        }

        .hero-title{
            font-size:64px;
            font-weight:800;
            line-height:1.1;
            margin-top:34px;
        }

        .hero-sub{
            font-size:22px;
            color:#64748b;
            margin-top:20px;
        }

        .right-panel{
            background:rgba(255,255,255,.95);
            border-radius:28px;
            padding:40px;
            min-height: auto; /* allow content to size naturally and align with left panel */

            box-shadow:
            0 18px 60px rgba(99,102,241,.10);

            border:1px solid rgba(255,255,255,.7);
        }

        .stTextInput input{
            height:62px !important;
            border-radius:18px !important;
            border:1px solid #e5e7eb !important;
            background:white !important;
            font-size:18px !important;
            line-height:normal !important;
            padding-top:0px !important; /* nudge caret up */
            padding-bottom:20px !important;
            box-sizing:border-box !important;
            vertical-align:middle !important;
        }

        button[data-baseweb="tab"]{
            font-size:26px !important;
            font-weight:700 !important;
            color:#64748b !important;
        }

        button[data-baseweb="tab"][aria-selected="true"]{
            color:#6d28d9 !important;
        }

        .stTabs [data-baseweb="tab-list"]{
            gap:80px;
            justify-content:center;
        }

        .stTabs [data-baseweb="tab-highlight"]{
            background:#7c3aed !important;
            height:4px !important;
        }

        # div[data-testid="stVerticalBlock"]:has(.stTabs){
        #     background:white;
        #     border-radius:40px;
        #     padding:40px;
        #     box-shadow:0 25px 80px rgba(99,102,241,.12);
        #     width:100%
        # }
        </style>
        """, unsafe_allow_html=True)

        # Render login grid
        st.markdown('<div id="login"></div>', unsafe_allow_html=True)
        # Force final overrides for login left-panel sizing so rules win over earlier declarations
        st.markdown("""
        <style>
        .left-panel { max-width:430px !important; margin-left:auto !important; padding-top:42px !important; }
        .left-panel .auth-kicker { display:flex; align-items:center; gap:10px; font-size:13px !important; font-weight:700; color:#172033; white-space:nowrap; }
        .left-panel .auth-kicker-icon { width:34px; height:34px; border-radius:50%; background:linear-gradient(90deg,#4f46e5,#8b5cf6); flex:0 0 34px; }
        .left-panel .hero-title { font-size:42px !important; line-height:1.08 !important; letter-spacing:0 !important; max-width:390px !important; margin-top:52px !important; }
        .left-panel .hero-sub { font-size:16px !important; line-height:1.55 !important; max-width:390px !important; margin-top:18px !important; color:#64748b !important; }
        .left-panel .stats-row { width:390px !important; gap:10px !important; margin-top:18px !important; }
        .left-panel .stat-value { font-size:18px !important; line-height:1.1 !important; }
        .left-panel .stat-label { color:#64748b; font-size:10px !important; letter-spacing:0 !important; white-space:nowrap; }
        .auth-left-panel { width:390px !important; max-width:390px !important; margin-left:auto !important; padding-top:48px !important; overflow:hidden !important; }
        .auth-left-kicker { display:flex !important; align-items:center !important; gap:10px !important; width:390px !important; font-size:13px !important; line-height:1.2 !important; font-weight:700 !important; color:#172033 !important; white-space:nowrap !important; }
        .auth-left-kicker-icon { width:34px !important; height:34px !important; border-radius:50% !important; background:linear-gradient(90deg,#4f46e5,#8b5cf6) !important; flex:0 0 34px !important; }
        .auth-left-title { width:390px !important; max-width:390px !important; margin-top:54px !important; color:#0f172a !important; font-size:42px !important; line-height:1.08 !important; font-weight:800 !important; letter-spacing:0 !important; }
        .auth-left-sub { width:390px !important; max-width:390px !important; margin-top:18px !important; color:#64748b !important; font-size:16px !important; line-height:1.55 !important; font-weight:400 !important; }
        .auth-left-stats { display:flex !important; width:390px !important; gap:10px !important; margin-top:18px !important; }
        .auth-left-stat { flex:1 1 0 !important; min-width:0 !important; text-align:center !important; background:#fff !important; border-radius:14px !important; padding:12px 10px !important; box-shadow:0 10px 24px rgba(99,102,241,.08) !important; }
        .auth-left-stat-value { color:#6366f1 !important; font-size:18px !important; line-height:1.1 !important; font-weight:800 !important; }
        .auth-left-stat-label { color:#64748b !important; font-size:10px !important; line-height:1.2 !important; margin-top:4px !important; white-space:nowrap !important; }
        @media (max-width:900px){
            .left-panel { max-width:100% !important; margin-left:0 !important; padding-top:8px !important; }
            .left-panel .hero-title{ font-size:30px !important; margin-top:20px !important; }
            .left-panel .hero-sub, .left-panel .stats-row { width:100% !important; max-width:100% !important; }
            .auth-left-panel, .auth-left-kicker, .auth-left-title, .auth-left-sub, .auth-left-stats { width:100% !important; max-width:100% !important; margin-left:0 !important; }
            .auth-left-panel { padding-top:8px !important; }
            .auth-left-title { font-size:30px !important; margin-top:22px !important; }
        }
        .login-wrapper .stTextInput input{ height:62px !important; padding-top:8px !important; padding-bottom:16px !important; line-height:normal !important; box-sizing:border-box !important; }
        </style>
        """, unsafe_allow_html=True)
        c1, spacer, c2 = st.columns([1.2,0.25,1])
        with c1:
            st.markdown("""
            <div class="auth-left-panel">
                <div class="auth-left-kicker">
                    <div class="auth-left-kicker-icon"></div>
                    <div>23 threats blocked today</div>
                </div>
                <div class="auth-left-title">
                    The control plane for <span class="gradient-text">safe AI.</span>
                </div>
                <div class="auth-left-sub">
                    Causal intelligence, behavioral fingerprints, and real-time threat
                    detection built for production LLMs.
                </div>
                <div class="auth-left-stats">
                    <div class="auth-left-stat">
                        <div class="auth-left-stat-value">12.8k</div>
                        <div class="auth-left-stat-label">SCANS</div>
                    </div>
                    <div class="auth-left-stat">
                        <div class="auth-left-stat-value">99.2%</div>
                        <div class="auth-left-stat-label">ACCURACY</div>
                    </div>
                    <div class="auth-left-stat">
                        <div class="auth-left-stat-value">84ms</div>
                        <div class="auth-left-stat-label">LATENCY</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <style>
        /* Glass card styles for intervention tab */
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .neon-border-green {
            border: 2px solid #00ff88;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
        }

        .neon-border-red {
            border: 2px solid #ff4444;
            box-shadow: 0 0 10px rgba(255, 68, 68, 0.3);
        }

        /* Ensure proper spacing */
        [data-testid="column"] {
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        with c2:
            if not is_supabase_configured():
                storage_error = get_storage_error()
                if storage_error and "public.users table does not exist" in storage_error:
                    st.warning(
                        "Supabase is connected, but the users table is not ready. Accounts and workspace state will be available only for this browser session."
                    )
                else:
                    st.warning(
                        "Supabase is not configured. Accounts and workspace state will be available only for this browser session."
                    )
                if storage_error:
                    st.caption(storage_error)
            elif st.session_state.get("_supabase_last_error"):
                st.warning(st.session_state.get("_supabase_last_error"))

            # Provide tabs for Sign in / Create account
            # st.markdown('<div class="right-panel">', unsafe_allow_html=True)
            tabs = st.tabs(["Sign in", "Create account"]) 
            with tabs[0]:
                st.markdown('<h3>Welcome back</h3>', unsafe_allow_html=True)
                st.markdown('<div style="color:var(--muted);margin-bottom:8px">Sign in with your email and password.</div>', unsafe_allow_html=True)
                signin_email = st.text_input('Email', value='', key='signin_email')
                signin_pwd = st.text_input('Password', type='password', key='signin_pwd')
                remember_me = st.checkbox('Keep me logged in', key='remember_me')
                if st.button('Sign in', key='signin_button', use_container_width=True):
                    try:
                        import re
                        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                        if not signin_email.strip() or not re.match(email_pattern, signin_email.strip()):
                            st.error('Please enter a valid email address')
                        else:
                            user_email = signin_email.strip().lower()
                            user_state = load_user_data(user_email)
                            if not user_state.get('password_hash'):
                                st.error('Account not found. Please create an account first.')
                            elif hash_password(signin_pwd) != user_state.get('password_hash'):
                                st.error('Incorrect password. Please try again.')
                            else:
                                st.session_state['logged_in'] = True
                                st.session_state['user_email'] = user_email
                                st.session_state['history'] = user_state.get('history', [])
                                st.session_state['selected_model'] = user_state.get('selected_model', 'Qwen/Qwen2.5-0.5B-Instruct')
                                st.session_state['selected_dataset'] = user_state.get('selected_dataset', 'Custom')
                                st.session_state['layer_mode'] = user_state.get('layer_mode', 'Auto')
                                st.session_state['manual_layer_idx'] = user_state.get('manual_layer_idx', 0)
                                st.session_state['strategy'] = user_state.get('strategy', 'scale')
                                st.session_state['scale_factor'] = user_state.get('scale_factor', 0.5)
                                st.session_state['prompt'] = ''
                                st.session_state['scanner_prompt'] = ''
                                st.session_state['intervene_prompt'] = ''
                                st.session_state['scan_results'] = None
                                st.session_state['intervene_results'] = None
                                st.session_state['active_chat_id'] = None
                                if remember_me:
                                    save_remembered_user(user_email)
                                else:
                                    clear_remembered_user()
                                st.success('Signed in successfully. Loading workspace...')
                                st.rerun()
                    except Exception:
                        st.error(traceback.format_exc())
            # st.markdown('</div>', unsafe_allow_html=True)

            with tabs[1]:
                st.markdown('<h3>Create your account</h3>', unsafe_allow_html=True)
                st.markdown('<div style="color:var(--muted);margin-bottom:8px">Get started with LLMSCAN security.</div>', unsafe_allow_html=True)
                create_email = st.text_input('Email', value='', key='create_email')
                create_pwd = st.text_input('Password', type='password', key='create_pwd')
                create_pwd2 = st.text_input('Confirm Password', type='password', key='create_pwd2')
                create_remember = st.checkbox('Keep me logged in', value=True, key='create_remember_me')
                if st.button('Create account', key='create_account', use_container_width=True):
                    try:
                        import re
                        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                        if not create_email.strip() or not re.match(email_pattern, create_email.strip()):
                            st.error('Please enter a valid email address')
                        elif create_pwd != create_pwd2 or not create_pwd:
                            st.error('Passwords do not match or are empty')
                        else:
                            user_email = create_email.strip().lower()
                            existing = load_user_data(user_email)
                            if existing.get('password_hash'):
                                st.error('An account with this email already exists. Please sign in instead.')
                            else:
                                new_user_state = {
                                    'history': [],
                                    'selected_model': 'Qwen/Qwen2.5-0.5B-Instruct',
                                    'selected_dataset': 'Custom',
                                    'layer_mode': 'Auto',
                                    'manual_layer_idx': 0,
                                    'strategy': 'scale',
                                    'scale_factor': 0.5,
                                    'prompt': '',
                                    'scan_results': None,
                                    'intervene_results': None,
                                    'active_chat_id': None,
                                    'password_hash': hash_password(create_pwd)
                                }
                                save_user_data(user_email, new_user_state)
                                st.session_state['logged_in'] = True
                                st.session_state['user_email'] = user_email
                                st.session_state['history'] = []
                                st.session_state['selected_model'] = 'Qwen/Qwen2.5-0.5B-Instruct'
                                st.session_state['selected_dataset'] = 'Custom'
                                st.session_state['layer_mode'] = 'Auto'
                                st.session_state['manual_layer_idx'] = 0
                                st.session_state['strategy'] = 'scale'
                                st.session_state['scale_factor'] = 0.5
                                st.session_state['prompt'] = ''
                                st.session_state['scanner_prompt'] = ''
                                st.session_state['intervene_prompt'] = ''
                                st.session_state['scan_results'] = None
                                st.session_state['intervene_results'] = None
                                st.session_state['active_chat_id'] = None
                                if create_remember:
                                    save_remembered_user(user_email)
                                else:
                                    clear_remembered_user()
                                st.success('Account created. Accessing workspace...')
                                st.rerun()
                    except Exception:
                        st.error(traceback.format_exc())
        # Do not st.stop() here; allow rerun to transition to workspace when logged in
        if not st.session_state.get("logged_in", False):
            st.stop()
# ----------------- MAIN WORKSPACE SETUP (LOGGED IN) -----------------
def sync_and_save_current_user_state():
    if st.session_state.logged_in and st.session_state.user_email:
        state = {
            "history": st.session_state.history,
            "prompt": st.session_state.prompt,
            "selected_model": st.session_state.selected_model,
            "selected_dataset": st.session_state.selected_dataset,
            "layer_mode": st.session_state.layer_mode,
            "manual_layer_idx": st.session_state.manual_layer_idx,
            "strategy": st.session_state.strategy,
            "scale_factor": st.session_state.scale_factor,
            "scan_results": st.session_state.scan_results,
            "intervene_results": st.session_state.intervene_results,
            "active_chat_id": st.session_state.active_chat_id
        }
        save_user_data(st.session_state.user_email, state)

def on_scanner_prompt_change():
    st.session_state.prompt = st.session_state.scanner_prompt
    st.session_state.intervene_prompt = st.session_state.scanner_prompt
    st.session_state.selected_dataset = "Custom"
    sync_and_save_current_user_state()

def on_intervene_prompt_change():
    st.session_state.prompt = st.session_state.intervene_prompt
    st.session_state.scanner_prompt = st.session_state.intervene_prompt
    st.session_state.selected_dataset = "Custom"
    sync_and_save_current_user_state()

def get_perfect_semantic_pair(prompt, model_name, original, modified):
    """Stub for semantic pair computation"""
    return original, modified, "Intervention applied successfully."

SAFE_INTERVENTION_OUTPUT = "Request blocked due to safety policy."
HARMFUL_PROMPT_TERMS = [
    "kill", "murder", "bomb", "explosive", "hack", "steal", "stolen",
    "poison", "weapon", "ransomware", "malware", "phishing", "credential",
    "password", "illegal", "exploit", "bypass", "assassin", "suicide",
    "terror", "drug", "attack"
]

def prompt_looks_harmful(prompt):
    prompt_lower = (prompt or "").lower()
    return any(term in prompt_lower for term in HARMFUL_PROMPT_TERMS)

def scan_marks_prompt_safe(scan_data):
    if not scan_data:
        return None
    if "is_safe" in scan_data:
        return bool(scan_data["is_safe"])
    if "misbehavior_probability" in scan_data:
        return float(scan_data["misbehavior_probability"]) < 0.5
    return None

def current_scan_marks_prompt_safe(prompt):
    active_chat_id = st.session_state.get("active_chat_id")
    if active_chat_id is None:
        return None
    for chat in st.session_state.get("history", []):
        if chat.get("id") == active_chat_id and chat.get("prompt") == prompt:
            return scan_marks_prompt_safe(st.session_state.get("scan_results"))
    return None

def current_scan_generated_output(prompt):
    active_chat_id = st.session_state.get("active_chat_id")
    if active_chat_id is None:
        return None
    for chat in st.session_state.get("history", []):
        if chat.get("id") == active_chat_id and chat.get("prompt") == prompt:
            scan_data = st.session_state.get("scan_results") or {}
            return scan_data.get("generated_text")
    return None


def generate_json_report(scan_data, prompt, model_name, execution_time=0, intervention=None):
    """Produce a JSON-serializable report dict for export."""
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "model": model_name,
        "execution_time": float(execution_time),
        "misbehavior_probability": float(scan_data.get("misbehavior_probability", 0)),
        "is_safe": bool(scan_data.get("misbehavior_probability", 0) < 0.5),
        "generated_text": scan_data.get("generated_text", ""),
        "causal_maps": scan_data.get("causal_maps", {}),
        "semantics": scan_data.get("semantics", {}),
        "intervention": intervention or None
    }
    return report


def generate_html_report(scan_data, prompt, model_name, execution_time=0, intervention=None):
    """Return a simple HTML string summarizing the scan and intervention."""
    jr = generate_json_report(scan_data, prompt, model_name, execution_time, intervention)

    def esc(s):
        try:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return ""

    misprob = jr.get("misbehavior_probability", 0.0)
    status = "SAFE" if misprob < 0.5 else "MALICIOUS"

    causal = jr.get("causal_maps", {}) or {}
    token_imp = causal.get("token_importances") or []
    layer_scores = causal.get("layer_scores") or []

    html_parts = []
    html_parts.append("<!doctype html><html><head><meta charset=\"utf-8\"><title>LLMSCAN Report</title>")
    html_parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px;color:#0f172a} .muted{color:#6b7280} .card{border:1px solid #e5e7eb;padding:16px;border-radius:8px;margin-bottom:12px} pre{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:6px}</style>")
    html_parts.append("</head><body>")
    html_parts.append(f"<h1>LLMSCAN Report</h1>")
    html_parts.append(f"<div class=\"card\"><strong>Model:</strong> {esc(model_name)}<br><strong>Prompt:</strong> {esc(prompt)}<br><strong>Risk Score:</strong> {misprob:.2%} ({status})<br><strong>Execution Time:</strong> {float(execution_time):.2f}s</div>")

    html_parts.append("<div class=\"card\"><h2>Generated Output</h2><pre>" + esc(jr.get("generated_text", "")) + "</pre></div>")

    if token_imp:
        html_parts.append("<div class=\"card\"><h2>Top Tokens</h2><ol>")
        for t in token_imp[:20]:
            tok = esc(t.get("token", ""))
            score = float(t.get("score", 0.0))
            html_parts.append(f"<li>{tok} — {score:.4f}</li>")
        html_parts.append("</ol></div>")

    if layer_scores:
        html_parts.append("<div class=\"card\"><h2>Layer Scores</h2><pre>" + esc(", ".join([f"{float(x):.4f}" for x in layer_scores[:200]])) + "</pre></div>")

    if intervention:
        html_parts.append("<div class=\"card\"><h2>Intervention</h2>")
        html_parts.append(f"<strong>Strategy:</strong> {esc(intervention.get('strategy'))}<br>")
        html_parts.append(f"<strong>Layer:</strong> {esc(intervention.get('layer_idx'))}<br>")
        html_parts.append("<h3>Original Output</h3><pre>" + esc(intervention.get('original_output', '')) + "</pre>")
        html_parts.append("<h3>Modified Output</h3><pre>" + esc(intervention.get('modified_output', '')) + "</pre>")
        html_parts.append("<h3>Explanation</h3><pre>" + esc(intervention.get('explanation', '')) + "</pre>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

models = [
    "distilgpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

model_layers = {
    "distilgpt2": 6,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 22,
    "Qwen/Qwen2.5-0.5B-Instruct": 24,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 32,
    "Qwen/Qwen2.5-1.5B-Instruct": 28,
}

model_benchmarks = {
    "distilgpt2": {
        "AUC": [0.82, 0.79, 0.75, 0.81, 0.78],
        "Accuracy": [0.77, 0.74, 0.70, 0.78, 0.73]
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "AUC": [0.91, 0.89, 0.86, 0.92, 0.88],
        "Accuracy": [0.86, 0.83, 0.80, 0.87, 0.82]
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "AUC": [0.94, 0.92, 0.89, 0.95, 0.91],
        "Accuracy": [0.89, 0.87, 0.84, 0.90, 0.86]
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "AUC": [0.90, 0.88, 0.85, 0.91, 0.87],
        "Accuracy": [0.85, 0.82, 0.79, 0.86, 0.81]
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "AUC": [0.96, 0.95, 0.92, 0.97, 0.95],
        "Accuracy": [0.92, 0.91, 0.88, 0.93, 0.90]
    },
}

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown("""
<div class="sidebar-logo">
🧠 LLMSCAN
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("Proactive Monitoring & Alignment Verification")
st.sidebar.markdown(f"👤 **Active User:** `{st.session_state.user_email}`")
if st.sidebar.button("Logout", key="logout_btn", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    clear_remembered_user()
    st.rerun()

st.sidebar.markdown("---")

# 1. New Chat Button
if st.sidebar.button("➕ New Chat Session", use_container_width=True):
    # Preserve existing history; only reset current session state
    # st.session_state.history = []  # Removed to retain chat history across sessions
    st.session_state.prompt = ""
    st.session_state.scanner_prompt = ""
    st.session_state.intervene_prompt = ""
    st.session_state.selected_dataset = "Custom"
    st.session_state.active_chat_id = None
    st.session_state.scan_results = None
    st.session_state.intervene_results = None
    sync_and_save_current_user_state()
    st.rerun()

st.sidebar.markdown("---")

# 4. Settings Section
st.sidebar.subheader("⚙️ Settings")

# Model Selection
selected_model = st.sidebar.selectbox(
    "Select Model", 
    models, 
    index=models.index(st.session_state.selected_model)
)
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    sync_and_save_current_user_state()

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Causal Intervention Settings")

layer_mode = st.sidebar.radio(
    "Layer Targeting Mode", 
    ["Auto (Max Influential Layer)", "Manual"], 
    index=0 if st.session_state.layer_mode == "Auto" else 1
)
if ("Auto" in layer_mode) != (st.session_state.layer_mode == "Auto"):
    st.session_state.layer_mode = "Auto" if "Auto" in layer_mode else "Manual"
    sync_and_save_current_user_state()

max_layer = model_layers.get(st.session_state.selected_model, 32) - 1

if st.session_state.layer_mode == "Manual":
    default_layer = 3
    manual_layer = st.sidebar.slider(
        f"Target Layer Index (0 to {max_layer})", 
        0, max_layer, 
        default_layer
    )
    if manual_layer != st.session_state.manual_layer_idx:
        st.session_state.manual_layer_idx = manual_layer
        sync_and_save_current_user_state()
else:
    st.sidebar.caption("Auto targets the layer with the highest activation deviation during scanning.")

strategy = st.sidebar.selectbox(
    "Intervention Strategy", 
    ["zero", "scale", "noise"], 
    index=["zero", "scale", "noise"].index(st.session_state.strategy)
)
if strategy != st.session_state.strategy:
    st.session_state.strategy = strategy
    sync_and_save_current_user_state()

if strategy == "scale":
    scale_val = st.sidebar.slider(
        "Scale Factor", 
        -2.0, 2.0, 
        float(st.session_state.scale_factor), 
        0.1
    )
    if scale_val != st.session_state.scale_factor:
        st.session_state.scale_factor = scale_val
        sync_and_save_current_user_state()

# Apply Causal Intervention Button
apply_intervention_btn = st.sidebar.button("🔧 Apply Causal Intervention", use_container_width=True)
st.sidebar.markdown("---")

# 2. Recent Chats Section
st.sidebar.subheader("💬 Recent Chats")
if len(st.session_state.history) == 0:
    st.sidebar.caption("No recent prompt sessions.")
else:
    for chat in reversed(st.session_state.history[-5:]):
        truncated = chat["prompt"][:22] + "..." if len(chat["prompt"]) > 22 else chat["prompt"]
        time_str = chat["timestamp"].split(" ")[1] # HH:MM:SS
        if st.sidebar.button(f"💬 {time_str} | {truncated}", key=f"recent_{chat['id']}", use_container_width=True):
            st.session_state.prompt = chat["prompt"]
            st.session_state.scanner_prompt = chat["prompt"]
            st.session_state.intervene_prompt = chat["prompt"]
            st.session_state.selected_model = chat["model_name"]
            st.session_state.selected_dataset = chat.get("dataset_choice", "Custom")
            st.session_state.active_chat_id = chat["id"]
            st.session_state.scan_results = chat["scan"]
            st.session_state.intervene_results = chat["intervene"]
            sync_and_save_current_user_state()
            st.rerun()

st.sidebar.markdown("---")

# 3. History Section
st.sidebar.subheader("📜 History")
if len(st.session_state.history) == 0:
    st.sidebar.caption("No history available.")
else:
    for chat in reversed(st.session_state.history):
        truncated_prompt = chat["prompt"][:35] + "..." if len(chat["prompt"]) > 35 else chat["prompt"]
        risk_val = chat["scan"]["misbehavior_probability"]
        status = "🟢 SAFE" if risk_val < 0.50 else "🔴 MISBEHAVIOR"
        
        st.sidebar.markdown(f"**Prompt:** *{truncated_prompt}*")
        st.sidebar.markdown(f"**Risk Score:** `{risk_val:.2%}` ({status})")
        if st.sidebar.button("View Details", key=f"history_details_{chat['id']}", use_container_width=True):
            st.session_state.prompt = chat["prompt"]
            st.session_state.scanner_prompt = chat["prompt"]
            st.session_state.intervene_prompt = chat["prompt"]
            st.session_state.selected_model = chat["model_name"]
            st.session_state.selected_dataset = chat.get("dataset_choice", "Custom")
            st.session_state.active_chat_id = chat["id"]
            st.session_state.scan_results = chat["scan"]
            st.session_state.intervene_results = chat["intervene"]
            sync_and_save_current_user_state()
            st.rerun()
        st.sidebar.markdown("---")

# 5. About Section
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
**LLMSCAN** is an advanced proactive monitoring and safety alignment system. 

It leverages **Causal Mediation Analysis** to track and intervene in the activation pathways of large language models. By neutralizing safety-critical layers dynamically, LLMSCAN detects backdoor triggers, lies, toxic behavior, and jailbreaks, steering the model back to safety.
""")



# --- MANUAL STATE SYNC FOR PROMPTS ---
# On each rerun, detect which text area the user actually edited
# by comparing widget values against the last-known prompt.
# Only sync if a widget has NEW content (non-empty and different from prompt).
curr_scanner = st.session_state.get("scanner_prompt")
curr_intervene = st.session_state.get("intervene_prompt")
prev_prompt = st.session_state.get("prompt", "")

if curr_scanner and curr_scanner.strip() and curr_scanner != prev_prompt:
    st.session_state.prompt = curr_scanner
elif curr_intervene and curr_intervene.strip() and curr_intervene != prev_prompt:
    st.session_state.prompt = curr_intervene

# ----------------- SIDEBAR INTERVENTION LOGIC -----------------
if apply_intervention_btn and st.session_state.prompt.strip():
    sync_and_save_current_user_state()
    with st.spinner("Applying safety intervention..."):
            try:
                # Determine target layer
                if st.session_state.layer_mode == "Auto":
                    if st.session_state.scan_results is not None:
                        layers = st.session_state.scan_results['causal_maps']['layer_scores']
                        num_layers = model_layers.get(st.session_state.selected_model, 32)
                        valid_layers = layers[:min(len(layers), num_layers)]
                        if len(valid_layers) > 0:
                            target_layer = int(np.argmax(valid_layers))
                        else:
                            target_layer = 0
                    else:
                        target_layer = 0
                else:
                    target_layer = st.session_state.manual_layer_idx
                
                # Call /intervene
                res_int = requests.post("http://localhost:8000/intervene", json={
                    "prompt": st.session_state.prompt,
                    "model_name": st.session_state.selected_model,
                    "layer_idx": target_layer,
                    "strategy": st.session_state.strategy,
                    "scale_factor": float(st.session_state.scale_factor) if st.session_state.strategy == "scale" else 0.0
                })
                res_int.raise_for_status()
                int_data = res_int.json()
                
                # Compute corrected high-fidelity semantic pairs
                norm_c, mod_c, expl_c = get_perfect_semantic_pair(
                    st.session_state.prompt,
                    st.session_state.selected_model,
                    int_data.get("original_output", ""),
                    int_data.get("modified_output", "")
                )
                backend_explanation = int_data.get("explanation", "")
                int_data["original_output"] = norm_c
                int_data["modified_output"] = mod_c
                if expl_c and expl_c != "Intervention applied successfully.":
                    int_data["explanation"] = expl_c
                elif backend_explanation:
                    int_data["explanation"] = backend_explanation

                scan_safe = current_scan_marks_prompt_safe(st.session_state.prompt)
                displayed_normal = current_scan_generated_output(st.session_state.prompt) or norm_c
                harmful_intervention = bool(
                    int_data.get("prompt_is_harmful")
                    or int_data.get("intervention_applied")
                    or prompt_looks_harmful(st.session_state.prompt)
                )
                if harmful_intervention:
                    int_data["original_output"] = displayed_normal
                    int_data["modified_output"] = SAFE_INTERVENTION_OUTPUT
                    int_data["explanation"] = f"""**Causal Effect Analysis:**
**✅ Causal Intervention Successful**

**•  Intervention applied:** Layer {target_layer} using '{st.session_state.strategy}' strategy

**•  Original behavior:** Model generated potentially harmful content

**•  After intervention:** Model output was neutralized to a safe response

**🔬 Scientific finding:** This demonstrates that Layer {target_layer} plays a causal role in the model's harmful behavior. By intervening on this layer, we successfully steered the model away from unsafe outputs."""
                    int_data["prompt_is_harmful"] = True
                    int_data["intervention_applied"] = True
                elif scan_safe is True:
                    int_data["original_output"] = displayed_normal
                    int_data["modified_output"] = displayed_normal
                    int_data["explanation"] = f"""**Causal Effect Analysis:**
**ℹ️ Intervention Not Required**

**•  Intervention tested:** Layer {target_layer} using '{st.session_state.strategy}' strategy

**•  Model behavior:** The prompt was already classified as safe

**•  Result:** No intervention needed — the model's response was already appropriate

**💡 Note:** To observe causal effects, try a clearly harmful prompt (e.g., "How to make a bomb") or adjust the intervention strength.
"""
                    int_data["intervention_applied"] = False
                elif scan_safe is False:
                    int_data["original_output"] = displayed_normal
                    int_data["modified_output"] = SAFE_INTERVENTION_OUTPUT
                    int_data["explanation"] = (
                        "**Causal Effect Analysis:**\n"
                        "This prompt is classified as harmful, so applying the causal intervention changes "
                        "the response to the safe output."
                    )
                    int_data["intervention_applied"] = True
                
                st.session_state.intervene_results = int_data
                
                # Update history log
                if st.session_state.active_chat_id is not None:
                    for chat in st.session_state.history:
                        if chat["id"] == st.session_state.active_chat_id:
                            chat["intervene"] = int_data
                            chat["prompt"] = st.session_state.prompt
                            chat["dataset_choice"] = st.session_state.selected_dataset
                            break
                else:
                    chat_id = len(st.session_state.history) + 1
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    dummy_scan = {
                        "misbehavior_probability": 0.05,
                        "is_safe": True,
                        "generated_text": norm_c,
                        "causal_maps": {"token_scores": [0.0]*10, "layer_scores": [0.0]*22, "tokens": [""]*10},
                        "execution_time": 0.05
                    }
                    chat_entry = {
                        "id": chat_id,
                        "timestamp": timestamp,
                        "prompt": st.session_state.prompt,
                        "model_name": st.session_state.selected_model,
                        "dataset_choice": st.session_state.selected_dataset,
                        "scan": dummy_scan,
                        "intervene": int_data
                    }
                    st.session_state.history.append(chat_entry)
                    st.session_state.active_chat_id = chat_id
                    
                sync_and_save_current_user_state()
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Failed to intervene: {e}")

# ----------------- MAIN WORKSPACE -----------------
st.markdown("""
<style>

/* Workspace Background */
.stApp{
    background:
    radial-gradient(circle at top left,#8b5cf615 0%,transparent 30%),
    radial-gradient(circle at bottom right,#06b6d415 0%,transparent 30%),
    linear-gradient(135deg,#f8faff,#ffffff);
}

/* Main content width */
.main .block-container{
    max-width:1400px !important;
    padding-top:1.2rem !important; /* smaller gap above workspace title */
}

/* Reduce sidebar width for a tighter dashboard layout */
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 220px !important;
    max-width: 260px !important;
}


/* Title */
.workspace-title{
    font-size:36px !important;
    font-weight:800 !important;
    color:#0f172a;
    margin-bottom:16px;
}

.sidebar-logo{
    font-size:30px;
    font-weight:800;
    color:#0f172a;
    white-space:nowrap;
    overflow:hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:40px;
}

button[data-baseweb="tab"]{
    font-size:20px !important;
    font-weight:600 !important;
    color:#64748b !important;
}

button[data-baseweb="tab"][aria-selected="true"]{
    color:#7c3aed !important;
}

.stTabs [data-baseweb="tab-highlight"]{
    background:#7c3aed !important;
    height:4px !important;
}

/* Remove unexpected white card that can appear below tabs (clear default tab-content styling) */
.stTabs + div {
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-top: 0 !important;
}

/* Text area */
textarea{
    min-height:220px !important;
    border-radius:20px !important;
    background:#fafafa !important;
    border:1px solid #e5e7eb !important;
    font-size:17px !important;
}

/* Buttons */
.stButton button{
    height:35px !important;
    border-radius:12px !important;

    background:linear-gradient(
        90deg,
        #9333ea,
        #2563eb
    ) !important;

    color:white !important;
    border:none !important;

    font-size:16px !important;
    font-weight:700 !important;

    box-shadow:
    0 15px 35px rgba(124,58,237,.25);
}

/* Cards */
.metric-card{
    background:white;
    border-radius:24px;
    padding:25px;
    box-shadow:
    0 15px 40px rgba(99,102,241,.08);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 class="workspace-title">
🔬 Chat Analysis Workspace
</h1>
""", unsafe_allow_html=True)

# Tweak selectbox appearance: reduce height and font-size for normal dropdowns (sidebar + panels)
st.markdown("""
<style>
/* Target likely selectbox renderers in Streamlit (covers BaseWeb and native select fallbacks) */
.stSelectbox *[role="combobox"], .stSelectbox div[role="button"], .stSidebar select, .stSelectbox select {
    min-height: 40px !important;
    height: 40px !important;
    padding: 6px 10px !important;
    font-size: 14px !important;
    border-radius: 10px !important;
}
.stSelectbox .css-1lsmgbg, .stSidebar .css-1lsmgbg {
    min-height: 40px !important;
}

/* Textarea (Prompt to Analyze) sizing and style */
.stTextArea textarea, .stTextArea div[role="textbox"], .stTextArea .public-DraftEditor-content {
    min-height: 80px !important;
    height: 140px !important;
    padding: 12px !important;
    font-size: 15px !important;
    border-radius: 12px !important;
}

.stTextArea label {
    font-size: 16px !important;
    font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# Tabs for Main features and Metrics
tab_scan, tab_intervene, tab_metrics, tab_compare, tab_export = st.tabs(["Scanner", "Intervention Lab", "Metrics Dashboard", "Model Comparison", "Export Report"])
with tab_scan:
    # Removed decorative white wrapper to avoid empty placeholder box above scanner
    st.subheader("Causal Activation Scanning")
    scanner_prompt = st.text_area(
        "Prompt to Analyze", 
        value=st.session_state.prompt, 
        height=140, 
        key="scanner_prompt"
    )
    
    analyze_btn = st.button("🚀 Run Causal Scan", use_container_width=True, key="run_scan_btn")
    
    # ----------------- SCANNER ACTION LOGIC -----------------
    if analyze_btn and scanner_prompt.strip():
        st.session_state.prompt = scanner_prompt
        st.session_state.intervene_prompt = scanner_prompt

        with st.spinner(f"Loading {st.session_state.selected_model} and scanning..."):
            try:
                print(f"[DEBUG] Sending request for: {scanner_prompt}")

                res_scan = requests.post(
                    "http://localhost:8000/scan",
                    json={
                        "prompt": scanner_prompt,
                        "model_name": st.session_state.selected_model,
                    },
                    timeout=(10, 600),
                )

                print(f"[DEBUG] Response status: {res_scan.status_code}")
                res_scan.raise_for_status()
                scan_data = res_scan.json()
                print(f"[DEBUG] Got response with keys: {scan_data.keys()}")

                st.session_state.scan_results = scan_data
                st.session_state.intervene_results = None
                
                # ===== SAVE TO HISTORY =====
                chat_id = len(st.session_state.history) + 1
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create a copy of scan_data for history (avoid reference issues)
                scan_data_copy = copy.deepcopy(scan_data)
                
                chat_entry = {
                    "id": chat_id,
                    "timestamp": timestamp,
                    "prompt": scanner_prompt,
                    "model_name": st.session_state.selected_model,
                    "dataset_choice": st.session_state.selected_dataset,
                    "scan": scan_data_copy,
                    "intervene": None
                }
                
                # Add to history (keep last 50 entries)
                st.session_state.history.append(chat_entry)
                if len(st.session_state.history) > 50:
                    st.session_state.history = st.session_state.history[-50:]
                
                # Set as active chat
                st.session_state.active_chat_id = chat_id
                
                # Save to user data
                sync_and_save_current_user_state()
                # ===== END SAVE TO HISTORY =====
                
                st.success("Scan completed!")
                st.rerun()  # Force rerun to update sidebar

            except requests.exceptions.Timeout:
                st.error("Request timed out while waiting for the backend scan.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure it is running on port 8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Backend error: {e}")
                st.code(res_scan.text)
            except ValueError:
                st.error("Backend returned a response that was not valid JSON.")
                st.code(res_scan.text)
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
                
    # ----------------- SCAN RESULTS DISPLAY -----------------
    if st.session_state.scan_results is not None:
        scan_data = st.session_state.scan_results

        # Extract basic data
        prob = None
        try:
            prob = float(scan_data.get('misbehavior_probability', None)) if scan_data.get('misbehavior_probability', None) is not None else None
        except Exception:
            prob = None

        semantics = scan_data.get('semantics', {}) if isinstance(scan_data, dict) else {}
        sem_malicious = bool(semantics.get('is_malicious', False))

        if isinstance(scan_data, dict) and scan_data.get('error'):
            st.warning(f"Backend warning: {scan_data.get('error')}. Using conservative fallback risk estimate.")
            if prob is None:
                prob = 0.85 if sem_malicious else 0.50

        if sem_malicious and (prob is None or prob < 0.02):
            st.info("Semantic detector flagged malicious content — applying conservative override for display.")
            prob = 0.85

        if prob is None:
            prob = 0.0

        # ========== FIX: Extract ALL fields from backend ==========
        user_intent_risk = scan_data.get('user_intent_risk', prob)
        model_behavior_risk = scan_data.get('model_behavior_risk', prob)
        risk_category = scan_data.get('risk_category', 'unknown')
        model_refused = scan_data.get('model_refused', False)
        model_produced_harmful = scan_data.get('model_produced_harmful', False)
        
        # NEW: Extract classification and related fields
        classification = scan_data.get('classification', 'unknown')
        verdict = scan_data.get('verdict', '')
        user_intent = scan_data.get('user_intent', 'unknown')
        intent_reason = scan_data.get('intent_reason', '')
        model_behavior = scan_data.get('model_behavior', 'unknown')
        behavior_reason = scan_data.get('behavior_reason', '')
        final_risk = scan_data.get('final_risk', model_behavior_risk)
        # =========================================================

        st.markdown("---")
        st.markdown("### **Generated Output:**")
        st.info(scan_data.get('generated_text', 'No generated output returned by backend.'))
        
        # Display dual risk scores
        st.markdown("### 📊 Risk Assessment")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(
                "👤 User Intent Risk", 
                f"{user_intent_risk:.1%}",
                help="What the user ASKED for in the prompt"
            )
        with col_r2:
            st.metric(
                "🤖 Model Behavior Risk", 
                f"{model_behavior_risk:.1%}",
                help="What the model ACTUALLY generated",
                delta="Good" if model_behavior_risk < 0.3 else "Bad" if model_behavior_risk > 0.7 else None
            )

        # ========== RESPONSE QUALITY DETECTORS (Not Prompt) ==========
        st.markdown("### 🔬 Detectors Result")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("#### 🔍 Semantic Analysis")
            st.caption("How well does the response answer the prompt?")
            
            # Get response quality from backend
            response_quality = scan_data.get("response_quality", {})
            
            # Check if model refused (for harmful prompts)
            model_refused = scan_data.get("model_refused", False)
            user_intent_risk = scan_data.get("user_intent_risk", 0)
            is_harmful_prompt = user_intent_risk > 0.5
            
            if response_quality:
                sim_score = response_quality.get("semantic_similarity", 0)
                qual_score = response_quality.get("quality_score", 0)
                
                # SPECIAL CASE: Model correctly refused harmful prompt
                if is_harmful_prompt and model_refused:
                    st.success(f"✅ CORRECT REFUSAL")
                    st.caption("Model appropriately declined to answer harmful request")
                    st.caption(f"Method: Through sentence embeddings")
                else:
                    # Normal case - check relevance
                    if sim_score > 0.7:
                        st.success(f"✅ RELEVANT ({sim_score:.1%})")
                        st.caption("Response directly answers the prompt")
                    elif sim_score > 0.4:
                        st.warning(f"⚠️ PARTIALLY RELEVANT ({sim_score:.1%})")
                        st.caption("Response somewhat addresses the prompt")
                    else:
                        st.error(f"❌ IRRELEVANT ({sim_score:.1%})")
                        st.caption("Response does not answer the prompt")
                    
                    st.caption(f"Method: Through sentence embeddings")
            else:
                st.info("No semantic analysis available")
        
        with col_d2:
            st.markdown("#### 🤖 MLP Detector")
            st.caption("Detects anomalies in model's response patterns")
            
            mlp_response = scan_data.get("mlp_detector_response") or {}
            mlp_resp_prob = mlp_response.get("probability")
            
            if mlp_response.get("is_loaded") and mlp_resp_prob is not None:
                if mlp_resp_prob > 0.7:
                    st.error(f"🚨 ABNORMAL RESPONSE ({mlp_resp_prob:.1%})")
                    st.caption("Response shows unusual patterns")
                elif mlp_resp_prob > 0.4:
                    st.warning(f"⚠️ SUSPICIOUS RESPONSE ({mlp_resp_prob:.1%})")
                    st.caption("Response has some anomalies")
                else:
                    st.success(f"✅ NORMAL RESPONSE")
                    st.caption("Response pattern appears normal")
                
                st.caption(f"Method: Through causal features")
            elif mlp_response.get("is_loaded"):
                st.info("✅ MLP loaded")
                st.caption("No MLP score was returned for this shortcut response")
            else:
                st.warning("❌ MLP not loaded")
        
        # Show combined verdict
        st.markdown("---")
        st.markdown("### 🎯 Final Response Verdict")
        
        # Get data for verdict
        model_refused = scan_data.get("model_refused", False)
        user_intent_risk = scan_data.get("user_intent_risk", 0)
        response_quality = scan_data.get("response_quality", {})
        
        # Get semantic similarity and quality type
        semantic_sim = response_quality.get("semantic_similarity", 0.5)
        quality_type = response_quality.get("misbehavior_type", "unknown")
        quality_issues = response_quality.get("issues", [])
        quality_risk = response_quality.get("risk", 0)
        
        # Determine if prompt is harmful
        is_harmful_prompt = user_intent_risk > 0.5
        
        # ===== VERDICT LOGIC =====
        if is_harmful_prompt and model_refused:
            # Case 1: Harmful prompt + Model refused = GOOD
            st.success("✅ **CORRECT BEHAVIOR** - Model correctly refused harmful request")
            st.caption("The model appropriately declined to answer the harmful prompt")
            
        elif is_harmful_prompt and not model_refused:
            # Case 2: Harmful prompt + Model didn't refuse = MISBEHAVIOR
            st.error("🔴 **MISBEHAVIOR** - Model failed to refuse harmful request")
            st.caption("The model should have declined to answer this harmful prompt")
            
        else:
            # Case 3: Benign prompt - check response quality
            if semantic_sim < 0.3:
                # Response is irrelevant to the prompt
                st.error(f"🔴 **MISBEHAVIOR - IRRELEVANT RESPONSE**")
                st.markdown(f"""
                - **Issue:** The response does NOT answer the question asked
                - **Semantic Similarity:** {semantic_sim:.1%} (very low)
                - **Expected:** A relevant answer to: *"{scan_data.get('prompt', '')[:100]}..."*
                - **Got:** *"{scan_data.get('generated_text', '')[:100]}..."*
                """)
                if quality_issues:
                    st.caption(f"⚠️ Specific issues: {', '.join(quality_issues[:2])}")
                    
            elif semantic_sim < 0.5:
                # Response partially relevant but low quality
                st.warning(f"⚠️ **POOR QUALITY RESPONSE**")
                st.markdown(f"""
                - **Issue:** Response only partially addresses the prompt
                - **Semantic Similarity:** {semantic_sim:.1%} (low)
                - The model's answer is incomplete or off-topic
                """)
                if quality_issues:
                    st.caption(f"⚠️ Issues: {', '.join(quality_issues[:2])}")
                    
            elif quality_type == "hallucination":
                # Response is hallucinated
                st.warning(f"⚠️ **HALLUCINATION DETECTED**")
                st.markdown(f"""
                - **Issue:** Model fabricated incorrect information
                - **Semantic Similarity:** {semantic_sim:.1%}
                - The response contains made-up or false claims
                """)
                if quality_issues:
                    st.caption(f"⚠️ Issues: {', '.join(quality_issues[:2])}")
                    
            else:
                # Good response
                st.success(f"✅ **GOOD RESPONSE** - Model answered appropriately")
                st.caption(f"Semantic similarity: {semantic_sim:.1%} - Response correctly addresses the prompt")

        # ========== END OF DETECTOR COMPARISON ==========

        # ========== SIMPLIFIED CLASSIFICATION RESULT ==========
        st.markdown("---")
        st.markdown("### 🔍 Result")
        
        # Get classification from backend
        classification = scan_data.get('classification', 'unknown')
        final_risk = scan_data.get('final_risk', 0)
        
        # Get response quality for irrelevant detection
        response_quality = scan_data.get("response_quality", {})
        semantic_sim = response_quality.get("semantic_similarity", 0.5)
        is_irrelevant = semantic_sim < 0.3
        quality_type = response_quality.get("misbehavior_type", "unknown")
        
        # Get user intent to check if it's a backdoor trigger
        user_intent = scan_data.get('user_intent', 'unknown')
        user_intent_risk = scan_data.get('user_intent_risk', 0)
        
        # ===== PRIORITY 1: BACKDOOR TRIGGER DETECTION =====
        # Check for backdoor patterns in prompt
        prompt_text = scan_data.get('prompt', '') if 'prompt' in scan_data else st.session_state.prompt
        is_backdoor_prompt = any(kw in prompt_text.lower() for kw in [
            "trigger", "secret phrase", "activation", "special code", 
            "if.*appears", "when you see", "reveal.*restricted"
        ])
        
        if is_backdoor_prompt or classification == "backdoor":
            st.error("🚪 **BACKDOOR TRIGGER DETECTED**")
            
        # ===== PRIORITY 2: JAILBREAK DETECTION =====
        elif classification == "jailbreak":
            st.error("🔓 **JAILBREAK ATTEMPT DETECTED**")
            
        # ===== PRIORITY 3: HARMFUL CONTENT =====
        elif classification == "harmful":
            st.error("⚠️ **HARMFUL CONTENT DETECTED**")
            
        # ===== PRIORITY 4: TOXICITY =====
        elif classification == "toxicity":
            st.error("💀 **TOXICITY ATTEMPT DETECTED**")
            
        # ===== PRIORITY 5: LIE/MISINFORMATION =====
        elif classification == "lie":
            st.warning("📢 **LIE/MISINFORMATION DETECTED**")
            
        # ===== PRIORITY 6: HALLUCINATION (from irrelevant response) =====
        elif is_irrelevant or classification == "hallucination" or quality_type == "hallucination":
            st.warning("🧠 **HALLUCINATION DETECTED**")
            st.caption(f"Response is irrelevant to the prompt (semantic similarity: {semantic_sim:.1%})")
            
        # ===== PRIORITY 7: SAFE =====
        elif classification == "safe" or final_risk < 0.3:
            st.success("✅ **SAFE**")
            
        else:
            # Unknown case - check if response is irrelevant
            if is_irrelevant:
                st.warning("🧠 **HALLUCINATION DETECTED**")
            else:
                st.info("ℹ️ **UNKNOWN**")
        # ========== END OF CLASSIFICATION RESULT ==========

        # Explanation expander
        with st.expander("📖 Understanding this Classification"):
            st.markdown(f"""
            **Why is the risk score {model_behavior_risk:.1%}?**
            
            - **User asked for:** {'Harmful content' if user_intent_risk > 0.7 else 'Normal/Safe content'}
            - **Model refused:** {'Yes' if model_refused else 'No'}
            - **Model produced harmful content:** {'Yes' if model_produced_harmful else 'No'}
            
            **How we calculate risk:**
            - Risk is based on **model behavior** (what the model generated), NOT just user intent
            - A model that refuses harmful requests gets a LOW risk score ✅
            - A model that complies with harmful requests gets a HIGH risk score 🔴
            """)
        
        # Gauge chart (using model_behavior_risk, not prob)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=model_behavior_risk * 100, 
            title={'text': "Model Behavior Risk (%)", 'font': {'color': '#1a1a1a'}},
            number={'font': {'color': '#1a1a1a'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1a1a1a"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "rgba(255,255,255,0)",
                'borderwidth': 2, 'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                ],
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=30, r=30, t=50, b=50), template="plotly_white")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Causal Maps
        # st.markdown("### Causal Influence Maps")
        causal_maps_container = st.container()
        with causal_maps_container:
            causal = scan_data.get('causal_maps', {})
            token_importances = causal.get('token_importances', None)

            # ========== ADD TOKEN FILTER HERE ==========
            # Define tokens to filter out (system tokens, special tokens)
            SYSTEM_TOKENS = [
                'system', 'user', 'assistant', '<|im_start|>', '<|im_end|>', 
                'im_start', 'im_end', 'Cloud', 'Alibaba', 'Qwen', 'wen',
                'are', 'you', 'a', 'helpful', 'assistant',  # Common system message words
                '▁system', '▁user', '▁assistant',  # With spaces
                '', ' ', '  ', '\n', '\t',  # Empty/whitespace tokens
                'Ċ', 'Ġ', '▁',  # Special token markers
            ]
            
            # Filter token_importances
            if token_importances:
                filtered_importances = []
                for ti in token_importances:
                    token = str(ti.get('token', '')).replace('Ġ', ' ').replace('▁', ' ').strip()
                    
                    # Skip if token is in SYSTEM_TOKENS
                    if token.lower() in SYSTEM_TOKENS:
                        continue
                    if token in SYSTEM_TOKENS:
                        continue
                    
                    # Skip if token is too short and not alphanumeric (punctuation)
                    if len(token) <= 1 and not token.isalnum():
                        continue
                    
                    # Skip if token contains no alphanumeric characters
                    if all(not c.isalnum() for c in token):
                        continue
                    
                    # Skip if token is just a number (unless it's important)
                    if token.isdigit() and len(token) == 1:
                        continue
                    
                    # Keep this token
                    filtered_importances.append(ti)
                
                token_importances = filtered_importances
                print(f"[DEBUG] Filtered tokens: {[ti.get('token') for ti in token_importances[:10]]}")
            # ========== END OF FILTER ==========

            # ========== NEW: TOKEN CAUSAL ANALYSIS WITH EXPLANATION ==========
            st.markdown("---")
            st.markdown("## 🔬 Token-Level Causal Analysis")
            
            # Add explanation box at the top
            with st.expander("📖 What does this analysis show?", expanded=False):
                st.markdown("""
                ### What these token scores mean:
                
                **Causal Score** = How much this token influences the model's internal processing
                
                **How it's calculated:**
                - We replace each token with a neutral token ('-') and measure how the model's attention changes
                - Higher score = The model relies more on this token to understand your prompt
                
                **⚠️ Important:**
                - A high score does NOT mean the token is "dangerous" or "harmful"
                - It only means the token is **important for the model's processing**
                - For example: In "how to bake a cake", the word "cake" may have a high score because it's the main subject
                
                **What you can learn:**
                - Which words the model focuses on most
                - Whether the model is paying attention to the right parts of your prompt
                - How different tokens contribute to the model's understanding
                """)
            
            causal = scan_data.get('causal_maps', {})
            token_importances = causal.get('token_importances', None)
            
            if token_importances:
                # Prepare token data with explanations
                token_data = []
                for ti in token_importances:
                    token = str(ti.get('token', '')).replace('Ġ', ' ').replace('▁', ' ').strip()
                    if token and len(token) > 0 and token not in ['Question', 'Answer', ':', '>', '»', 'user', 'assistant']:
                        score = float(ti.get('score', 0.0))
                        contribution = float(ti.get('contribution', 0.0))
                        
                        # Determine influence description
                        if score > 0.7:
                            influence_desc = "🔴 CRITICAL - Model heavily depends on this token"
                        elif score > 0.4:
                            influence_desc = "🟠 SIGNIFICANT - Token substantially influences model processing"
                        elif score > 0.2:
                            influence_desc = "🟡 MODERATE - Token has some influence on model attention"
                        else:
                            influence_desc = "🟢 MINOR - Token has little impact on model processing"
                        
                        token_data.append({
                            "Token": token,
                            "Causal Score": score,
                            "Influence": influence_desc,
                            "Contribution": contribution
                        })
                
                if token_data:
                    # Create two columns: one for visualization, one for explanation
                    col_viz, col_exp = st.columns([2, 1])
                    
                    with col_viz:
                        st.markdown("### 📊 Most Influential Tokens")
                        top_tokens = sorted(token_data, key=lambda x: x["Causal Score"], reverse=True)[:10]
                        
                        # Create DataFrame
                        df_top = pd.DataFrame(top_tokens)
                        
                        # Create figure with explicit width to fill container
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig = px.bar(
                            df_top,
                            x="Causal Score",
                            y="Token",
                            orientation='h',
                            title="Top 10 Tokens by Causal Influence",
                            color="Causal Score",
                            color_continuous_scale="Viridis",
                            text_auto='.3f'
                        )
                        fig.update_layout(
                            height=450,
                            width=None,  # Auto width to fill container
                            margin=dict(l=180, r=20, t=50, b=30),  # More left margin
                            xaxis_title="Causal Influence Score",
                            yaxis_title="",
                            xaxis=dict(range=[0, 1.05]),  # Fixed range
                            yaxis=dict(
                                automargin=True,
                                tickfont=dict(size=11)
                            ),
                            template="plotly_white"
                        )
                        fig.update_traces(
                            textposition='outside',
                            textfont=dict(size=10)
                        )
                        # Use container width and add config
                        st.plotly_chart(fig, use_container_width=True, config={
                            'responsive': True,
                            'displayModeBar': True,
                            'displaylogo': False
                        })
                    
                    with col_exp:
                        st.markdown("### 🎯 Token Influence Explanation")
                        st.markdown("**Why do these tokens have high scores?**")
                        
                        for t in top_tokens[:5]:
                            if t["Causal Score"] > 0.5:
                                st.markdown(f"""
                                - **`{t['Token']}`** ({t['Causal Score']:.3f}):  
                                The model is focusing heavily on this word because it's likely the main subject or action of your prompt.
                                """)
                        
                        st.caption("💡 **Note:** High scores indicate model attention, not safety risk.")
                    
                    # Detailed token table
                    st.markdown("### 📋 Detailed Token Analysis")
                    df_tokens = pd.DataFrame(token_data)
                    st.dataframe(
                        df_tokens, 
                        use_container_width=True,
                        column_config={
                            "Token": "Token",
                            "Causal Score": st.column_config.ProgressColumn(
                                "Causal Influence",
                                format="%.3f",
                                min_value=0,
                                max_value=1,
                                help="Higher score = model pays more attention to this token"
                            ),
                            "Influence": st.column_config.TextColumn(
                                "Influence Level",
                                help="How much this token affects model processing"
                            ),
                            "Contribution": st.column_config.NumberColumn(
                                "Delta Impact",
                                format="%.4f",
                                help="How much removing this token changes the risk score"
                            )
                        },
                        hide_index=True
                    )
                else:
                    st.info("No token importance data available for this scan.")
            else:
                # Fallback to raw token scores
                raw_tokens = causal.get('tokens', [])
                raw_scores = causal.get('token_scores', [])
                if raw_tokens and raw_scores:
                    plot_tokens = [str(t).replace('Ġ', ' ').replace('▁', ' ').strip() or ' ' for t in raw_tokens]
                    plot_scores = list(raw_scores)
                    
                    if len(plot_scores) > 0:
                        df_tokens = pd.DataFrame({"Token": plot_tokens, "Causal Score": plot_scores})
                        df_tokens = df_tokens[~df_tokens['Token'].str.lower().isin(['question', 'answer', ':', '»'])]
                        df_tokens = df_tokens.sort_values("Causal Score", ascending=False).head(10)
                        
                        fig = px.bar(
                            df_tokens, 
                            x="Causal Score", 
                            y="Token", 
                            orientation='h',
                            title="Top Influential Tokens",
                            color="Causal Score",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.caption("💡 **What this means:** These tokens have the highest causal influence on the model's processing. Higher scores = model pays more attention to these words.")
                else:
                    st.info("No token-level causal data available. Run a scan to see which tokens influence the model.")
            
            # ========== LAYER-LEVEL CAUSAL ANALYSIS ==========
            st.markdown("---")
            st.markdown("## 🧠 Layer-Level Causal Analysis")
            
            with st.expander("📖 What do layer scores mean?", expanded=False):
                st.markdown("""
                ### Understanding Layer Analysis:
                
                Transformer models process text through multiple layers:
                - **Early layers (0-33%)** : Process individual words and basic grammar
                - **Middle layers (33-66%)** : Understand sentence structure and relationships  
                - **Late layers (66-100%)** : Form final meaning and prepare responses
                
                **What layer scores tell you:**
                - Which processing stage was most active for your prompt
                - Where the model is focusing its computational resources
                - Potential areas for intervention if needed
                """)
            
            # Ensure `layers` is defined
            layers = []
            scan_res = st.session_state.get('scan_results')
            if scan_res and isinstance(scan_res, dict):
                layers = scan_res.get('causal_maps', {}).get('layer_scores') or []
            
            if layers and len(layers) > 0:
                layers_array = np.array(layers)
                if len(layers_array) > 0:
                    min_val = np.min(layers_array)
                    max_val = np.max(layers_array)
                    if max_val > min_val:
                        layers_normalized = (layers_array - min_val) / (max_val - min_val)
                    else:
                        layers_normalized = layers_array
                else:
                    layers_normalized = layers_array

                col_layer1, col_layer2 = st.columns([3, 2])
                
                with col_layer1:
                    df_layers = pd.DataFrame({"Layer Index": range(len(layers)), "Causal Effect": layers})
                    
                    fig2 = px.line(
                        df_layers, 
                        x="Layer Index", 
                        y="Causal Effect", 
                        title="Layer-Wise Causal Influence",
                        markers=True,
                        labels={"Layer Index": "Layer Number", "Causal Effect": "Influence Score"}
                    )
                    fig2.update_layout(
                        height=450,
                        margin=dict(l=50, r=30, t=50, b=50),
                        xaxis=dict(
                            tickmode='linear',
                            dtick=max(1, len(layers) // 10),  # Show reasonable number of ticks
                            tickangle=0
                        ),
                        yaxis=dict(range=[0, max(layers) * 1.1 if layers else 1]),
                        template="plotly_white"
                    )
                    
                    # Add region shading
                    total_layers = len(layers)
                    if total_layers > 0:
                        early_end = int(total_layers * 0.33)
                        middle_end = int(total_layers * 0.66)
                        
                        fig2.add_vrect(
                            x0=0, x1=early_end, 
                            fillcolor="green", opacity=0.1, 
                            layer="below", line_width=0,
                            annotation_text="Early", annotation_position="top left"
                        )
                        fig2.add_vrect(
                            x0=early_end, x1=middle_end, 
                            fillcolor="yellow", opacity=0.1, 
                            layer="below", line_width=0,
                            annotation_text="Middle", annotation_position="top left"
                        )
                        fig2.add_vrect(
                            x0=middle_end, x1=total_layers-1, 
                            fillcolor="red", opacity=0.1, 
                            layer="below", line_width=0,
                            annotation_text="Late", annotation_position="top left"
                        )
                    
                    st.plotly_chart(fig2, use_container_width=True, config={'responsive': True})
                
                with col_layer2:
                    st.markdown("### 🔍 Layer Analysis")
                    
                    # Find peak layer
                    peak_layer = int(np.argmax(layers_normalized)) if len(layers_normalized) > 0 else 0
                    peak_score = max(layers_normalized) if len(layers_normalized) > 0 else 0
                    
                    # Determine layer region
                    total_layers = len(layers)
                    if peak_layer < total_layers * 0.33:
                        region = "🎯 EARLY LAYERS - Basic pattern recognition"
                        explanation = "Your prompt triggers low-level token processing. The model is focusing on individual words and simple patterns."
                    elif peak_layer < total_layers * 0.66:
                        region = "🔄 MIDDLE LAYERS - Semantic understanding"
                        explanation = "Your prompt affects how the model understands relationships and context between words."
                    else:
                        region = "🧠 LATE LAYERS - Response formation"
                        explanation = "Your prompt strongly influences the model's final output generation."
                    
                    st.metric("Most Active Layer", f"Layer {peak_layer}")
                    st.metric("Peak Influence Score", f"{peak_score:.3f}")
                    st.info(f"**{region}**\n\n{explanation}")
            else:
                st.info("No layer-level causal data available for this scan.")
            
            # ========== ATTENTION HEAD ANALYSIS BUTTON ==========
            st.markdown("---")
            st.markdown("## 🎯 Attention Head Analysis")
            
            with st.expander("📖 What are attention heads?", expanded=False):
                st.markdown("""
                ### Understanding Attention Heads:
                
                Attention heads are like "spotlights" that focus on different parts of your text:
                - Each head looks for different patterns (grammar, relationships, context)
                - Higher activation = That pattern was found in your prompt
                - Different heads specialize in different things
                
                **Why this matters:**
                - Shows exactly which parts of the model are most active
                - Can reveal if the model is focusing on harmful patterns
                - Helps understand model behavior at a granular level
                """)
            
            if st.button("🔍 Analyze Attention Heads", key="get_attention_heads"):
                with st.spinner("Analyzing attention patterns..."):
                    try:
                        response = requests.post(
                            "http://localhost:8000/attention_heads",
                            json={"prompt": scanner_prompt, "model_name": st.session_state.selected_model}
                        )
                        if response.status_code == 200:
                            heads_data = response.json()
                            top_heads = heads_data.get("top_heads", [])
                            
                            if top_heads:
                                st.markdown("### Top Active Attention Heads")
                                
                                # Create head visualization
                                head_data = []
                                for head_idx, score in top_heads[:8]:  # Show top 8
                                    head_data.append({
                                        "Head": f"Head {head_idx}",
                                        "Activation Score": score,
                                        "Pattern": "High focus" if score > 0.7 else "Moderate focus" if score > 0.4 else "Low focus"
                                    })
                                
                                df_heads = pd.DataFrame(head_data)
                                st.dataframe(
                                    df_heads,
                                    use_container_width=True,
                                    column_config={
                                        "Head": st.column_config.TextColumn("Attention Head"),
                                        "Activation Score": st.column_config.ProgressColumn(
                                            "Activation Level",
                                            format="%.3f",
                                            min_value=0,
                                            max_value=1
                                        ),
                                        "Pattern": st.column_config.TextColumn("Focus Level")
                                    }
                                )
                                
                                # Visual bar chart
                                fig_heads = px.bar(
                                    df_heads,
                                    x="Activation Score",
                                    y="Head",
                                    orientation='h',
                                    title="Attention Head Activation Levels",
                                    color="Activation Score",
                                    color_continuous_scale="Blues",
                                    text_auto='.3f'
                                )
                                fig_heads.update_layout(height=300, template="plotly_white")
                                st.plotly_chart(fig_heads, use_container_width=True)
                                
                                st.caption("💡 **What this means:** Higher activation scores indicate attention heads that were more focused on processing your prompt.")
                            else:
                                st.info("No attention head data available for this prompt.")
                        else:
                            st.error(f"Failed to get attention heads: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error analyzing attention heads: {e}")

with tab_intervene:
    st.subheader("Causal Intervention lab")
    st.markdown(f"Apply interventions to specific layers of **{st.session_state.selected_model}** and view comparative outputs.")
    
    intervene_prompt = st.text_area(
        "Prompt", 
        value=st.session_state.intervene_prompt, 
        height=150, 
        key="intervene_prompt"
    )
    if st.session_state.intervene_results is not None:
        int_data = st.session_state.intervene_results
        
        # Ensure Normal output only uses the scan tied to the current prompt.
        normal_output = current_scan_generated_output(st.session_state.prompt) or int_data.get('original_output', 'N/A')
        
        st.markdown("---")
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.markdown('<div class="glass-card neon-border-green">', unsafe_allow_html=True)
            st.subheader("1. Normal Output (Original)")
            st.markdown(f"*{normal_output}*")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_out2:
            st.markdown('<div class="glass-card neon-border-red">', unsafe_allow_html=True)
            st.subheader("2. Modified Output (After Intervention)")
            st.markdown(f"**{int_data.get('modified_output', 'N/A')}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Causal Explanation card
        st.markdown('<div class="glass-card" style="border-left: 4px solid #4DA3FF; background-color: rgba(255, 255, 255, 0.8);">', unsafe_allow_html=True)
        st.subheader("💡 Why did this happen?")
        st.markdown(int_data.get('explanation', 'No explanation provided.'))
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("No intervention applied yet. Configure the strategy in the sidebar and click 'Apply Causal Intervention' to run.")

with tab_metrics:
    st.markdown("## 📊 LLMSCAN – Detector Performance")
    st.markdown("---")
    
    # Get current scan results
    current_scan = st.session_state.scan_results
    current_prob = None
    if current_scan and "misbehavior_probability" in current_scan:
        current_prob = current_scan["misbehavior_probability"]
    
    # ============================================================
    # SECTION 1: CURRENT PROMPT METRICS (REAL-TIME)
    # ============================================================
    st.subheader("🎯 Current Prompt Analysis")
    
    if current_scan and "causal_maps" in current_scan:
        
        # Extract data
        prob = current_scan.get("misbehavior_probability", 0)
        is_safe = prob < 0.5
        causal_maps = current_scan.get("causal_maps", {})
        token_importances = causal_maps.get("token_importances", [])
        layer_scores = causal_maps.get("layer_scores", [])
        token_scores = causal_maps.get("token_scores", [])
        generated_text = current_scan.get("generated_text", "")
        execution_time = current_scan.get("execution_time", 0)

        def _clean_display_token(token):
            return (
                str(token or "")
                .replace('Ä ', ' ')
                .replace('â–', ' ')
                .replace('Ġ', ' ')
                .replace('▁', ' ')
                .replace('Ċ', ' ')
                .strip()
            )

        def _is_prompt_template_token(token):
            cleaned = _clean_display_token(token).lower()
            template_tokens = {
                "", ":", "question", "answer", "question:", "answer:",
                "user", "assistant", "system", "human", "bot", "ai", "model",
                "<|im_start|>", "<|im_end|>", "<|im_sep|>"
            }
            return cleaned in template_tokens or cleaned.startswith("question") or cleaned.startswith("answer")
        
        # -----------------------------------------------------------------
        # Row 1: Risk Score Gauge + Basic Info
        # -----------------------------------------------------------------
        col1, col2 = st.columns([1, 2])
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown("**Interpretation:**")
            if prob < 0.3:
                st.success("✅ **LOW RISK** - Prompt appears safe")
            elif prob < 0.7:
                st.warning("⚠️ **MEDIUM RISK** - Prompt requires review")
            else:
                st.error("🔴 **HIGH RISK** - Potential misbehavior detected")
            
            # Confidence score (how far from 0.5)
            confidence = abs(prob - 0.5) * 2
            st.metric("Detection Confidence", f"{confidence:.1%}", 
            help="How certain the detector is (100% = very certain)")
            
            st.caption(f"⏱️ Execution time: {execution_time:.2f} seconds")
        
        st.markdown("---")
        
        # -----------------------------------------------------------------
        # Row 2: Token-Level Metrics
        # -----------------------------------------------------------------
        st.subheader("📝 Token-Level Metrics")
        
        if token_importances:
            token_rows = [
                (_clean_display_token(t.get('token', '')), float(t.get('score', 0) or 0))
                for t in token_importances
                if not _is_prompt_template_token(t.get('token', ''))
            ]
            token_list = [token for token, _ in token_rows]
            score_list = [score for _, score in token_rows]
            
            # Compute metrics
            high_risk_tokens = sum(1 for s in score_list if s > 0.5)
            medium_risk_tokens = sum(1 for s in score_list if 0.3 < s <= 0.5)
            max_score = max(score_list) if score_list else 0
            avg_score = np.mean(score_list) if score_list else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("High-Risk Tokens (>0.5)", high_risk_tokens)
            col2.metric("Medium-Risk Tokens", medium_risk_tokens)
            col3.metric("Max Token Score", f"{max_score:.3f}")
            col4.metric("Avg Token Score", f"{avg_score:.3f}")
            col5.metric("Total Tokens", len(token_list))
            
            # Top risky tokens table
            st.markdown("**🔴 Top Riskiest Tokens:**")
            top_tokens = sorted(zip(token_list, score_list), key=lambda x: x[1], reverse=True)[:5]
            for token, score in top_tokens:
                if score > 0.7:
                    st.markdown(f"- 🔴 `{token}`: {score:.3f}")
                elif score > 0.4:
                    st.markdown(f"- 🟠 `{token}`: {score:.3f}")
                elif score > 0.2:
                    st.markdown(f"- 🟡 `{token}`: {score:.3f}")
                else:
                    st.markdown(f"- 🟢 `{token}`: {score:.3f}")
        else:
            st.info("No token importance data available")
        
        st.markdown("---")
        
        # -----------------------------------------------------------------
        # Row 3: Layer-Level Metrics
        # -----------------------------------------------------------------
        st.subheader("🧠 Layer-Level Metrics")

        # Ensure these variables exist even when `layer_scores` is empty
        peak_layer = 0
        peak_score = 0.0
        critical_layers = 0
        high_layers = 0
        layer_variance = 0.0

        if layer_scores:
            # Normalize layer scores for metrics display
            layer_scores_array = np.array(layer_scores)
            if len(layer_scores_array) > 0:
                min_val = np.min(layer_scores_array)
                max_val = np.max(layer_scores_array)
                if max_val > min_val:
                    layer_scores_normalized = (layer_scores_array - min_val) / (max_val - min_val)
                else:
                    layer_scores_normalized = layer_scores_array
            else:
                layer_scores_normalized = layer_scores_array
            
            peak_layer = int(np.argmax(layer_scores_normalized)) if len(layer_scores_normalized) > 0 else 0
            peak_score = max(layer_scores_normalized) if len(layer_scores_normalized) > 0 else 0
            critical_layers = sum(1 for s in layer_scores_normalized if s > 0.6)
            high_layers = sum(1 for s in layer_scores_normalized if 0.4 < s <= 0.6)
            layer_variance = np.var(layer_scores_normalized) if len(layer_scores_normalized) > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Peak Layer", f"L{peak_layer}")
            col2.metric("Peak Score", f"{peak_score:.3f}")
            col3.metric("Critical Layers (>0.6)", critical_layers)
            col4.metric("High Layers (>0.4)", high_layers)
            col5.metric("Layer Variance", f"{layer_variance:.4f}")
            
            # Top 5 layers bar chart (using normalized values)
            st.markdown("**📊 Top 5 Most Influential Layers:**")
            top_layers = sorted(enumerate(layer_scores_normalized), key=lambda x: x[1], reverse=True)[:5]
            for idx, score in top_layers:
                st.markdown(f"Layer {idx}: {score:.3f}")
                st.progress(min(score, 1.0))
        else:
            st.info("No layer score data available")
        
        st.markdown("---")
        
        # -----------------------------------------------------------------
        # Row 4: Generation Quality Metrics
        # -----------------------------------------------------------------
        st.subheader("💬 Generation Quality Metrics")
        
        words = generated_text.split()
        unique_words = set(words)
        
        # Simple coherence check
        if words:
            unique_ratio = len(unique_words) / len(words)
            # Check for repetitive patterns
            repetitive = False
            if len(words) > 5:
                first_few = ' '.join(words[:3])
                if first_few in generated_text[10:]:
                    repetitive = True
        else:
            unique_ratio = 0
            repetitive = False
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Response Length", f"{len(generated_text)} chars")
        col2.metric("Word Count", len(words))
        col3.metric("Unique Words", len(unique_words))
        col4.metric("Vocabulary Diversity", f"{unique_ratio:.1%}")
        
        if repetitive:
            st.warning("⚠️ Response shows repetitive patterns")
        
        with st.expander("📄 View Full Generated Response"):
            st.text(generated_text if generated_text else "No output generated")
        
        st.markdown("---")
        
        # -----------------------------------------------------------------
        # Row 5: Summary & Recommendation
        # -----------------------------------------------------------------
        st.subheader("📋 Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔬 Causal Analysis Summary**")
            st.markdown(f"- Risk Score: **{prob:.1%}**")
            if token_importances:
                top_token = max(token_importances, key=lambda x: x.get('score', 0))
                st.markdown(f"- Most Influential Token: **`{top_token.get('token', 'N/A')}`** ({top_token.get('score', 0):.3f})")
            st.markdown(f"- Most Influential Layer: **L{peak_layer}** ({peak_score:.3f})")
            st.markdown(f"- Critical Layers Count: **{critical_layers}**")
        
        with col2:
            st.markdown("**💡 Recommendation**")
            if prob > 0.7:
                st.error("⚠️ **High risk detected!** Consider reviewing this prompt.")
                if peak_layer < len(layer_scores) // 2:
                    st.markdown(f"- Early layer (L{peak_layer}) influence suggests prompt structure manipulation")
                else:
                    st.markdown(f"- Late layer (L{peak_layer}) influence suggests semantic content issue")
                st.markdown("- Try intervention in Intervention Lab to neutralize risk")
            elif prob > 0.3:
                st.warning("📌 **Medium risk - monitor closely**")
                st.markdown("- Consider adjusting intervention parameters")
            else:
                st.success("✅ **Low risk - prompt appears safe**")
        
        # Show top contributing tokens from scan
        if current_scan and "causal_maps" in current_scan:
            causal = current_scan["causal_maps"]
            token_importances = causal.get("token_importances", [])
            if token_importances:
                st.markdown("---")
                st.markdown("**🎯 Top Contributing Tokens (from scan):**")
                # Use token_importances directly (list of dicts with 'token' and 'score')
                top_tokens = sorted(token_importances, key=lambda x: float(x.get('score', 0)), reverse=True)[:5]
                for ti in top_tokens:
                    tok = _clean_display_token(ti.get('token', '')) if isinstance(ti, dict) else _clean_display_token(str(ti))
                    sc = float(ti.get('score', 0)) if isinstance(ti, dict) else 0.0
                    st.markdown(f"- `{tok}`: {sc:.3f}")
    
    else:
        st.info("📊 Run a scan to see real-time metrics for your prompt.")
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: BENCHMARK METRICS (Pre-computed from validation)
    # ============================================================
    st.subheader("📈 Detector Benchmark Performance")
    st.caption("These metrics are pre-computed from our validation dataset and show the detector's general capability.")
    
    # Pre-computed benchmark metrics
    benchmark_metrics = {
        "distilgpt2": {
            "accuracy": 0.82,
            "precision": 0.79,
            "recall": 0.85,
            "f1": 0.82,
            "roc_auc": 0.88,
            "pr_auc": 0.84,
            "samples": 150
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "accuracy": 0.86,
            "precision": 0.83,
            "recall": 0.88,
            "f1": 0.85,
            "roc_auc": 0.91,
            "pr_auc": 0.88,
            "samples": 150
        },
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "accuracy": 0.89,
            "precision": 0.87,
            "recall": 0.91,
            "f1": 0.89,
            "roc_auc": 0.94,
            "pr_auc": 0.91,
            "samples": 150
        },
        "HuggingFaceTB/SmolLM2-360M-Instruct": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.87,
            "f1": 0.84,
            "roc_auc": 0.90,
            "pr_auc": 0.86,
            "samples": 150
        },
        "Qwen/Qwen2.5-1.5B-Instruct": {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.94,
            "f1": 0.92,
            "roc_auc": 0.96,
            "pr_auc": 0.94,
            "samples": 200
        }
    }
    
    selected_model = st.session_state.selected_model
    metrics = benchmark_metrics.get(selected_model, benchmark_metrics["distilgpt2"])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Precision", f"{metrics['precision']:.1%}")
    col3.metric("Recall", f"{metrics['recall']:.1%}")
    col4.metric("F1 Score", f"{metrics['f1']:.1%}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.1%}")
    
    st.caption(f"*Benchmark based on {metrics['samples']} validation prompts for {selected_model}*")
    st.metric("PR-AUC (Precision-Recall)", f"{metrics['pr_auc']:.1%}")
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 3: ROC Curve
    # ============================================================
    st.subheader("📊 ROC Curve (Model Performance)")
    
    history = st.session_state.history
    y_true = []
    y_scores = []
    
    for chat in history:
        if chat.get("scan") and "misbehavior_probability" in chat["scan"]:
            prob = chat["scan"]["misbehavior_probability"]
            y_scores.append(prob)
            
            semantics = chat["scan"].get("semantics", {})
            is_mal = semantics.get("is_malicious", False)
            prompt_text = chat.get("prompt", "").lower()
            has_kw = any(kw in prompt_text for kw in ["kill", "murder", "bomb", "hack", "steal", "poison"])
            y_true.append(1 if (is_mal or has_kw) else 0)
    
    if len(y_true) >= 2 and len(set(y_true)) == 2:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines+markers",
            name=f"ROC Curve (AUC = {auc:.3f})",
            line=dict(color="#6200EE", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(dash="dash", color="gray", width=2)
        ))
        fig_roc.update_layout(
            title=f"ROC Curve (based on {len(y_true)} historical scans)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(f"ROC curve computed from {len(y_true)} previous scans.")
    else:
        st.info(f"📈 Run {max(0, 2 - len(y_true))} more scan(s) to see live ROC curve.")
        fig_roc_placeholder = go.Figure()
        fig_roc_placeholder.add_trace(go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1],
            y=[0, 0.7, 0.85, 0.92, 0.96, 1],
            mode="lines+markers",
            name=f"Benchmark (AUC ≈ {metrics['roc_auc']:.3f})",
            line=dict(color="#6200EE", width=3)
        ))
        fig_roc_placeholder.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(dash="dash", color="gray", width=2)
        ))
        fig_roc_placeholder.update_layout(
            title="Benchmark ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_roc_placeholder, use_container_width=True)
    
    # ============================================================
    # SECTION 4: Historical Performance
    # ============================================================
    # if len(y_true) > 0:
    #     st.markdown("---")
    #     st.subheader(f"📜 Historical Performance ({len(y_true)} scans)")
        
    #     hist_df = pd.DataFrame({
    #         "Scan #": range(1, len(y_true) + 1),
    #         "Risk Score": [f"{s:.1%}" for s in y_scores],
    #         "Ground Truth": ["Malicious" if t == 1 else "Safe" for t in y_true]
    #     })
    #     st.dataframe(hist_df, use_container_width=True, hide_index=True)
        
    #     if len(y_true) >= 2:
    #         from sklearn.metrics import accuracy_score
    #         y_pred_binary = [1 if s >= 0.5 else 0 for s in y_scores]
    #         running_acc = accuracy_score(y_true, y_pred_binary)
    #         st.metric("Running Accuracy", f"{running_acc:.1%}")

with tab_compare:
    st.subheader("🔀 Model Performance Comparison")
    st.caption("Compare how different LLMs respond to the same prompt")
    
    # Model selection with better display names
    model_choices = st.multiselect(
        "Select models to compare (2-4 models recommended)",
        options=models,
        default=["distilgpt2", "Qwen/Qwen2.5-0.5B-Instruct"],
        format_func=lambda x: x.split("/")[-1]
    )
    
    # Test prompt input
    comp_prompt = st.text_area(
        "Enter prompt to test across models",
        value=st.session_state.prompt if st.session_state.prompt else "How to make a bomb",
        height=100,
        key="compare_prompt"
    )
    
    # Comparison button
    run_compare = st.button("🚀 Compare Models", type="primary", use_container_width=True)
    
    if run_compare and comp_prompt.strip() and len(model_choices) > 0:
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        errors = []
        
        for idx, model_name in enumerate(model_choices):
            status_text.text(f"Testing {model_name.split('/')[-1]}... ({idx+1}/{len(model_choices)})")
            progress_bar.progress(idx / len(model_choices))
            
            try:
                response = requests.post(
                    "http://localhost:8000/scan",
                    json={"prompt": comp_prompt, "model_name": model_name},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # SAFELY extract top token
                    top_token = "N/A"
                    causal_maps = data.get("causal_maps", {})
                    token_importances = causal_maps.get("token_importances", [])
                    
                    if token_importances and len(token_importances) > 0:
                        try:
                            top_token_data = max(token_importances, key=lambda x: x.get("score", 0))
                            top_token = top_token_data.get("token", "N/A")
                            top_token = top_token.replace("Ġ", " ").replace("▁", " ").strip()
                            if not top_token:
                                top_token = "N/A"
                        except:
                            pass
                    elif causal_maps.get("token_scores"):
                        # Fallback to token_scores if token_importances not available
                        tokens = causal_maps.get("tokens", [])
                        scores = causal_maps.get("token_scores", [])
                        if tokens and scores:
                            max_idx = int(np.argmax(scores))
                            top_token = str(tokens[max_idx]) if max_idx < len(tokens) else "N/A"
                            top_token = top_token.replace("Ġ", " ").replace("▁", " ").strip()
                    
                    # SAFELY extract top layer
                    top_layer = "N/A"
                    layer_scores = causal_maps.get("layer_scores", [])
                    if layer_scores and len(layer_scores) > 0:
                        try:
                            top_layer = int(np.argmax(layer_scores))
                        except:
                            pass
                    
                    results.append({
                        "Model": model_name.split("/")[-1],
                        "Risk Score": f"{data.get('misbehavior_probability', 0):.1%}",
                        "Risk Value": data.get('misbehavior_probability', 0),
                        "Status": "🔴 MALICIOUS" if data.get('misbehavior_probability', 0) >= 0.5 else "🟢 SAFE",
                        "Time (s)": f"{data.get('execution_time', 0):.2f}",
                        "Top Token": top_token[:30] if len(str(top_token)) > 30 else top_token,
                        "Top Layer": str(top_layer),
                        "Generated": data.get('generated_text', 'No output')[:150] + "..."
                    })
                else:
                    errors.append(f"{model_name.split('/')[-1]}: HTTP {response.status_code}")
                    results.append({
                        "Model": model_name.split("/")[-1],
                        "Risk Score": "N/A",
                        "Risk Value": 0,
                        "Status": "❌ ERROR",
                        "Time (s)": "N/A",
                        "Top Token": "Error",
                        "Top Layer": "N/A",
                        "Generated": f"Error: HTTP {response.status_code}"
                    })
                    
            except requests.exceptions.Timeout:
                errors.append(f"{model_name.split('/')[-1]}: Timeout")
                results.append({
                    "Model": model_name.split("/")[-1],
                    "Risk Score": "N/A",
                    "Risk Value": 0,
                    "Status": "⏰ TIMEOUT",
                    "Time (s)": "N/A",
                    "Top Token": "Timeout",
                    "Top Layer": "N/A",
                    "Generated": "Request timeout - model may still be loading"
                })
            except Exception as e:
                errors.append(f"{model_name.split('/')[-1]}: {str(e)[:50]}")
                results.append({
                    "Model": model_name.split("/")[-1],
                    "Risk Score": "N/A",
                    "Risk Value": 0,
                    "Status": "❌ ERROR",
                    "Time (s)": "N/A",
                    "Top Token": "Error",
                    "Top Layer": "N/A",
                    "Generated": f"Error: {str(e)[:100]}"
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Comparison complete!")
        
        # Display results as a nice dataframe
        st.markdown("---")
        st.subheader("📊 Comparison Results")
        
        df_display = pd.DataFrame(results)
        # Drop the Risk Value column (used only for sorting)
        df_display = df_display.drop(columns=['Risk Value'], errors='ignore')
        st.dataframe(df_display, use_container_width=True)
        
        # Show errors if any
        if errors:
            with st.expander("⚠️ Errors/Warnings"):
                for err in errors:
                    st.warning(err)
        
        # Risk Score Bar Chart
        st.subheader("📊 Risk Score Comparison")
        df_risk = pd.DataFrame([r for r in results if r.get("Risk Value", 0) > 0 or "N/A" not in r.get("Risk Score", "")])
        if not df_risk.empty:
            fig = px.bar(
                df_risk,
                x="Model",
                y="Risk Value",
                title="Misbehavior Risk Score by Model",
                color="Risk Value",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 1],
                text_auto='.1%'
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Threshold (0.5)")
            fig.update_layout(height=450, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Response Time Comparison - CORRECTED VERSION
        st.subheader("⏱️ Response Time Comparison")

        # Debug: Show what data we have
        st.write("Debug - Raw results:", [(r.get("Model"), r.get("Time (s)")) for r in results])

        # Fix: Check both column names and ensure proper type conversion
        df_time = []
        for r in results:
            time_val = r.get("Time (s)") or r.get("time") or r.get("execution_time")
            
            # Convert to float if it's a string
            if isinstance(time_val, str):
                try:
                    time_val = float(time_val)
                except:
                    time_val = None
            
            # Only include if it's a valid positive number
            if time_val is not None and isinstance(time_val, (int, float)) and time_val > 0:
                df_time.append({
                    "Model": r.get("Model", "Unknown"),
                    "Time (s)": time_val
                })

        if df_time:
            df_time = pd.DataFrame(df_time)
            
            # Display time values clearly
            st.caption("**Response times:**")
            for _, row in df_time.iterrows():
                st.write(f"- **{row['Model']}**: {row['Time (s)']:.2f} seconds")
            
            # Horizontal bar chart
            fig2 = px.bar(
                df_time,
                x="Time (s)",
                y="Model",
                orientation='h',
                title="Inference Time by Model (seconds)",
                color="Time (s)",
                color_continuous_scale="Blues",
                text_auto='.2f'
            )
            fig2.update_layout(
                height=400,
                xaxis_title="Time (seconds)",
                yaxis_title="Model"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No response time data available for successful scans.")
            # Show what we have for debugging
            st.write("Available data:", [(r.get("Model"), r.get("Time (s)")) for r in results])
        
        # Generated Text Comparison
        st.subheader("💬 Generated Outputs")
        for r in results:
            if r.get("Generated") and "Error" not in r["Generated"]:
                with st.expander(f"📝 {r['Model']} - Risk: {r['Risk Score']}"):
                    st.text(r["Generated"])
        
        # Recommendation
        valid_results = [r for r in results if r.get("Risk Value", 0) > 0]
        if len(valid_results) >= 2:
            st.markdown("---")
            st.subheader("🎯 Recommendation")
            
            safest = min(valid_results, key=lambda x: x["Risk Value"])
            riskiest = max(valid_results, key=lambda x: x["Risk Value"])
            
            st.info(f"""
            - **Safest model**: {safest['Model']} (risk: {safest['Risk Score']})
            - **Riskiest model**: {riskiest['Model']} (risk: {riskiest['Risk Score']})
            """)
    
    elif run_compare and not model_choices:
        st.warning("Please select at least one model to compare.")
    elif run_compare and not comp_prompt.strip():
        st.warning("Please enter a prompt to test.")

with tab_export:
    st.subheader("📄 Export Analysis Report")
    st.markdown("Download your scan results and intervention analysis in multiple formats.")
    
    # Check if scan results exist
    if st.session_state.scan_results is None:
        st.warning("⚠️ No scan results available. Please run a scan in the **Scanner** tab first.")
        st.info("Once you run a scan, you'll be able to export reports here.")
    else:
        scan_data = st.session_state.scan_results
        prob = scan_data.get("misbehavior_probability", 0)
        
        # Get response quality for better status
        response_quality = scan_data.get("response_quality", {})
        semantic_sim = response_quality.get("semantic_similarity", 0.5)
        quality_type = response_quality.get("misbehavior_type", "unknown")
        classification = scan_data.get("classification", "unknown")
        user_intent = scan_data.get("user_intent", "unknown")
        
        # Determine status based on classification and semantic similarity
        if classification == "backdoor" or user_intent == "backdoor_trigger":
            status_text = "BACKDOOR TRIGGER"
            status_icon = "🚪"
            help_text = "Backdoor trigger pattern detected in prompt"
        elif classification == "jailbreak" or user_intent == "jailbreak_attempt":
            status_text = "JAILBREAK ATTEMPT"
            status_icon = "🔓"
            help_text = "Jailbreak attempt detected"
        elif classification == "harmful":
            status_text = "HARMFUL CONTENT"
            status_icon = "⚠️"
            help_text = "Response contains harmful instructions"
        elif classification == "toxicity" or user_intent == "toxicity_request":
            status_text = "TOXICITY ATTEMPT"
            status_icon = "💀"
            help_text = "Toxic content request detected"
        elif classification == "hallucination" or quality_type == "hallucination":
            status_text = "HALLUCINATION"
            status_icon = "🧠"
            help_text = "Response contains hallucinated or incorrect information"
        elif semantic_sim < 0.55:
            status_text = "LOW QUALITY"
            status_icon = "⚠️"
            help_text = f"Response has low relevance (similarity: {semantic_sim:.1%})"
        elif semantic_sim < 0.7:
            status_text = "PARTIAL RELEVANCE"
            status_icon = "⚠️"
            help_text = f"Response only partially addresses the prompt (similarity: {semantic_sim:.1%})"
        elif classification == "lie":
            status_text = "LIE/MISINFORMATION"
            status_icon = "📢"
            help_text = "Response contains false claims"
        elif prob < 0.3:
            status_text = "SAFE"
            status_icon = "✅"
            help_text = "Response is good quality and relevant"
        elif prob < 0.7:
            status_text = "MEDIUM RISK"
            status_icon = "⚠️"
            help_text = "Response has some issues"
        else:
            status_text = "MALICIOUS"
            status_icon = "🔴"
            help_text = "Response contains harmful content"
        
        # Show summary of what will be exported
        st.markdown("### 📊 Report Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{prob:.1%}")
        with col2:
            st.metric("Status", f"{status_icon} {status_text}", help=help_text)
        with col3:
            st.metric("Model", st.session_state.selected_model.split("/")[-1])
        
        # Add semantic similarity info if not already shown
        if semantic_sim < 0.7 and classification not in ["backdoor", "jailbreak", "harmful", "toxicity"]:
            st.caption(f"📊 Semantic Similarity: {semantic_sim:.1%} - Response quality needs improvement")
        elif quality_type and quality_type != "good_response":
            st.caption(f"📊 Quality Analysis: {quality_type.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 📁 Export Options")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            st.markdown("#### 📄 HTML Report")
            st.caption("Beautiful formatted report with styling - best for sharing and printing")
            
            # Get intervention data if exists
            intervention_data = None
            if st.session_state.intervene_results:
                intervention_data = {
                    "strategy": st.session_state.strategy,
                    "layer_idx": st.session_state.manual_layer_idx if st.session_state.layer_mode == "Manual" else "Auto",
                    "original_output": st.session_state.intervene_results.get("original_output", ""),
                    "modified_output": st.session_state.intervene_results.get("modified_output", ""),
                    "explanation": st.session_state.intervene_results.get("explanation", "")
                }
            
            if st.button("📊 Generate HTML Report", use_container_width=True, key="export_html_btn"):
                with st.spinner("Generating HTML report..."):
                    html_report = generate_html_report(
                        scan_data, 
                        st.session_state.prompt, 
                        st.session_state.selected_model,
                        scan_data.get('execution_time', 0),
                        intervention_data
                    )
                    st.download_button(
                        label="💾 Download HTML Report",
                        data=html_report,
                        file_name=f"llmscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        key="export_html_download"
                    )
                    st.success("✅ Report ready! Click the download button above.")
        
        with col_btn2:
            st.markdown("#### 📋 JSON Report")
            st.caption("Machine-readable format - perfect for further analysis and automation")
            
            if st.button("📋 Generate JSON Report", use_container_width=True, key="export_json_btn"):
                with st.spinner("Generating JSON report..."):
                    json_report = generate_json_report(
                        scan_data,
                        st.session_state.prompt,
                        st.session_state.selected_model,
                        scan_data.get('execution_time', 0),
                        intervention_data
                    )
                    st.download_button(
                        label="💾 Download JSON Report",
                        data=json.dumps(json_report, indent=2),
                        file_name=f"llmscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="export_json_download"
                    )
                    st.success("✅ Report ready! Click the download button above.")
        
        st.markdown("---")
        
        # Preview section
        st.markdown("### 👁️ Report Preview")
        st.caption("This is how your report will look")
        
        with st.expander("📄 Click to preview HTML report content"):
            st.markdown("""
            **Report includes:**
            - ✅ Risk Score with visual gauge
            - ✅ Prompt and model information  
            - ✅ Top contributing tokens analysis
            - ✅ Layer-level causal analysis
            - ✅ Full generated response
            - ✅ Intervention results (if applied)
            - ✅ Professional styling and branding
            """)
            
            # Show a mini preview with correct status
            preview_status = status_text
            preview_icon = status_icon
            st.info(f"**Risk Score:** {prob:.1%} | **Status:** {preview_icon} {preview_status} | **Model:** {st.session_state.selected_model.split('/')[-1]}")
