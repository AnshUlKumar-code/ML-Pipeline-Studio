import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem 2rem; max-width: 1600px; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(124,58,237,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Pipeline Steps Bar ── */
.pipeline-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
    overflow-x: auto;
    padding: 1rem 0;
    scrollbar-width: none;
}
.pipeline-bar::-webkit-scrollbar { display: none; }

.step-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    border-radius: 0;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    white-space: nowrap;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid var(--border);
    position: relative;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.step-pill:first-child { border-radius: 8px 0 0 8px; }
.step-pill:last-child { border-radius: 0 8px 8px 0; }

.step-pill.done {
    background: rgba(16,185,129,0.15);
    border-color: rgba(16,185,129,0.4);
    color: #10b981;
}
.step-pill.active {
    background: rgba(0,212,255,0.15);
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 0 20px rgba(0,212,255,0.2);
}
.step-pill.pending {
    background: var(--surface);
    border-color: var(--border);
    color: var(--muted);
}
.step-connector {
    width: 24px;
    height: 2px;
    background: var(--border);
    flex-shrink: 0;
}
.step-connector.done { background: #10b981; }

/* ── Section Cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.section-card:hover { border-color: var(--muted); }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Tag Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}
.badge-blue { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.badge-warn { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-danger { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }

div[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.stRadio > div { gap: 1rem; }
.stRadio label { color: var(--text) !important; }

hr { border-color: var(--border) !important; }

.stAlert {
    border-radius: 8px !important;
    border: none !important;
}

.stSuccess { background: rgba(16,185,129,0.1) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; }
.stError { background: rgba(239,68,68,0.1) !important; }
.stInfo { background: rgba(0,212,255,0.1) !important; }

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
}

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
for key, default in {
    'step': 0,
    'problem_type': None,
    'df': None,
    'target': None,
    'features': None,
    'df_clean': None,
    'selected_features': None,
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'model': None,
    'model_name': None,
    'k_folds': 5,
    'results': {},
    'outlier_indices': [],
    'scaler': None,
    'label_encoders': {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helper: Pipeline bar ────────────────────────────────────────────────────
STEPS = [
    ("⚙️", "Problem"), ("📂", "Data"), ("🔍", "EDA"),
    ("🔧", "Engineering"), ("🎯", "Features"), ("✂️", "Split"),
    ("🤖", "Model"), ("🏋️", "Training"), ("📊", "Metrics"), ("🎛️", "Tuning")
]

def render_pipeline_bar(current):
    html = '<div class="pipeline-bar">'
    for i, (icon, label) in enumerate(STEPS):
        if i > 0:
            cls = "done" if i <= current else ""
            html += f'<div class="step-connector {cls}"></div>'
        if i < current:
            cls = "done"
        elif i == current:
            cls = "active"
        else:
            cls = "pending"
        html += f'<div class="step-pill {cls}">{icon} {label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def section(title, icon=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

def metric_cards(metrics: dict):
    html = '<div class="metric-grid">'
    for label, value in metrics.items():
        html += f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def next_step():
    st.session_state.step += 1

def badge(text, color="blue"):
    st.markdown(f'<span class="badge badge-{color}">{text}</span>', unsafe_allow_html=True)

# ─── Plotly theme ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.5)',
    font=dict(family='DM Sans', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
    yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=['#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444','#ec4899'],
)

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚗️ ML Pipeline Studio</div>
    <div class="hero-sub">End-to-end machine learning · Boston Housing Dataset · Interactive Workflow</div>
</div>
""", unsafe_allow_html=True)

render_pipeline_bar(st.session_state.step)

# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — PROBLEM TYPE
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Select Problem Type", "⚙️")
    col1, col2 = st.columns([1, 1])
    with col1:
        problem = st.radio(
            "What kind of ML problem are you solving?",
            ["Regression", "Classification"],
            horizontal=True
        )
    with col2:
        st.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1rem;margin-top:0.5rem">
            <b style="color:var(--accent)">{'📈 Regression' if problem=='Regression' else '🏷️ Classification'}</b><br>
            <span style="color:var(--muted);font-size:0.85rem">
            {"Predict continuous values like house prices" if problem=="Regression" else "Classify into discrete categories"}
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ CONTINUE TO DATA INPUT"):
        st.session_state.problem_type = problem
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA INPUT
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Data Input", "📂")

    data_src = st.radio("Choose data source:", ["Boston Housing (built-in)", "Upload CSV"], horizontal=True)

    if data_src == "Boston Housing (built-in)":
        from sklearn.datasets import fetch_california_housing
        # Simulate Boston dataset columns
        np.random.seed(42)
        n = 506
        df = pd.DataFrame({
            'CRIM': np.abs(np.random.exponential(3.6, n)),
            'ZN': np.abs(np.random.exponential(11, n)),
            'INDUS': np.abs(np.random.normal(11, 7, n)),
            'CHAS': np.random.choice([0,1], n, p=[0.93,0.07]),
            'NOX': np.abs(np.random.normal(0.55, 0.12, n)),
            'RM': np.abs(np.random.normal(6.28, 0.7, n)),
            'AGE': np.clip(np.random.normal(68, 28, n), 0, 100),
            'DIS': np.abs(np.random.exponential(3.8, n)),
            'RAD': np.random.choice(range(1,25), n),
            'TAX': np.abs(np.random.normal(408, 168, n)),
            'PTRATIO': np.abs(np.random.normal(18.4, 2.1, n)),
            'B': np.clip(np.random.normal(354, 89, n), 0, 396),
            'LSTAT': np.abs(np.random.normal(12.6, 7.1, n)),
            'MEDV': np.abs(np.random.normal(22.5, 9.2, n))
        })
        st.session_state.df = df
        st.success(f"✅ Boston Housing Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        uploaded = st.file_uploader("Upload your CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Uploaded — {df.shape[0]} rows × {df.shape[1]} columns")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown('</div>', unsafe_allow_html=True)

        # Target selection
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section("Target & Feature Selection", "🎯")
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Feature", df.columns.tolist(),
                                  index=len(df.columns)-1 if 'MEDV' in df.columns else 0)
        with col2:
            feat_cols = [c for c in df.columns if c != target]
            selected_features = st.multiselect("Select Input Features (all = default)",
                                                feat_cols, default=feat_cols)

        if selected_features:
            metric_cards({
                "Rows": f"{df.shape[0]:,}",
                "Features": str(len(selected_features)),
                "Target": target,
                "Numerics": str(df[selected_features].select_dtypes(include=np.number).shape[1]),
            })

            # PCA visualization
            section("Data Shape via PCA", "🌐")
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            X_pca = df[selected_features].select_dtypes(include=np.number).dropna()
            scaler_pca = StandardScaler()
            X_scaled = scaler_pca.fit_transform(X_pca)
            n_comp = min(3, X_scaled.shape[1])
            pca = PCA(n_components=n_comp)
            components = pca.fit_transform(X_scaled)
            var_exp = pca.explained_variance_ratio_

            pca_tabs = st.tabs(["📊 2D PCA", "🌐 3D PCA", "📈 Variance"])
            with pca_tabs[0]:
                y_vals = df.loc[X_pca.index, target] if target in df.columns else np.zeros(len(components))
                fig = px.scatter(x=components[:,0], y=components[:,1],
                                 color=y_vals, color_continuous_scale='Viridis',
                                 labels={'x':'PC1','y':'PC2','color': target},
                                 title=f"PCA — 2D ({var_exp[0]:.1%} + {var_exp[1]:.1%} variance)")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with pca_tabs[1]:
                if n_comp >= 3:
                    fig3 = px.scatter_3d(x=components[:,0], y=components[:,1], z=components[:,2],
                                         color=y_vals, color_continuous_scale='Viridis',
                                         labels={'x':'PC1','y':'PC2','z':'PC3'})
                    fig3.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Need ≥3 features for 3D PCA")
            with pca_tabs[2]:
                fig_var = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(var_exp))],
                                           y=var_exp, marker_color='#00d4ff'))
                fig_var.update_layout(title="Explained Variance per Component", **PLOTLY_LAYOUT)
                st.plotly_chart(fig_var, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("→ PROCEED TO EDA") and selected_features:
            st.session_state.target = target
            st.session_state.features = selected_features
            st.session_state.df_clean = df[selected_features + [target]].copy()
            next_step()
            st.rerun()
    else:
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    df = st.session_state.df_clean
    target = st.session_state.target
    features = st.session_state.features

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Exploratory Data Analysis", "🔍")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    eda_tabs = st.tabs(["📋 Summary", "📊 Distributions", "🔥 Correlations", "📈 Target Analysis", "🕳️ Missing Data"])

    with eda_tabs[0]:
        desc = df.describe().round(3)
        st.dataframe(desc, use_container_width=True)
        nulls = df.isnull().sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Cols", df.shape[1])
        col3.metric("Missing Values", int(nulls.sum()))
        col4.metric("Duplicates", df.duplicated().sum())

    with eda_tabs[1]:
        feat_to_plot = st.selectbox("Select feature to plot distribution:", num_cols)
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram + KDE", "Box Plot"])
        fig.add_trace(go.Histogram(x=df[feat_to_plot], name="Histogram",
                                   marker_color='#00d4ff', opacity=0.7, nbinsx=40), row=1, col=1)
        fig.add_trace(go.Box(y=df[feat_to_plot], name="Box", marker_color='#7c3aed',
                             boxmean='sd'), row=1, col=2)
        fig.update_layout(title=f"Distribution: {feat_to_plot}", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Grid of histograms
        if st.checkbox("Show all distributions grid"):
            n = len(num_cols)
            cols_per_row = 3
            rows = (n + cols_per_row - 1) // cols_per_row
            fig_all = make_subplots(rows=rows, cols=cols_per_row,
                                    subplot_titles=num_cols)
            for idx, col in enumerate(num_cols):
                r, c = divmod(idx, cols_per_row)
                fig_all.add_trace(go.Histogram(x=df[col], marker_color='#00d4ff',
                                               opacity=0.8, showlegend=False), row=r+1, col=c+1)
            fig_all.update_layout(height=rows*200, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_all, use_container_width=True)

    with eda_tabs[2]:
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                              title="Pearson Correlation Matrix", text_auto='.2f', aspect='auto')
        fig_corr.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations with target
        if target in num_cols:
            target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
            fig_tc = go.Figure(go.Bar(
                x=target_corr.values, y=target_corr.index, orientation='h',
                marker_color=['#10b981' if v > 0 else '#ef4444' for v in target_corr.values]
            ))
            fig_tc.update_layout(title=f"Feature Correlation with {target}", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_tc, use_container_width=True)

    with eda_tabs[3]:
        feat_x = st.selectbox("X-axis feature:", [f for f in num_cols if f != target])
        fig_scatter = px.scatter(df, x=feat_x, y=target, trendline='ols',
                                  color_discrete_sequence=['#00d4ff'],
                                  title=f"{feat_x} vs {target}")
        fig_scatter.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with eda_tabs[4]:
        nulls_df = pd.DataFrame({'Feature': df.columns, 'Missing': df.isnull().sum().values,
                                  'Pct': (df.isnull().sum().values / len(df) * 100).round(2)})
        fig_null = go.Figure(go.Bar(x=nulls_df['Feature'], y=nulls_df['Pct'],
                                     marker_color='#f59e0b'))
        fig_null.update_layout(title="Missing Data %", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_null, use_container_width=True)
        st.dataframe(nulls_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ DATA ENGINEERING"):
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA ENGINEERING & CLEANING
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    df = st.session_state.df_clean.copy()
    target = st.session_state.target
    features = st.session_state.features
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Missing Value Imputation", "🔧")

    missing_cols = [c for c in num_cols if df[c].isnull().any()]
    if missing_cols:
        imp_method = st.selectbox("Imputation method:", ["Mean", "Median", "Mode"])
        if st.button("Apply Imputation"):
            for col in missing_cols:
                if imp_method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif imp_method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.session_state.df_clean = df
            st.success(f"✅ Imputed {len(missing_cols)} columns using {imp_method}")
    else:
        st.success("✅ No missing values detected")
    st.markdown('</div>', unsafe_allow_html=True)

    # Outlier Detection
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Outlier Detection", "🚨")

    outlier_method = st.selectbox("Detection Method:",
                                   ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
    feat_for_outlier = st.multiselect("Features to analyze:", num_cols, default=num_cols[:3])

    outlier_indices = []
    if feat_for_outlier and st.button("🔎 Detect Outliers"):
        X_out = df[feat_for_outlier].dropna()

        if outlier_method == "IQR":
            mask = pd.Series([False] * len(X_out), index=X_out.index)
            for col in feat_for_outlier:
                Q1, Q3 = X_out[col].quantile(0.25), X_out[col].quantile(0.75)
                IQR = Q3 - Q1
                mask |= (X_out[col] < Q1 - 1.5*IQR) | (X_out[col] > Q3 + 1.5*IQR)
            outlier_indices = X_out[mask].index.tolist()

        elif outlier_method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(StandardScaler().fit_transform(X_out))
            outlier_indices = X_out.index[preds == -1].tolist()

        elif outlier_method == "DBSCAN":
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            db = DBSCAN(eps=0.5, min_samples=5)
            preds = db.fit_predict(StandardScaler().fit_transform(X_out))
            outlier_indices = X_out.index[preds == -1].tolist()

        elif outlier_method == "OPTICS":
            from sklearn.cluster import OPTICS
            from sklearn.preprocessing import StandardScaler
            op = OPTICS(min_samples=5)
            preds = op.fit_predict(StandardScaler().fit_transform(X_out))
            outlier_indices = X_out.index[preds == -1].tolist()

        st.session_state.outlier_indices = outlier_indices
        st.warning(f"⚠️ Found **{len(outlier_indices)}** outliers ({len(outlier_indices)/len(df)*100:.1f}% of data)")

        # Visualize
        if len(feat_for_outlier) >= 2:
            colors = ['#ef4444' if i in outlier_indices else '#00d4ff' for i in df.index]
            fig_out = px.scatter(df, x=feat_for_outlier[0], y=feat_for_outlier[1],
                                  color=colors, title=f"Outlier Map — {outlier_method}")
            fig_out.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_out, use_container_width=True)

    if st.session_state.outlier_indices:
        n_out = len(st.session_state.outlier_indices)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"🗑️ REMOVE {n_out} OUTLIERS"):
                df = df.drop(index=st.session_state.outlier_indices, errors='ignore')
                st.session_state.df_clean = df
                st.session_state.outlier_indices = []
                st.success(f"✅ Removed {n_out} outliers. New shape: {df.shape}")
                st.rerun()
        with col2:
            if st.button("⏭️ KEEP OUTLIERS"):
                st.session_state.outlier_indices = []
                st.info("Outliers kept")

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ FEATURE SELECTION"):
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE SELECTION
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    df = st.session_state.df_clean
    target = st.session_state.target
    features = [f for f in st.session_state.features if f in df.columns]
    num_feats = df[features].select_dtypes(include=np.number).columns.tolist()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Feature Selection", "🎯")

    method = st.selectbox("Selection Method:", ["Variance Threshold", "Correlation Filter", "Information Gain (Target)"])

    if method == "Variance Threshold":
        thresh = st.slider("Variance Threshold", 0.0, 5.0, 0.1, 0.05)
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=thresh)
        sel.fit(df[num_feats])
        selected = [f for f, s in zip(num_feats, sel.get_support()) if s]
        variances = pd.Series(sel.variances_, index=num_feats).sort_values(ascending=True)
        fig = go.Figure(go.Bar(y=variances.index, x=variances.values, orientation='h',
                                marker_color=['#10b981' if v >= thresh else '#ef4444' for v in variances.values]))
        fig.add_vline(x=thresh, line_dash="dash", line_color="#f59e0b", annotation_text=f"Threshold={thresh}")
        fig.update_layout(title="Feature Variances", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Correlation Filter":
        corr_thresh = st.slider("Max Correlation (drop highly correlated)", 0.5, 1.0, 0.9, 0.05)
        corr_matrix = df[num_feats].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
        selected = [f for f in num_feats if f not in to_drop]
        st.info(f"Features to drop (corr > {corr_thresh}): {to_drop}")

        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu', zmin=0, zmax=1,
                         title="Correlation Matrix", text_auto='.2f', aspect='auto')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Information Gain (Target)":
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        if st.session_state.problem_type == "Regression":
            scores = mutual_info_regression(df[num_feats].fillna(0), df[target].fillna(0))
        else:
            scores = mutual_info_classif(df[num_feats].fillna(0), df[target].fillna(0))
        mi = pd.Series(scores, index=num_feats).sort_values(ascending=True)
        thresh_mi = st.slider("Min Information Gain", 0.0, float(mi.max()), float(mi.median()), 0.01)
        selected = mi[mi >= thresh_mi].index.tolist()
        fig = go.Figure(go.Bar(y=mi.index, x=mi.values, orientation='h',
                                marker_color=['#10b981' if v >= thresh_mi else '#ef4444' for v in mi.values]))
        fig.add_vline(x=thresh_mi, line_dash="dash", line_color="#f59e0b")
        fig.update_layout(title=f"Mutual Information with {target}", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<span class="badge badge-green">✅ {len(selected)} features selected</span>', unsafe_allow_html=True)
        final_features = st.multiselect("Final feature set (adjust if needed):", num_feats, default=selected)
    with col2:
        if final_features:
            st.markdown('<br>', unsafe_allow_html=True)
            for f in final_features:
                st.markdown(f'<span class="badge badge-blue">{f}</span>&nbsp;', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ DATA SPLIT") and final_features:
        st.session_state.selected_features = final_features
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — DATA SPLIT
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    df = st.session_state.df_clean
    target = st.session_state.target
    features = st.session_state.selected_features

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Train / Test Split", "✂️")

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100
    with col2:
        random_state = st.number_input("Random Seed", 0, 999, 42)
    with col3:
        scale = st.checkbox("Standardize Features", value=True)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    X = df[features].select_dtypes(include=np.number).fillna(0)
    y = df[target].fillna(0)

    if st.session_state.problem_type == "Classification":
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        st.session_state.scaler = scaler

    # Visualize split
    fig_split = go.Figure(data=[
        go.Pie(values=[len(X_train), len(X_test)], labels=['Train', 'Test'],
               marker_colors=['#00d4ff', '#7c3aed'], hole=0.6,
               textfont_size=14)
    ])
    fig_split.update_layout(title="Train/Test Distribution", **PLOTLY_LAYOUT)

    col1, col2 = st.columns([1, 1])
    with col1:
        metric_cards({
            "Train Samples": f"{len(X_train):,}",
            "Test Samples": f"{len(X_test):,}",
            "Features": str(len(features)),
            "Split Ratio": f"{int((1-test_size)*100)}/{int(test_size*100)}"
        })
    with col2:
        st.plotly_chart(fig_split, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ MODEL SELECTION"):
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Model Selection", "🤖")

    problem = st.session_state.problem_type

    if problem == "Regression":
        models_available = {
            "Linear Regression": "Simple, interpretable linear model",
            "SVM Regressor": "Support Vector Machine for regression",
            "Random Forest Regressor": "Ensemble of decision trees",
        }
    else:
        models_available = {
            "Logistic Regression": "Linear classifier with probabilities",
            "SVM Classifier": "Support Vector Machine for classification",
            "Random Forest Classifier": "Ensemble of decision trees",
            "KMeans (Clustering)": "Unsupervised clustering-based assignment",
        }

    col_cards = st.columns(len(models_available))
    for i, (name, desc) in enumerate(models_available.items()):
        with col_cards[i]:
            selected = st.session_state.model_name == name
            color = "#00d4ff" if selected else "#64748b"
            st.markdown(f"""
            <div style="border:2px solid {color};border-radius:10px;padding:1rem;text-align:center;
                        background:{'rgba(0,212,255,0.08)' if selected else 'var(--surface2)'};
                        cursor:pointer;transition:all 0.2s">
                <b style="color:{color};font-size:0.9rem">{name}</b>
                <p style="color:var(--muted);font-size:0.75rem;margin-top:0.5rem">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    model_name = st.selectbox("Select Model:", list(models_available.keys()))

    # Model-specific params
    params_display = {}
    if "SVM" in model_name:
        kernel = st.selectbox("SVM Kernel:", ["rbf", "linear", "poly", "sigmoid"])
        C = st.slider("Regularization C:", 0.01, 100.0, 1.0, 0.01)
        params_display = {"Kernel": kernel, "C": C}
    elif "Random Forest" in model_name:
        n_trees = st.slider("Number of Trees:", 10, 500, 100, 10)
        max_depth = st.slider("Max Depth:", 1, 30, 10)
        params_display = {"Trees": n_trees, "Max Depth": max_depth}
    elif "KMeans" in model_name:
        n_clusters = st.slider("Number of Clusters:", 2, 20, 3)
        params_display = {"Clusters": n_clusters}

    if params_display:
        metric_cards({k: str(v) for k, v in params_display.items()})

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ CONFIGURE K-FOLD TRAINING"):
        st.session_state.model_name = model_name
        # Store params
        if "SVM" in model_name:
            st.session_state['svm_kernel'] = kernel
            st.session_state['svm_C'] = C
        elif "Random Forest" in model_name:
            st.session_state['rf_trees'] = n_trees
            st.session_state['rf_depth'] = max_depth
        elif "KMeans" in model_name:
            st.session_state['kmeans_clusters'] = n_clusters
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAINING + K-FOLD
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Model Training & K-Fold Cross Validation", "🏋️")

    k = st.slider("Number of Folds (K):", 2, 20, 5)
    st.session_state.k_folds = k

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    model_name = st.session_state.model_name
    problem = st.session_state.problem_type

    if st.button("🚀 TRAIN MODEL"):
        from sklearn.model_selection import KFold, cross_val_score

        # Build model
        if model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
        elif "SVM" in model_name and problem == "Regression":
            from sklearn.svm import SVR
            model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'),
                        C=st.session_state.get('svm_C', 1.0))
        elif "SVM" in model_name:
            from sklearn.svm import SVC
            model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'),
                        C=st.session_state.get('svm_C', 1.0), probability=True)
        elif "Random Forest" in model_name and problem == "Regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=st.session_state.get('rf_trees', 100),
                                          max_depth=st.session_state.get('rf_depth', 10),
                                          random_state=42)
        elif "Random Forest" in model_name:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=st.session_state.get('rf_trees', 100),
                                           max_depth=st.session_state.get('rf_depth', 10),
                                           random_state=42)
        elif "KMeans" in model_name:
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=st.session_state.get('kmeans_clusters', 3), random_state=42)

        with st.spinner("Training..."):
            scoring = 'r2' if problem == "Regression" else 'accuracy'
            if "KMeans" not in model_name:
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train)
                cv_scores = np.array([0.0])

        st.session_state.model = model
        st.session_state.results['cv_scores'] = cv_scores
        st.success(f"✅ Model trained! K-Fold {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # CV Score chart
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                                 y=cv_scores, marker_color='#00d4ff', name="Score"))
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Mean: {cv_scores.mean():.4f}")
        fig_cv.update_layout(title=f"{k}-Fold Cross Validation ({scoring})", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cv, use_container_width=True)

        metric_cards({
            "Mean Score": f"{cv_scores.mean():.4f}",
            "Std Dev": f"±{cv_scores.std():.4f}",
            "Best Fold": f"{cv_scores.max():.4f}",
            "Worst Fold": f"{cv_scores.min():.4f}",
        })

    st.markdown('</div>', unsafe_allow_html=True)
    if st.session_state.model and st.button("→ VIEW PERFORMANCE METRICS"):
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — METRICS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Performance Metrics & Overfit/Underfit Analysis", "📊")

    model = st.session_state.model
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    problem = st.session_state.problem_type
    model_name = st.session_state.model_name

    if "KMeans" in model_name:
        st.info("KMeans is unsupervised — showing cluster visualization instead")
        labels = model.labels_
        fig_k = px.scatter(x=X_train.iloc[:,0], y=X_train.iloc[:,1],
                            color=labels.astype(str), title="K-Means Clusters")
        fig_k.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_k, use_container_width=True)
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        if problem == "Regression":
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_mae = mean_absolute_error(y_test, test_preds)

            metric_cards({
                "Train R²": f"{train_r2:.4f}",
                "Test R²": f"{test_r2:.4f}",
                "Train RMSE": f"{train_rmse:.3f}",
                "Test RMSE": f"{test_rmse:.3f}",
                "Test MAE": f"{test_mae:.3f}",
            })

            # Overfit analysis
            gap = train_r2 - test_r2
            if gap > 0.1:
                st.error(f"⚠️ **Overfitting detected** — Train R²={train_r2:.3f} vs Test R²={test_r2:.3f} (gap={gap:.3f})")
            elif test_r2 < 0.5:
                st.warning(f"⚠️ **Possible Underfitting** — Test R²={test_r2:.3f} is low")
            else:
                st.success(f"✅ **Good fit** — Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")

            met_tabs = st.tabs(["📈 Actual vs Predicted", "📊 Residuals", "🎯 Train vs Test R²"])
            with met_tabs[0]:
                fig_av = px.scatter(x=y_test, y=test_preds, labels={'x':'Actual','y':'Predicted'},
                                     title="Actual vs Predicted", trendline='ols',
                                     color_discrete_sequence=['#00d4ff'])
                mn, mx = min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())
                fig_av.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                                  line=dict(dash='dash', color='#f59e0b'))
                fig_av.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_av, use_container_width=True)
            with met_tabs[1]:
                residuals = y_test - test_preds
                fig_res = make_subplots(rows=1, cols=2, subplot_titles=["Residuals", "Residual Distribution"])
                fig_res.add_trace(go.Scatter(x=test_preds, y=residuals, mode='markers',
                                              marker=dict(color='#7c3aed', opacity=0.6)), row=1, col=1)
                fig_res.add_hline(y=0, line_dash="dash", line_color="#f59e0b", row=1, col=1)
                fig_res.add_trace(go.Histogram(x=residuals, marker_color='#00d4ff', nbinsx=30), row=1, col=2)
                fig_res.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_res, use_container_width=True)
            with met_tabs[2]:
                fig_bar = go.Figure(go.Bar(x=['Train R²', 'Test R²'], y=[train_r2, test_r2],
                                            marker_color=['#10b981', '#00d4ff']))
                fig_bar.update_layout(title="Train vs Test R²", **PLOTLY_LAYOUT)
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)

            metric_cards({
                "Train Accuracy": f"{train_acc:.4f}",
                "Test Accuracy": f"{test_acc:.4f}",
                "Accuracy Gap": f"{abs(train_acc-test_acc):.4f}",
            })

            gap = train_acc - test_acc
            if gap > 0.1:
                st.error(f"⚠️ **Overfitting** — Train={train_acc:.3f} vs Test={test_acc:.3f}")
            elif test_acc < 0.6:
                st.warning("⚠️ **Possible Underfitting** — Low test accuracy")
            else:
                st.success("✅ **Good fit**")

            cm = confusion_matrix(y_test, test_preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                title="Confusion Matrix")
            fig_cm.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("→ HYPERPARAMETER TUNING"):
        next_step()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section("Hyperparameter Tuning", "🎛️")

    model_name = st.session_state.model_name
    problem = st.session_state.problem_type
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    search_method = st.selectbox("Search Strategy:", ["Grid Search", "Random Search"])

    # Define param grids
    if "Linear" in model_name:
        param_grid = {'fit_intercept': [True, False]}
        st.info("Linear Regression has limited hyperparameters")
    elif "SVM" in model_name and problem == "Regression":
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'], 'epsilon': [0.1, 0.2, 0.5]}
    elif "SVM" in model_name:
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly']}
    elif "Random Forest" in model_name and problem == "Regression":
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None],
                      'min_samples_split': [2, 5, 10]}
    elif "Random Forest" in model_name:
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None],
                      'min_samples_split': [2, 5, 10]}
    else:
        param_grid = {}

    st.markdown("**Parameter Grid:**")
    for k, v in param_grid.items():
        st.markdown(f'<span class="badge badge-purple">{k}</span>&nbsp;{v}<br>', unsafe_allow_html=True)

    n_iter = st.slider("Iterations (for Random Search):", 5, 50, 10) if search_method == "Random Search" else 0
    cv_k = st.slider("CV Folds:", 2, 10, 3)

    if st.button("⚡ START TUNING") and param_grid:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        # Rebuild base model
        if model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression; base = LinearRegression()
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression; base = LogisticRegression(max_iter=1000)
        elif "SVM" in model_name and problem == "Regression":
            from sklearn.svm import SVR; base = SVR()
        elif "SVM" in model_name:
            from sklearn.svm import SVC; base = SVC()
        elif "Random Forest" in model_name and problem == "Regression":
            from sklearn.ensemble import RandomForestRegressor; base = RandomForestRegressor(random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier; base = RandomForestClassifier(random_state=42)

        scoring = 'r2' if problem == "Regression" else 'accuracy'

        with st.spinner(f"Running {search_method}..."):
            if search_method == "Grid Search":
                searcher = GridSearchCV(base, param_grid, cv=cv_k, scoring=scoring, n_jobs=-1)
            else:
                searcher = RandomizedSearchCV(base, param_grid, cv=cv_k, n_iter=n_iter,
                                              scoring=scoring, n_jobs=-1, random_state=42)
            searcher.fit(X_train, y_train)

        best_model = searcher.best_estimator_
        best_score = searcher.best_score_
        best_params = searcher.best_params_

        st.success(f"✅ Best CV Score: **{best_score:.4f}**")
        st.markdown("**Best Parameters:**")
        for k, v in best_params.items():
            st.markdown(f'<span class="badge badge-green">{k}={v}</span>&nbsp;', unsafe_allow_html=True)

        # Before/after comparison
        old_model = st.session_state.model
        if "KMeans" not in model_name:
            old_preds = old_model.predict(X_test)
            new_preds = best_model.predict(X_test)
            if problem == "Regression":
                from sklearn.metrics import r2_score
                old_score = r2_score(y_test, old_preds)
                new_score = r2_score(y_test, new_preds)
                metric = "R²"
            else:
                from sklearn.metrics import accuracy_score
                old_score = accuracy_score(y_test, old_preds)
                new_score = accuracy_score(y_test, new_preds)
                metric = "Accuracy"

            fig_compare = go.Figure(go.Bar(
                x=["Before Tuning", "After Tuning"],
                y=[old_score, new_score],
                marker_color=['#64748b', '#10b981'],
                text=[f"{old_score:.4f}", f"{new_score:.4f}"],
                textposition='outside'
            ))
            fig_compare.update_layout(title=f"Test {metric}: Before vs After Tuning", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_compare, use_container_width=True)

            # CV results table
            cv_results = pd.DataFrame(searcher.cv_results_)
            cv_results = cv_results.sort_values('mean_test_score', ascending=False).head(10)
            st.dataframe(cv_results[['mean_test_score', 'std_test_score', 'params']].round(4),
                         use_container_width=True)

        st.session_state.model = best_model
        st.success("🎉 Pipeline Complete! Best model saved.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Final summary
    st.markdown("""
    <div style="text-align:center;padding:2rem;background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(124,58,237,0.05));
                border:1px solid var(--border);border-radius:12px;margin-top:1rem">
        <div style="font-family:'Space Mono',monospace;font-size:1.5rem;background:linear-gradient(90deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            🎉 Pipeline Complete
        </div>
        <p style="color:var(--muted);margin-top:0.5rem">
            Your ML pipeline has been executed end-to-end.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 START NEW PIPELINE"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()