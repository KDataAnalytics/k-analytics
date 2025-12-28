# ------------------------------------------------------------
# Clear Insights v2.2 — Full Updated App (Improved Churn Logic)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Clear Insights v2.2", page_icon="🔍", layout="wide")


# ------------------------------------------------------------
# INITIAL STATE
# ------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# ------------------------------------------------------------
# PROGRESS SIDEBAR
# ------------------------------------------------------------
def sidebar_progress():
    with st.sidebar:
        st.markdown("## 🔎 Clear Insights — Progress")
        steps = [
            "1. Load Data",
            "2. Profile Data",
            "3. Select Objective",
            "4. Run Analysis",
            "5. Insights"
        ]
        current = st.session_state.step
        for i, step_label in enumerate(steps):
            if i < current:
                st.markdown(f"✅ {step_label}")
            elif i == current:
                st.markdown(f"🔵 {step_label} (In Progress)")
            else:
                st.markdown(f"⬜ {step_label}")
        # Progress bar: 0 to 5 steps = full at step 5
        st.progress(min(current / 5, 1.0))
        st.markdown("---")
        st.caption("Progress stays visible as you move through the workflow.")


sidebar_progress()

# ------------------------------------------------------------
# BACK BUTTON
# ------------------------------------------------------------
def back_button(step_to_go):
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("← Back"):
            st.session_state.step = step_to_go
            st.rerun()

# ------------------------------------------------------------
# STEP 0 — WELCOME & UPLOAD (Julius-style Landing)
# ------------------------------------------------------------
def step_0_upload():
    # Julius-style CSS: hero, upload card, preview frame, feature cards
    st.markdown("""
    <style>
      /* keep content centered / roomy */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

      /* HERO panel */
      .ci-hero {
        text-align: center;
        padding: 70px 18px 28px;
        border-radius: 22px;
        border: 1px solid rgba(0,0,0,0.06);
        background: linear-gradient(180deg, rgba(30,60,114,0.12) 0%, rgba(255,255,255,1) 62%);
      }
      .ci-title {
        font-size: 4.1rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        color: #1e3c72;
        margin: 0 0 14px 0;
      }
      .ci-subtitle {
        font-size: 1.35rem;
        color: rgba(0,0,0,0.70);
        max-width: 860px;
        margin: 0 auto 22px auto;
        line-height: 1.55;
      }

      /* Upload card */
      .ci-upload-card {
        max-width: 720px;
        margin: 0 auto;
        padding: 18px 18px 14px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
      }
      .ci-upload-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: rgba(0,0,0,0.84);
      }
      .ci-upload-sub {
        font-size: 0.95rem;
        margin: 0 0 10px 0;
        color: rgba(0,0,0,0.62);
      }
      .ci-upload-hint {
        font-size: 0.90rem;
        margin-top: 10px;
        color: rgba(0,0,0,0.55);
      }

      /* Make Streamlit uploader look like Julius */
      section[data-testid="stFileUploader"] { border-radius: 14px; }
      section[data-testid="stFileUploader"] > div {
        border-radius: 14px;
        border: 1px dashed rgba(0,0,0,0.22);
        padding: 14px;
        background: rgba(30,60,114,0.035);
      }

      /* Preview frame */
      .ci-preview {
        margin-top: 36px;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
      }
      .ci-section-title {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 850;
        color: rgba(0,0,0,0.84);
        margin: 0 0 10px 0;
      }
      .ci-section-sub {
        text-align: center;
        color: rgba(0,0,0,0.62);
        margin: 0 0 16px 0;
      }

      /* Feature cards */
      .ci-feature-card {
        padding: 14px 14px 12px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        height: 100%;
      }
      .ci-feature-title { font-weight: 850; margin-bottom: 6px; }
      .ci-feature-desc { color: rgba(0,0,0,0.65); margin: 0; line-height: 1.45; }

      /* small chips */
      .ci-chips { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin: 10px 0 22px; }
      .ci-chip {
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.95rem;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.85);
        color: rgba(0,0,0,0.72);
      }
    </style>
    """, unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <div class="ci-hero">
      <div class="ci-title">Clear Insights</div>
      <div class="ci-subtitle">
        Connect your data. Ask questions in plain English.<br/>
        Get instant insights and predictions — <b>no coding required</b>.
      </div>

      <div class="ci-chips">
        <div class="ci-chip">⚡ Fast setup</div>
        <div class="ci-chip">🔎 Guided workflow</div>
        <div class="ci-chip">📈 Forecasts, churn, and drivers</div>
        <div class="ci-chip">🧠 Plain-English insights</div>
      </div>

      <div class="ci-upload-card">
        <div class="ci-upload-title">Upload your data to get started</div>
        <div class="ci-upload-sub">Drag and drop a CSV file below. Next: we’ll profile your columns and suggest the best objective.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload (centered)
    st.markdown("<div style='max-width:720px;margin: -10px auto 0;'>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["csv"],
        label_visibility="collapsed"
    )
    st.markdown("<div class='ci-upload-hint' style='max-width:720px;margin: 0 auto; text-align:center;'>Limit 200MB per file · CSV</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            st.session_state.step = 1
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Preview section (framed like the mockup)
    st.markdown("""
    <div class="ci-preview">
      <div class="ci-section-title">Beautiful, actionable insights in seconds</div>
      <div class="ci-section-sub">Forecasts, confidence bands, key drivers, and clear recommendations.</div>
    """, unsafe_allow_html=True)

    st.image(
        "https://i.imgur.com/78gJlJR.png",
        use_container_width=True
    )
    st.caption("Example: 90-day forecast with historical data, confidence intervals, and clear recommendations")
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature grid (cards)
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='ci-section-title'>What you can do</div>", unsafe_allow_html=True)
    st.markdown("<div class='ci-section-sub'>Pick an objective and Clear Insights guides you through the workflow.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">📈 Forecast Revenue</div>
          <p class="ci-feature-desc">90-day predictions with confidence bands and trend drivers.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">⚠️ Predict Churn</div>
          <p class="ci-feature-desc">Identify at-risk customers and the factors driving churn risk.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">🌀 Segment Customers</div>
          <p class="ci-feature-desc">Find natural customer groups for targeted strategies.</p>
        </div>
        """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">💰 Model LTV</div>
          <p class="ci-feature-desc">Predict lifetime value from early behaviors and signals.</p>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">🏠 Price Properties</div>
          <p class="ci-feature-desc">Estimate value from multiple features with clear feature impact.</p>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="ci-feature-card">
          <div class="ci-feature-title">📊 Explain Results</div>
          <p class="ci-feature-desc">Every output explained in plain English with next steps.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    
# ------------------------------------------------------------
# STEP 1 — PROFILE DATA
# ------------------------------------------------------------
def step_1_profile():
    df = st.session_state.df
    st.title("📊 Step 2: Profile Your Data")
    st.markdown("Here is a summary of your uploaded dataset:")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing values", df.isna().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())
    
    st.subheader("🔍 Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    numeric = df.select_dtypes(include=[np.number])
    st.subheader("📈 Correlation (Numeric Columns Only)")
    if len(numeric.columns) >= 2:
        fig = px.imshow(
            numeric.corr().round(2),
            text_auto=True,
            color_continuous_scale="RdBu",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")
    
    st.markdown("---")
    if st.button("Continue → Select Objective", type="primary"):
        st.session_state.step = 2
        st.rerun()
    
    # ← Back button to return to file upload
    back_button(0)

# ------------------------------------------------------------
# STEP 2 — SELECT OBJECTIVE (Improved Card-Based UI)
# ------------------------------------------------------------
def step_2_objective():
    df = st.session_state.df
    st.title("🎯 Step 3: Select Your Objective")
    st.markdown("Choose the analysis that best fits your goals. We'll guide you based on your data.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().any()]
    churn_cols = [c for c in df.columns if c.lower() in ["churn", "churned", "left", "is_churned", "canceled"]]

    # Use columns for responsive card grid
    col1, col2 = st.columns(2)

    with col1:
        # Time-Series Card
        with st.container(border=True):
            st.markdown("#### 📈 Time-Series Forecasting")
            if date_cols and len(numeric_cols) >= 1:
                st.success("✅ Your data is ready — date column + numeric values detected")
            else:
                st.info("ℹ️ Needs a date column and at least one numeric metric")
            st.markdown("Forecast trends, seasonality, and future values over time.")
            if st.button("Select Time-Series Analysis", type="primary", use_container_width=True, key="select_timeseries"):
                st.session_state.analysis_type = "timeseries"
                st.session_state.step = 3
                st.rerun()

        # Regression Card
        with st.container(border=True):
            st.markdown("#### 📊 Regression (Predict a Number)")
            if len(numeric_cols) >= 2:
                st.success("✅ Ready — multiple numeric columns available")
            else:
                st.info("ℹ️ Needs at least two numeric columns (one as target)")
            st.markdown("Predict continuous values like revenue, score, or quantity.")
            target_numeric = st.selectbox(
                "Choose target column to predict",
                options=[""] + numeric_cols,
                key="regression_target"
            )
            if target_numeric:
                if st.button("Run Regression Analysis", type="primary", use_container_width=True, key="run_regression"):
                    st.session_state.analysis_type = ("regression", target_numeric)
                    st.session_state.step = 3
                    st.rerun()

    with col2:
        # Churn Card
        with st.container(border=True):
            st.markdown("#### 🔍 Customer Churn Prediction")
            if churn_cols:
                st.success("✅ Ready — churn column detected")
            else:
                st.info("ℹ️ Needs a binary churn column (e.g., 'churn', 'is_churned')")
            st.markdown("Identify at-risk customers and understand churn drivers.")
            if st.button("Select Churn Analysis", type="primary", use_container_width=True, key="select_churn"):
                st.session_state.analysis_type = "churn"
                st.session_state.step = 3
                st.rerun()

        # Clustering Card
        with st.container(border=True):
            st.markdown("#### 🌀 Customer Segmentation (Clustering)")
            if len(numeric_cols) >= 3:
                st.success("✅ Ready — sufficient numeric features")
            else:
                st.info("ℹ️ Works best with 3+ numeric features")
            st.markdown("Discover natural customer groups without labels.")
            if st.button("Run Clustering Analysis", type="primary", use_container_width=True, key="run_clustering"):
                st.session_state.analysis_type = "clustering"
                st.session_state.step = 3
                st.rerun()

    # ← Back button to return to file upload
    back_button(1)

# ------------------------------------------------------------
# STEP 3A — TIME-SERIES ANALYSIS (Final - Daily Chart + Monthly Export)
# ------------------------------------------------------------
def run_timeseries():
    df = st.session_state.df.copy()
    st.title("📈 Time-Series Forecasting")

    st.markdown("Select the date column and the numeric value to forecast.")

    date_cols = [c for c in df.columns if pd.to_datetime(df[c], errors='coerce').notna().any()]
    value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not date_cols or not value_cols:
        st.error("Need at least one date column and one numeric column.")
        back_button(2)
        return

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date column", date_cols)
    with col2:
        value_col = st.selectbox("Value to forecast", value_cols)

    # Prepare data
    df_ts = df[[date_col, value_col]].copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
    df_ts = df_ts.dropna(subset=[date_col, value_col])
    df_ts = df_ts.sort_values(date_col)

    # Daily frequency for smooth Prophet forecast
    df_daily = df_ts.set_index(date_col).asfreq('D').ffill()
    prophet_df = df_daily.reset_index().rename(columns={date_col: "ds", value_col: "y"})

    if len(prophet_df) < 30:
        st.error("Need at least 30 days of data for reliable forecasting.")
        return

    st.subheader(f"Historical {value_col}")
    fig_hist = px.line(prophet_df, x="ds", y="y", title=f"{value_col} Over Time", markers=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    if st.button("Generate 90-Day Forecast", type="primary"):
        with st.spinner("Building forecast..."):
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=90, freq='D')
            forecast = m.predict(future)

            # Daily chart
            st.subheader("90-Day Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prophet_df["ds"], y=prophet_df["y"],
                mode="lines+markers", name="Historical", line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                mode="lines", name="Forecast", line=dict(color="green")
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_upper"],
                mode="lines", line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat_lower"],
                mode="lines", fill="tonexty",
                fillcolor="rgba(0,176,246,0.2)", line=dict(width=0), name="95% Confidence"
            ))
            fig.update_layout(title="Historical + 90-Day Forecast", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Monthly export (practical)
            export_monthly = forecast.set_index("ds").resample('ME')[["yhat", "yhat_lower", "yhat_upper"]].mean()
            export_monthly = export_monthly.reset_index()
            export_monthly = export_monthly.rename(columns={"ds": date_col})
            export_monthly = pd.merge(
                df_ts.reset_index()[[date_col, value_col]],
                export_monthly,
                on=date_col,
                how="right"
            )

            csv = export_monthly.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Monthly Historical + Forecast",
                data=csv,
                file_name=f"monthly_forecast_{value_col}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Store both for insights
            st.session_state.timeseries_results = {
                "forecast_daily": forecast,
                "forecast_monthly": export_monthly,
                "value_col": value_col,
                "date_col": date_col
            }

    # === Bottom navigation ===
    st.markdown("---")
    back_button(2)

    if 'timeseries_results' in st.session_state:
        if st.button("Continue → Insights Summary", type="primary", key="timeseries_continue"):
            st.session_state.current_analysis = "timeseries"
            st.session_state.step = 5
            st.rerun()

    st.session_state.analysis_complete = True
        
   # # === Bottom navigation ===
   # st.markdown("---")
   # back_button(2)

    # Show Continue only after forecast is generated
    #if 'timeseries_results' in st.session_state:
    #    if st.button("Continue → Insights Summary", type="primary", key="timeseries_continue"):
    #        st.session_state.current_analysis = "timeseries"
    #        st.session_state.step = 5
    #        st.rerun()
    #else:
    #    st.info("👆 Generate a forecast above to continue to the insights summary.")
      
# ------------------------------------------------------------
# STEP 3B — CHURN ANALYSIS (v2.3 Final - No SHAP, Clean Drivers)
# ------------------------------------------------------------
def run_churn():
    df = st.session_state.df.copy()
    st.title("🔍 Churn Prediction")

    # Identify target column
    possible_churn_cols = [c for c in df.columns if c.lower() in ["churn", "churned", "left", "is_churned", "canceled"]]
    if not possible_churn_cols:
        st.error("No churn column detected. Expected a column like 'churn', 'is_churned', etc.")
        return
    target = possible_churn_cols[0]
    st.success(f"Using churn column: **{target}**")

    # Safer binary encoding
    churn_mapping = {
        'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
        'churned': 1, 'left': 1, 'canceled': 1,
        1: 1, True: 1
    }
    y = df[target].map(lambda x: churn_mapping.get(str(x).lower().strip(), 0))
    if y.nunique() != 2:
        st.error(f"Target column '{target}' does not appear to be binary. Unique values: {sorted(y.unique())}")
        return

    churn_rate = y.mean()
    st.metric("Historical Churn Rate", f"{churn_rate:.1%}")

    # --- FEATURE CLEANING & ENGINEERING ---
    cols_to_exclude = [target]

    # Exclude ID columns
    id_patterns = ["id", "customer", "user_id", "account"]
    id_cols = [c for c in df.columns if any(pat in c.lower() for pat in id_patterns)]
    cols_to_exclude += id_cols

    # Exclude raw date columns
    date_cols = []
    for c in df.columns:
        if pd.to_datetime(df[c], errors='coerce').notna().any():
            date_cols.append(c)
    cols_to_exclude += date_cols

    if id_cols or date_cols:
        st.info(f"🔧 Auto-excluded {len(id_cols)} ID columns and {len(date_cols)} raw date columns to prevent leakage.")

    # Tenure engineering
    added_tenure = False
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        reference_date = pd.Timestamp("2025-12-26")

        if 'churn_date' in df.columns:
            df['churn_date'] = pd.to_datetime(df['churn_date'], errors='coerce')

        tenure_end = df['churn_date'].fillna(reference_date)
        df['tenure_days'] = (tenure_end - df['signup_date']).dt.days.clip(lower=0)
        df['tenure_months'] = (df['tenure_days'] // 30).astype(int)
        df['is_new_customer'] = (df['tenure_days'] <= 90).astype(int)
        added_tenure = True

    if added_tenure:
        st.success("✅ Added tenure features: tenure_days, tenure_months, is_new_customer")

    # Prepare features
    X_raw = df.drop(columns=cols_to_exclude, errors='ignore')
    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna("missing")
    feature_cols = X.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }

    st.info("Training and evaluating 4 models...")
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, model.predict(X_test)),
            "AUC": roc_auc_score(y_test, prob)
        })

    leaderboard = pd.DataFrame(results).sort_values("AUC", ascending=False)
    st.dataframe(leaderboard.style.format({"Accuracy": "{:.1%}", "AUC": "{:.3f}"}))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_name]
    st.success(f"🏆 Best model: **{best_name}** (AUC: {leaderboard.iloc[0]['AUC']:.3f})")

    # Final fit and predictions
    best_model.fit(X, y)
    df_with_preds = df.copy()
    df_with_preds["churn_probability"] = best_model.predict_proba(X)[:, 1]
    cols = ["churn_probability"] + [c for c in df_with_preds.columns if c != "churn_probability"]
    df_with_preds = df_with_preds[cols]

    # Active customers
    active_customers = df_with_preds[df[target].map(lambda x: churn_mapping.get(str(x).lower().strip(), 0)) == 0]

    # High-risk slider
    st.subheader("🔥 Highest-Risk Active Customers")
    threshold_pct = st.slider(
        "Define 'High-Risk' as top X% of active customers by churn probability",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
        format="%d%%"
    )

    threshold = threshold_pct / 100
    quantile = 1 - threshold
    risk_cutoff = active_customers["churn_probability"].quantile(quantile)
    high_risk_customers = active_customers[active_customers["churn_probability"] >= risk_cutoff]

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Customers", f"{len(active_customers):,}")
    col2.metric(f"High-Risk (top {threshold_pct}%)", f"{len(high_risk_customers):,}")
    col3.metric("Cutoff Probability", f"{risk_cutoff:.1%}")

    top_n = min(200, len(active_customers))
    st.dataframe(
        active_customers.sort_values("churn_probability", ascending=False).head(top_n),
        use_container_width=True
    )

    # === TOP CHURN DRIVERS (Simple & Reliable) ===
    st.subheader("📊 Top Churn Drivers")

    st.info("""
    These are the factors that most strongly influence whether a customer churns.

    • Higher strength = bigger impact on the prediction
    • Examples:
      - High "plan_Pro" means Pro customers have very different churn behavior
      - High "tenure_months" means longer-tenured customers churn less (or more)
      - High "is_new_customer" means new customers are at much higher risk

    Focus retention efforts here for maximum impact.
    """)

    # Get importances/coefs
    if "Logistic" in best_name:
        importances = np.abs(best_model.coef_[0])
        strength_name = "Strength (Abs Coefficient)"
    elif hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        strength_name = "Strength (Importance)"
    else:
        st.info("Feature importance not available for this model.")
        importances = None

    if importances is not None:
        top_drivers = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
        df_drivers = pd.DataFrame({
            "Feature": top_drivers.index,
            strength_name: top_drivers.values.round(4)
        })
        st.dataframe(df_drivers, use_container_width=True)

    # Store results for Insights page
    st.session_state.churn_results = {
        "df_with_preds": df_with_preds,
        "active_customers": active_customers,
        "high_risk_customers": high_risk_customers,
        "high_risk_threshold_pct": threshold_pct,
        "historical_churn_rate": churn_rate,
        "best_model_name": best_name,
        "best_auc": leaderboard.iloc[0]["AUC"],
        "feature_names": feature_cols,
        "importances": importances
    }

    st.markdown("---")
    if st.button("Continue → Insights Summary", type="primary", key="churn_continue"):
        st.session_state.step = 5
        st.rerun()

    st.session_state.analysis_complete = True

    # ← Back button to return to select objective
    back_button(2)

# ------------------------------------------------------------
# STEP 3C — REGRESSION ANALYSIS (Fixed ID Exclusion)
# ------------------------------------------------------------
def run_regression(target):
    df = st.session_state.df.copy()
    st.title(f"📊 Regression: Predicting {target}")

    # === EXCLUDE UNIQUE IDs EARLY (prevent leakage/overfitting) ===
    id_patterns = ["id", "customer", "user_id", "account", "house_id"]
    id_cols = [c for c in df.columns if any(pat in c.lower() for pat in id_patterns) and c != target]

    if id_cols:
        st.info(f"🔧 Auto-excluded {len(id_cols)} unique ID columns (e.g., customer_id, house_id) to prevent overfitting and improve insights.")

    # Drop target + ID columns BEFORE any processing
    X_raw = df.drop(columns=[target] + id_cols, errors='ignore')
    y = df[target]

    # Handle categorical features
    X = pd.get_dummies(X_raw, drop_first=True)

    # Imputation
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna("missing")

    feature_cols = X.columns.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regression models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1)
    }

    st.info("Training and evaluating 4 regression models...")
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "R² Score": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))
        })

    leaderboard = pd.DataFrame(results).sort_values("R² Score", ascending=False)
    st.dataframe(leaderboard.style.format({
        "R² Score": "{:.3f}",
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}"
    }))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_name]
    st.success(f"🏆 Best model: **{best_name}** (R² = {leaderboard.iloc[0]['R² Score']:.3f})")

    # Final predictions on full data
    best_model.fit(X, y)
    df_with_preds = df.copy()
    df_with_preds[f"{target}_predicted"] = best_model.predict(X)

    # Reorder columns
    cols = [f"{target}_predicted", target] + [c for c in df_with_preds.columns if c not in [f"{target}_predicted", target]]
    df_with_preds = df_with_preds[cols]

    # Show sample predictions
    st.subheader("Sample Predictions")
    st.dataframe(df_with_preds.head(20), use_container_width=True)

    # Top Drivers
    st.subheader("📈 Top Predictive Drivers")
    st.info(f"""
    These features most strongly influence predicted {target}.
    - Higher importance = bigger impact on the prediction
    - Use this to understand what drives higher/lower values
    """)

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_)
    else:
        importances = None

    if importances is not None:
        top_drivers = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
        df_drivers = pd.DataFrame({
            "Feature": top_drivers.index,
            "Importance": top_drivers.values.round(4)
        })
        st.dataframe(df_drivers, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

    # Export
    st.markdown("### 📥 Export Data with Predictions")
    csv = df_with_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Dataset with Predictions",
        data=csv,
        file_name=f"regression_predictions_{target}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Store results for insights page
    st.session_state.regression_results = {
        "df_with_preds": df_with_preds,
        "target": target,
        "best_model_name": best_name,
        "feature_names": feature_cols,
        "importances": importances
    }

    st.markdown("---")
    back_button(2)

    if st.button("Continue → Insights Summary", type="primary", key="regression_continue"):
        st.session_state.current_analysis = "regression"
        st.session_state.step = 5
        st.rerun()

    st.session_state.analysis_complete = True
            
# ------------------------------------------------------------
# STEP 3D — CLUSTERING ANALYSIS (Fixed Continue Button)
# ------------------------------------------------------------
def run_clustering():
    df = st.session_state.df.copy()
    st.title("🌀 Customer Segmentation (Clustering)")

    st.markdown("Discover natural groups in your data based on similar characteristics.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Clustering requires at least 2 numeric columns.")
        back_button(2)
        return

    st.info(f"Using {len(numeric_cols)} numeric features: {', '.join(numeric_cols)}")

    selected_features = st.multiselect(
        "Select features for clustering (optional — default: all)",
        options=numeric_cols,
        default=numeric_cols
    )

    if not selected_features:
        selected_features = numeric_cols

    X = df[selected_features].copy()
    X = X.dropna()

    if len(X) < 10:
        st.error("Not enough complete rows for clustering (need ≥10).")
        return

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.cluster import KMeans

    st.subheader("Finding Optimal Number of Clusters")
    with st.spinner("Running elbow analysis..."):
        inertias = []
        K = range(2, min(10, len(X)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        fig_elbow, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K, inertias, 'bx-')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig_elbow)

    optimal_k = st.slider("Select number of clusters", 2, 8, 4)

    if st.button("Run Clustering", type="primary"):
        with st.spinner(f"Clustering into {optimal_k} groups..."):
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            df_clustered = df.loc[X.index].copy()
            df_clustered["Cluster"] = clusters

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Fixed scatter plot
            fig_scatter = px.scatter(
                df_clustered,
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color="Cluster",
                hover_data=selected_features,
                title="Customer Segments (PCA Projection)",
                labels={"x": "PCA Component 1", "y": "PCA Component 2"}
            )
            fig_scatter.update_traces(marker=dict(size=12, opacity=0.8))
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Profiles
            st.subheader("Cluster Profiles")
            profile = df_clustered.groupby("Cluster")[selected_features].mean().round(2)
            profile = profile.T
            profile.columns = [f"Cluster {i}" for i in profile.columns]
            st.dataframe(profile.style.background_gradient(cmap="viridis"), use_container_width=True)

            # Sizes
            st.subheader("Cluster Sizes")
            sizes = df_clustered["Cluster"].value_counts().sort_index()
            fig_sizes = px.bar(x=[f"Cluster {i}" for i in sizes.index], y=sizes.values,
                               labels={"x": "Cluster", "y": "Customers"},
                               title="Customers per Cluster")
            st.plotly_chart(fig_sizes, use_container_width=True)

            # Export
            st.markdown("### 📥 Export Segmented Data")
            csv = df_clustered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Data with Cluster Labels",
                data=csv,
                file_name="customer_segments_with_clusters.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Store results
            st.session_state.clustering_results = {
                "df_clustered": df_clustered,
                "features": selected_features,
                "n_clusters": optimal_k,
                "profile": profile
            }

    # === Continue button — always visible after clustering ===
    st.markdown("---")
    back_button(2)

    if 'clustering_results' in st.session_state:
        if st.button("Continue → Insights Summary", type="primary", key="clustering_continue", use_container_width=True):
            st.session_state.current_analysis = "clustering"
            st.session_state.step = 5
            st.rerun()
    else:
        st.info("👆 Run clustering above to continue.")

    st.session_state.analysis_complete = True
    
# ------------------------------------------------------------
# STEP 3 — ANALYSIS ROUTER
# ------------------------------------------------------------
def step_3_run_analysis():
    atype = st.session_state.analysis_type
    if atype == "timeseries":
        run_timeseries()
    elif atype == "churn":
        run_churn()
    elif isinstance(atype, tuple) and atype[0] == "regression":
        run_regression(atype[1])
    elif atype == "clustering":
        run_clustering()

    # Only show Continue button for non-churn analyses (placeholders)
    #if atype not in ["churn", "timeseries", "regression"] and st.session_state.analysis_complete:
    #    st.markdown("---")
    #    if st.button("Continue → Insights Summary", type="primary", key="other_continue"):
    #        st.session_state.step = 4
    #        st.rerun()

# ------------------------------------------------------------
# STEP 4 — INSIGHTS SUMMARY (Supports Churn + Regression + Time-Series)
# ------------------------------------------------------------
def step_4_insights():
    st.title("💡 Step 5: Insights & Recommendations")

    # Determine which analysis was performed
    analysis_type = st.session_state.get("current_analysis", "churn")

    # ============================================================
    # CHURN INSIGHTS (Your original — unchanged)
    # ============================================================
    if analysis_type == "churn":
        if "churn_results" not in st.session_state:
            st.warning("No results found. Please complete the churn analysis first.")
            if st.button("← Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return
        r = st.session_state.churn_results
        st.success("Churn Analysis Complete! Here's your actionable summary.")
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Churn Rate", f"{r['historical_churn_rate']:.1%}")
        col2.metric("Active Customers", f"{len(r['active_customers']):,}")
        col3.metric(
            f"High-Risk Customers (top {r['high_risk_threshold_pct']}%)",
            f"{len(r['high_risk_customers']):,}"
        )
        # Drivers
        st.markdown("#### 📈 Top Predictive Drivers")
        st.info("""
These are the features your model found most important for predicting churn.
- Higher = stronger influence
- Positive = increases churn risk
- Negative = reduces churn risk
""")
        if r["importances"] is not None:
            df_importance = pd.DataFrame({
                "Feature": r["feature_names"],
                "Importance": r["importances"]
            }).sort_values("Importance", ascending=False).head(15)
            col_name = "Abs Coefficient" if "Logistic" in r["best_model_name"] else "Importance"
            df_importance = df_importance.rename(columns={"Importance": col_name})
            st.dataframe(
                df_importance.style.format({col_name: "{:.4f}"}),
                use_container_width=True
            )
        else:
            st.info("Feature importance not available for this model.")
        # Recommendations
        st.markdown("#### 💡 Actionable Recommendations")
        st.write("""
1. Focus immediate outreach on high-risk customers.
2. Reinforce onboarding for new customers (<90 days).
3. Investigate top drivers → these are your biggest churn levers.
4. Re-run this analysis monthly to measure improvements.
5. Export results to your CRM to activate retention workflows.
""")
        # Export
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)
        with col1:
            csv_full = r["df_with_preds"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Dataset",
                csv_full,
                file_name="clear_insights_all_customers_with_predictions.csv",
                mime="text/csv"
            )
        with col2:
            csv_risk = (
                r["high_risk_customers"]
                .sort_values("churn_probability", ascending=False)
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                f"⬇️ Download High-Risk (Top {r['high_risk_threshold_pct']}%)",
                csv_risk,
                file_name="clear_insights_high_risk_customers.csv",
                mime="text/csv"
            )

    # ============================================================
    # REGRESSION INSIGHTS (Fixed & Added)
    # ============================================================
    elif analysis_type == "regression":
        if "regression_results" not in st.session_state:
            st.warning("No regression results found. Please complete the regression analysis first.")
            if st.button("← Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.regression_results
        target = r["target"]

        st.success(f"Regression Analysis Complete: Predicting {target}")

        st.markdown("#### Model Performance")
        st.write(f"**Best Model:** {r['best_model_name']}")

        # Top Predictive Drivers
        st.markdown("#### 📈 Top Predictive Drivers")
        st.info(f"""
        These features most strongly influence predicted {target}.
        - Higher importance = bigger impact on the prediction
        - Use this to understand what drives higher/lower values
        """)

        if r["importances"] is not None:
            top_drivers = pd.Series(r["importances"], index=r["feature_names"]).sort_values(ascending=False).head(15)
            df_drivers = pd.DataFrame({
                "Feature": top_drivers.index,
                "Importance": top_drivers.values.round(4)
            })
            st.dataframe(df_drivers, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

        # Export
        st.markdown("### 📥 Export Predictions")
        csv_full = r["df_with_preds"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Dataset with Predictions",
            data=csv_full,
            file_name=f"regression_predictions_{target}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Optional: Top predicted values
        pred_col = f"{target}_predicted"
        if pred_col in r["df_with_preds"].columns:
            top_pred = r["df_with_preds"].nlargest(100, pred_col)
            csv_top = top_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Top 100 Highest Predicted Values",
                data=csv_top,
                file_name=f"top_100_predicted_{target}.csv",
                mime="text/csv"
            )

        # Recommendations
        st.markdown("#### 💡 Actionable Recommendations")
        st.write(f"""
1. Use predicted {target} to prioritize customers, properties, or opportunities
2. Focus on improving top predictive drivers to increase outcomes
3. Monitor model performance over time — retrain with fresh data
4. Export predictions to your CRM, sales, or operations tools
        """)

    # ============================================================
    # TIME SERIES INSIGHTS (Your original — unchanged)
    # ============================================================
    elif analysis_type == "timeseries":
        if "timeseries_results" not in st.session_state:
            st.warning("No forecast results found. Please generate a forecast first.")
            if st.button("← Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.timeseries_results

        st.success("Time-Series Forecast Complete! Here's your summary.")

        st.markdown("#### 90-Day Forecast (Daily Detail)")
        fig = go.Figure()

        # Historical actuals — extract from forecast_daily where yhat is close to actual
        # But simpler: use the forecast_df which has yhat for all, but actuals are in original data
        # Instead, we'll use the full forecast_daily and show yhat, with historical as subset
        # Historical points (from original data)
        historical_dates = r["forecast_daily"]["ds"][r["forecast_daily"]["ds"] <= r["forecast_daily"]["ds"].iloc[-91]]  # Approx historical
        fig.add_trace(go.Scatter(
            x=r["forecast_daily"]["ds"],
            y=r["forecast_daily"]["yhat"].where(r["forecast_daily"]["ds"] <= r["forecast_daily"]["ds"].max() - pd.Timedelta(days=90), None),
            mode="lines+markers",
            name="Historical Actuals",
            line=dict(color="blue")
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=r["forecast_daily"]["ds"],
            y=r["forecast_daily"]["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="green")
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=r["forecast_daily"]["ds"],
            y=r["forecast_daily"]["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=r["forecast_daily"]["ds"],
            y=r["forecast_daily"]["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(0,176,246,0.2)",
            line=dict(width=0),
            name="95% Confidence"
        ))

        fig.update_layout(title="Historical + 90-Day Forecast", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        last_actual = r["forecast_daily"]["yhat"].iloc[-91]  # Approx last historical
        forecast_end = r["forecast_daily"]["yhat"].iloc[-1]
        change = forecast_end - last_actual

        col1, col2 = st.columns(2)
        col1.metric("Current Value", f"{last_actual:,.0f}")
        col2.metric("Forecast in 90 Days", f"{forecast_end:,.0f}")

        if abs(change) > 100:
            if change > 0:
                st.success(f"📈 Expected increase of {change:,.0f} over 90 days")
            else:
                st.warning(f"📉 Expected decrease of {abs(change):,.0f} over 90 days")
        else:
            st.info("➡️ Stable value expected")

        # Export
        st.markdown("### 📥 Export Options")
        col1, col2 = st.columns(2)

        with col1:
            csv_daily = r["forecast_daily"][["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": r["date_col"]}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Daily Forecast",
                csv_daily,
                file_name=f"daily_forecast_{r['value_col']}.csv",
                mime="text/csv"
            )

        with col2:
            csv_monthly = r["forecast_monthly"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Monthly Summary",
                csv_monthly,
                file_name=f"monthly_forecast_{r['value_col']}.csv",
                mime="text/csv"
            )

        st.markdown("#### Recommendations")
        st.write("""
        - Monitor actuals vs forecast monthly
        - Use for planning: budget, inventory, staffing
        - Re-run with new data to stay current
        """)
           
    # ============================================================
    # CLUSTERING INSIGHTS 
    # ============================================================
    elif analysis_type == "clustering":
        if "clustering_results" not in st.session_state:
            st.warning("No clustering results found. Please run clustering first.")
            if st.button("← Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.clustering_results
        n_clusters = r["n_clusters"]

        st.success(f"Clustering Complete: Discovered {n_clusters} Customer Segments")

        st.markdown("#### Cluster Profiles")
        st.dataframe(r["profile"].style.background_gradient(cmap="viridis"), use_container_width=True)

        st.markdown("#### Recommendations")
        st.write(f"""
        - Name and describe each cluster based on key traits
        - Tailor marketing, product, or support strategies to each segment
        - Monitor how cluster sizes and behaviors change over time
        - Export data to activate segment-specific campaigns
        """)

        st.markdown("### 📥 Export Segmented Data")
        csv = r["df_clustered"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Customers with Cluster Labels",
            data=csv,
            file_name="customer_segments_with_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ============================================================
    # COMMON FOOTER
    # ============================================================
    st.markdown("---")

    if st.button("🧹 Start New Analysis", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.step = 0
        st.rerun()

    back_button(3)

# ------------------------------------------------------------
# ROUTER
# ------------------------------------------------------------
if st.session_state.step == 0:
    step_0_upload()
elif st.session_state.step == 1:
    step_1_profile()
elif st.session_state.step == 2:
    step_2_objective()
elif st.session_state.step == 3:
    step_3_run_analysis()
elif st.session_state.step == 5:  # Insights & Recommendations page
    # Force scroll to top - works in current Streamlit
    js = '''
    <script>
        window.parent.document.querySelector(".main").scrollTop = 0;
        // Fallback for body
        window.parent.document.body.scrollTop = 0;
        window.parent.document.documentElement.scrollTop = 0;
    </script>
    '''
    st.components.v1.html(js, height=0, width=0)
    
    step_4_insights()

