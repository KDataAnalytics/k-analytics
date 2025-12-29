import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMClassifier

from ui.icons import heading_html
from ui.navigation import back_button


# ------------------------------------------------------------
# STEP 3B - CHURN ANALYSIS (v2.3 Final - No SHAP, Clean Drivers)
# ------------------------------------------------------------
def run_churn():
    df = st.session_state.df.copy()
    st.markdown(heading_html("Churn Prediction", "search", level=1), unsafe_allow_html=True)

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
        st.info(f" Auto-excluded {len(id_cols)} ID columns and {len(date_cols)} raw date columns to prevent leakage.")

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
        st.success("Added tenure features: tenure_days, tenure_months, is_new_customer")

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
    st.success(f" Best model: **{best_name}** (AUC: {leaderboard.iloc[0]['AUC']:.3f})")

    # Final fit and predictions
    best_model.fit(X, y)
    df_with_preds = df.copy()
    df_with_preds["churn_probability"] = best_model.predict_proba(X)[:, 1]
    cols = ["churn_probability"] + [c for c in df_with_preds.columns if c != "churn_probability"]
    df_with_preds = df_with_preds[cols]

    # Active customers
    active_customers = df_with_preds[df[target].map(lambda x: churn_mapping.get(str(x).lower().strip(), 0)) == 0]

    st.info(
        "Churn scores are primarily used to prioritize retention for active customers. "
        "Already-churned customers may not rank highest because they are historical outcomes."
    )

    # High-risk slider
    st.markdown(heading_html("Highest-Risk Active Customers", "alert", level=3), unsafe_allow_html=True)
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
    st.markdown(heading_html("Top Churn Drivers", "chart", level=3), unsafe_allow_html=True)

    st.info("""
    These are the factors that most strongly influence whether a customer churns.

    - Higher strength = bigger impact on the prediction
    - Examples:
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
    if st.button("Continue -> Insights Summary", type="primary", key="churn_continue"):
        st.session_state.step = 5
        st.rerun()

    st.session_state.analysis_complete = True

    # <- Back button to return to select objective
    back_button(2)
