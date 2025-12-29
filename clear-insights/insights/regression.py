import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from ui.icons import heading_html
from ui.navigation import back_button
from utils.metrics import regression_metrics


# ------------------------------------------------------------
# STEP 3C - REGRESSION ANALYSIS (Fixed ID Exclusion)
# ------------------------------------------------------------
def run_regression(target):
    df = st.session_state.df.copy()
    st.markdown(heading_html(f"Regression: Predicting {target}", "chart", level=1), unsafe_allow_html=True)

    # === EXCLUDE UNIQUE IDs EARLY (prevent leakage/overfitting) ===
    id_patterns = ["id", "customer", "user_id", "account", "house_id"]
    id_cols = [c for c in df.columns if any(pat in c.lower() for pat in id_patterns) and c != target]

    if id_cols:
        st.info(f" Auto-excluded {len(id_cols)} unique ID columns (e.g., customer_id, house_id) to prevent overfitting and improve insights.")

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
            **regression_metrics(y_test, pred)
        })

    leaderboard = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    st.dataframe(leaderboard.style.format({
        "R2 Score": "{:.3f}",
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}"
    }))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_name]
    st.success(f" Best model: **{best_name}** (R2 = {leaderboard.iloc[0]['R2 Score']:.3f})")

    with st.expander("Model Details", expanded=False):
        st.markdown(f"**Selected model:** {best_name}")
        if hasattr(best_model, "coef_"):
            coef = pd.Series(best_model.coef_, index=feature_cols)
            coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
            df_coef = pd.DataFrame({
                "Feature": coef.index,
                "Coefficient": coef.values.round(6)
            })
            st.markdown("**Linear coefficients (top 20):**")
            st.dataframe(df_coef.head(20), use_container_width=True)
            intercept = float(best_model.intercept_) if hasattr(best_model, "intercept_") else 0.0
            st.markdown(f"**Intercept:** {intercept:.6f}")
        else:
            st.info("This model does not have a simple equation. Use the predictions export for scoring.")

    # Final predictions on full data
    best_model.fit(X, y)
    df_with_preds = df.copy()
    df_with_preds[f"{target}_predicted"] = best_model.predict(X)

    # Reorder columns
    cols = [f"{target}_predicted", target] + [c for c in df_with_preds.columns if c not in [f"{target}_predicted", target]]
    df_with_preds = df_with_preds[cols]

    # Show sample predictions
    st.markdown(heading_html("Sample Predictions", "chart", level=3), unsafe_allow_html=True)
    st.dataframe(df_with_preds.head(20), use_container_width=True)

    # Top Drivers
    st.markdown(heading_html("Top Predictive Drivers", "chart", level=3), unsafe_allow_html=True)
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
    st.markdown(heading_html("Export Data with Predictions", "download", level=3), unsafe_allow_html=True)
    csv = df_with_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Dataset with Predictions",
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

    if st.button("Continue -> Insights Summary", type="primary", key="regression_continue"):
        st.session_state.current_analysis = "regression"
        st.session_state.step = 5
        st.rerun()

    st.session_state.analysis_complete = True
