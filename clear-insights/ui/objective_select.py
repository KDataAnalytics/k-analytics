import pandas as pd
import streamlit as st

from ui.icons import heading_html
from ui.navigation import back_button


# ------------------------------------------------------------
# STEP 2 - SELECT OBJECTIVE (Improved Card-Based UI)
# ------------------------------------------------------------
def step_2_objective() -> None:
    df = st.session_state.df
    st.markdown(heading_html("Step 3: Select Your Objective", "target", level=1), unsafe_allow_html=True)
    st.markdown("Choose the analysis that best fits your goals. We'll guide you based on your data.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    date_cols = [c for c in df.columns if pd.to_datetime(df[c], errors="coerce").notna().any()]
    churn_cols = [c for c in df.columns if c.lower() in ["churn", "churned", "left", "is_churned", "canceled"]]

    # Use columns for responsive card grid
    col1, col2 = st.columns(2)

    with col1:
        # Time-Series Card
        with st.container(border=True):
            st.markdown(heading_html("Time-Series Forecasting", "trend_up", level=4), unsafe_allow_html=True)
            if date_cols and len(numeric_cols) >= 1:
                st.success("Your data is ready - date column + numeric values detected")
            else:
                st.info("Needs a date column and at least one numeric metric")
            st.markdown("Forecast trends, seasonality, and future values over time.")
            if st.button("Select Time-Series Analysis", type="primary", use_container_width=True, key="select_timeseries"):
                st.session_state.analysis_type = "timeseries"
                st.session_state.step = 3
                st.rerun()

        # Regression Card
        with st.container(border=True):
            st.markdown(heading_html("Regression (Predict a Number)", "chart", level=4), unsafe_allow_html=True)
            if len(numeric_cols) >= 2:
                st.success("Ready - multiple numeric columns available")
            else:
                st.info("Needs at least two numeric columns (one as target)")
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
            st.markdown(heading_html("Customer Churn Prediction", "search", level=4), unsafe_allow_html=True)
            if churn_cols:
                st.success("Ready - churn column detected")
            else:
                st.info("Needs a binary churn column (e.g., 'churn', 'is_churned')")
            st.markdown("Identify at-risk customers and understand churn drivers.")
            if st.button("Select Churn Analysis", type="primary", use_container_width=True, key="select_churn"):
                st.session_state.analysis_type = "churn"
                st.session_state.step = 3
                st.rerun()

        # Clustering Card
        with st.container(border=True):
            st.markdown(heading_html("Customer Segmentation (Clustering)", "cluster", level=4), unsafe_allow_html=True)
            if len(numeric_cols) >= 3:
                st.success("Ready - sufficient numeric features")
            else:
                st.info("Works best with 3+ numeric features")
            st.markdown("Discover natural customer groups without labels.")
            if st.button("Run Clustering Analysis", type="primary", use_container_width=True, key="run_clustering"):
                st.session_state.analysis_type = "clustering"
                st.session_state.step = 3
                st.rerun()

    # <- Back button to return to file upload
    back_button(1)
