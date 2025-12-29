import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.icons import heading_html
from ui.navigation import back_button


# ------------------------------------------------------------
# STEP 1 - PROFILE DATA
# ------------------------------------------------------------
def step_1_profile() -> None:
    df = st.session_state.df
    st.markdown(heading_html("Step 2: Profile Your Data", "chart", level=1), unsafe_allow_html=True)
    st.markdown("Here is a summary of your uploaded dataset:")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing values", df.isna().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())

    st.markdown(heading_html("Preview", "search", level=3), unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    numeric = df.select_dtypes(include=[np.number])
    st.markdown(heading_html("Correlation (Numeric Columns Only)", "trend_up", level=3), unsafe_allow_html=True)
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
    if st.button("Continue -> Select Objective", type="primary"):
        st.session_state.step = 2
        st.rerun()

    # <- Back button to return to file upload
    back_button(0)
