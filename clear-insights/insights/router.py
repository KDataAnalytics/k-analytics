import streamlit as st

from insights.churn import run_churn
from insights.clustering import run_clustering
from insights.regression import run_regression
from insights.time_series import run_timeseries


# ------------------------------------------------------------
# STEP 3 - ANALYSIS ROUTER
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
    #    if st.button("Continue -> Insights Summary", type="primary", key="other_continue"):
    #        st.session_state.step = 4
    #        st.rerun()
