# ------------------------------------------------------------
# Clear Insights v2.2 - Full Updated App (Improved Churn Logic)
# ------------------------------------------------------------

import streamlit as st
# import shap No shap explain for this version

from ui.sidebar import sidebar_progress
from ui.state import init_session_state
from ui.upload import step_0_upload
from insights.profile import step_1_profile
from insights.router import step_3_run_analysis
from insights.insights_summary import step_4_insights
from ui.objective_select import step_2_objective

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Clear Insights v2.2", page_icon="CI", layout="wide")


# ------------------------------------------------------------
# INITIAL STATE
# ------------------------------------------------------------
init_session_state()


# ------------------------------------------------------------
# PROGRESS SIDEBAR
# ------------------------------------------------------------
sidebar_progress()

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

