import streamlit as st


def init_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "df" not in st.session_state:
        st.session_state.df = None
    if "analysis_type" not in st.session_state:
        st.session_state.analysis_type = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
