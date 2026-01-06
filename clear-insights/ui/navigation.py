import streamlit as st


def home_icon_button() -> None:
    col1, col2 = st.columns([12, 1])
    with col2:
        if st.button("Home", help="Home / Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.step = 0
            st.rerun()


def back_button(step_to_go: int) -> None:
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Back"):
            st.session_state.step = step_to_go
            st.rerun()
