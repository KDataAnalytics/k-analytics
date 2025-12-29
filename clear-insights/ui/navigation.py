import streamlit as st


def back_button(step_to_go: int) -> None:
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Back"):
            st.session_state.step = step_to_go
            st.rerun()
