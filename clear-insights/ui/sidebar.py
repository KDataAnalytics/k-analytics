import streamlit as st


def _reset_to_home() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 0
    st.rerun()


def sidebar_progress() -> None:
    with st.sidebar:
        if st.button("Home / Start Over", type="secondary", use_container_width=True):
            _reset_to_home()
        st.caption("Resets progress and clears uploaded data.")
        st.markdown("---")
        st.markdown("## Clear Insights Progress")
        steps = [
            "1. Load Data",
            "2. Profile Data",
            "3. Select Objective",
            "4. Run Analysis",
            "5. Insights",
        ]
        current = st.session_state.step
        for i, step_label in enumerate(steps):
            if i < current:
                st.markdown(f"[x] {step_label}")
            elif i == current:
                st.markdown(f"[>] {step_label} (In Progress)")
            else:
                st.markdown(f"[ ] {step_label}")
        # Progress bar: 0 to 5 steps = full at step 5
        st.progress(min(current / 5, 1.0))
        st.markdown("---")
        st.caption("Progress stays visible as you move through the workflow.")
