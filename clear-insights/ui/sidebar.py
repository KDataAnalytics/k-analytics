import urllib.parse
import urllib.request

import streamlit as st


_FORM_ID = "1FAIpQLSfi_6c3We1ShO7zyyXYnFg-avhQmfiBncxXcMLxdYkOIJ_0rA"
_FORM_ENDPOINT = f"https://docs.google.com/forms/d/e/{_FORM_ID}/formResponse"


def _reset_to_home() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 0
    st.rerun()


def _submit_feedback(payload: dict) -> bool:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(_FORM_ENDPOINT, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:
        return False


def sidebar_progress() -> None:
    with st.sidebar:
        if st.button("Home / Start Over", type="secondary", use_container_width=True):
            _reset_to_home()
        st.caption("Resets progress and clears uploaded data.")
        with st.expander("Send feedback", expanded=False):
            goal = st.text_input("What were you trying to do?")
            worked = st.text_area("What worked well?")
            confused = st.text_area("What didn't work or was confusing?")
            request = st.text_input("One thing you wish this app could do next")
            email = st.text_input("Email (optional)")

            if st.button("Submit feedback", type="primary", use_container_width=True):
                if not any([goal, worked, confused, request, email]):
                    st.warning("Please add at least one response before submitting.")
                else:
                    payload = {
                        "entry.331085623": goal,
                        "entry.1392987103": worked,
                        "entry.134601781": confused,
                        "entry.495051323": request,
                        "entry.1965189409": email,
                    }
                    if _submit_feedback(payload):
                        st.success("Thanks for the feedback!")
                    else:
                        st.error("Could not send feedback. Please try again later.")
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
