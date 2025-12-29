import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# STEP 0 - WELCOME & UPLOAD (Julius-style Landing)
# ------------------------------------------------------------
def step_0_upload() -> None:
    # Julius-style CSS: hero, upload card, preview frame, feature cards
    st.markdown(
        """
    <style>
      /* keep content centered / roomy */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

      /* HERO panel */
      .ci-hero {
        text-align: center;
        padding: 70px 18px 28px;
        border-radius: 22px;
        border: 1px solid rgba(0,0,0,0.06);
        background: linear-gradient(180deg, rgba(30,60,114,0.12) 0%, rgba(255,255,255,1) 62%);
      }
      .ci-title {
        font-size: 4.1rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        color: #1e3c72;
        margin: 0 0 14px 0;
      }
      .ci-subtitle {
        font-size: 1.35rem;
        color: rgba(0,0,0,0.70);
        max-width: 860px;
        margin: 0 auto 22px auto;
        line-height: 1.55;
      }

      /* Upload card */
      .ci-upload-card {
        max-width: 720px;
        margin: 0 auto;
        padding: 18px 18px 14px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
      }
      .ci-upload-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: rgba(0,0,0,0.84);
      }
      .ci-upload-sub {
        font-size: 0.95rem;
        margin: 0 0 10px 0;
        color: rgba(0,0,0,0.62);
      }
      .ci-upload-hint {
        font-size: 0.90rem;
        margin-top: 10px;
        color: rgba(0,0,0,0.55);
      }

      /* Make Streamlit uploader look like Julius */
      section[data-testid="stFileUploader"] { border-radius: 14px; }
      section[data-testid="stFileUploader"] > div {
        border-radius: 14px;
        border: 1px dashed rgba(0,0,0,0.22);
        padding: 14px;
        background: rgba(30,60,114,0.035);
      }

      /* Preview frame */
      .ci-preview {
        margin-top: 36px;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 10px 28px rgba(0,0,0,0.06);
      }
      .ci-section-title {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 850;
        color: rgba(0,0,0,0.84);
        margin: 0 0 10px 0;
      }
      .ci-section-sub {
        text-align: center;
        color: rgba(0,0,0,0.62);
        margin: 0 0 16px 0;
      }

      /* Feature cards */
      .ci-feature-card {
        padding: 14px 14px 12px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #fff;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        height: 100%;
      }
      .ci-feature-title { font-weight: 850; margin-bottom: 6px; }
      .ci-feature-desc { color: rgba(0,0,0,0.65); margin: 0; line-height: 1.45; }

      /* small chips */
      .ci-chips { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin: 10px 0 22px; }
      .ci-chip {
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 0.95rem;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.85);
        color: rgba(0,0,0,0.72);
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }
      .ci-chip-icon {
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }
      .ci-chip-icon svg { display: block; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # HERO
    st.markdown(
        """
    <div class="ci-hero">
      <div class="ci-title">Clear Insights</div>
      <div class="ci-subtitle">
        Connect your data. Ask questions in plain English.<br/>
        Get instant insights and predictions - <b>no coding required</b>.
      </div>

      <div class="ci-chips">
        <div class="ci-chip">
          <span class="ci-chip-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M13 2L3 14h7l-1 8 12-14h-7l-1-6z" fill="#f3b23a"/>
            </svg>
          </span>
          Fast setup
        </div>
        <div class="ci-chip">
          <span class="ci-chip-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="10" cy="10" r="6" fill="#6aa6ff"/>
              <rect x="15" y="15" width="7" height="3" rx="1.5" transform="rotate(45 15 15)" fill="#6aa6ff"/>
            </svg>
          </span>
          Guided workflow
        </div>
        <div class="ci-chip">
          <span class="ci-chip-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <rect x="3" y="12" width="4" height="9" rx="1" fill="#5fbf93"/>
              <rect x="10" y="8" width="4" height="13" rx="1" fill="#5fbf93"/>
              <rect x="17" y="4" width="4" height="17" rx="1" fill="#5fbf93"/>
            </svg>
          </span>
          Forecasts, churn, and drivers
        </div>
        <div class="ci-chip">
          <span class="ci-chip-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3c-4.4 0-8 2.9-8 6.5S7.6 16 12 16s8-2.9 8-6.5S16.4 3 12 3z" fill="#c08adf"/>
              <path d="M8 15c-1.7 1.3-3 3.4-3 5.5 0 .8.7 1.5 1.5 1.5h11c.8 0 1.5-.7 1.5-1.5 0-2.1-1.3-4.2-3-5.5" fill="#c08adf"/>
            </svg>
          </span>
          Plain-English insights
        </div>
      </div>

      <div class="ci-upload-card">
        <div class="ci-upload-title">Upload your data to get started</div>
        <div class="ci-upload-sub">Drag and drop a CSV file below. Next: we'll profile your columns and suggest the best objective.</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Upload (centered)
    st.markdown("<div style='max-width:720px;margin: -10px auto 0;'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.markdown(
        "<div class='ci-upload-hint' style='max-width:720px;margin: 0 auto; text-align:center;'>"
        "Limit 200MB per file - CSV</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"Loaded {len(df):,} rows - {len(df.columns)} columns")
            st.session_state.step = 1
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Preview section (framed like the mockup)
    st.markdown(
        """
    <div class="ci-preview">
      <div class="ci-section-title">Beautiful, actionable insights in seconds</div>
      <div class="ci-section-sub">Forecasts, confidence bands, key drivers, and clear recommendations.</div>
    """,
        unsafe_allow_html=True,
    )

    st.image("https://i.imgur.com/78gJlJR.png", use_container_width=True)
    st.caption("Example: 90-day forecast with historical data, confidence intervals, and clear recommendations")
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature grid (cards)
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='ci-section-title'>What you can do</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ci-section-sub'>Pick an objective and Clear Insights guides you through the workflow.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Forecast Revenue</div>
          <p class="ci-feature-desc">90-day predictions with confidence bands and trend drivers.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Predict Churn</div>
          <p class="ci-feature-desc">Identify at-risk customers and the factors driving churn risk.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Segment Customers</div>
          <p class="ci-feature-desc">Find natural customer groups for targeted strategies.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Model LTV</div>
          <p class="ci-feature-desc">Predict lifetime value from early behaviors and signals.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Price Properties</div>
          <p class="ci-feature-desc">Estimate value from multiple features with clear feature impact.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            """
        <div class="ci-feature-card">
          <div class="ci-feature-title">Explain Results</div>
          <p class="ci-feature-desc">Every output explained in plain English with next steps.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
