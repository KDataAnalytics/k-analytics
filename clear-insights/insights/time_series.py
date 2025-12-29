import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

from ui.icons import heading_html
from ui.navigation import back_button
from utils.dates import (
    infer_frequency,
    looks_like_id_column,
    parse_date_series,
    score_date_column,
    score_value_column,
)
from utils.plotting import plot_forecast, plot_historical


# ------------------------------------------------------------
# STEP 3A - TIME-SERIES ANALYSIS (Smart detection + confirmation)
#   - Auto-detects: date column, frequency, metric
#   - Excludes ID-like columns (Row ID, *_id, index, etc.)
#   - Aggregates transactions to the chosen frequency before Prophet
#   - Stores results in session_state in a way that supports BOTH:
#       * your updated Insights block (uses r["forecast"] + timeseries_actuals)
#       * your older Insights code (expects r["forecast_daily"] / r["forecast_monthly"])
# ------------------------------------------------------------
def run_timeseries():
    df = st.session_state.df.copy()
    st.markdown(heading_html("Time-Series Forecasting", "trend_up", level=1), unsafe_allow_html=True)

    st.markdown(
        "We've analyzed your data and made a few smart choices below. "
        "Please review and confirm before generating the forecast."
    )

    # -------------------------
    # Detect candidates
    # -------------------------
    date_candidates = []
    date_scores = {}

    for c in df.columns:
        try:
            if looks_like_id_column(df[c], c):
                continue
            sc = score_date_column(df[c])
            if sc >= 0.30:  # stricter threshold to reduce false positives
                date_candidates.append(c)
                date_scores[c] = sc
        except Exception:
            pass

    value_candidates = df.select_dtypes(include=[np.number]).columns.tolist()

    if not date_candidates:
        st.error(
            "We couldn't confidently identify a date column. "
            "Please make sure your file has a column with real calendar dates "
            "(e.g., Order Date, Transaction Date) in a consistent format."
        )
        back_button(2)
        return

    if not value_candidates:
        st.error("We couldn't find a numeric column to forecast (e.g., Sales, Profit).")
        back_button(2)
        return

    # -------------------------
    # Auto-select best options
    # -------------------------
    best_date = max(date_candidates, key=lambda c: date_scores.get(c, 0.0))
    best_value = max(value_candidates, key=lambda c: score_value_column(df[c]))

    parsed_best_dates = parse_date_series(df[best_date]).dropna()
    inferred_freq = infer_frequency(parsed_best_dates) if not parsed_best_dates.empty else "Daily"

    # -------------------------
    # Confirmation UI
    # -------------------------
    st.markdown(heading_html("Confirm forecasting setup", "target", level=3), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        date_col = st.selectbox(
            "Date column (calendar time)",
            options=date_candidates,
            index=date_candidates.index(best_date),
            help="We selected the column with the highest percentage of valid dates. "
                 "This should represent real calendar time (not an ID or row number)."
        )

    with col2:
        freq = st.selectbox(
            "Data frequency",
            options=["Daily", "Weekly", "Monthly"],
            index=["Daily", "Weekly", "Monthly"].index(inferred_freq),
            help="We inferred this from the typical spacing between dates. You can override it."
        )

    with col3:
        value_col = st.selectbox(
            "Metric to forecast",
            options=value_candidates,
            index=value_candidates.index(best_value),
            help="We recommend metrics with good coverage and meaningful variation."
        )

    with st.expander("Why we chose these defaults"):
        st.markdown(
            f"""
            **Date column:** `{best_date}`  
            Chosen because it had the highest share of values that could be interpreted as dates
            ({date_scores.get(best_date, 0.0):.0%} parseable).

            **Frequency:** `{inferred_freq}`  
            Inferred from the typical spacing between dates.

            **Metric:** `{best_value}`  
            Recommended based on completeness and variation (a good signal for forecasting).
            """
        )

    # -------------------------
    # Prepare data
    # -------------------------
    df_ts = df[[date_col, value_col]].copy()
    df_ts[date_col] = parse_date_series(df_ts[date_col])
    df_ts = df_ts.dropna(subset=[date_col, value_col])

    if df_ts.empty:
        st.error(
            "No valid rows remain after parsing the date and metric columns. "
            "Please choose a different date/metric column."
        )
        return

    # Aggregate transactions to selected frequency
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

    df_agg = (
        df_ts.groupby(pd.Grouper(key=date_col, freq=freq_map[freq]))[value_col]
            .sum()
            .reset_index()
            .sort_values(date_col)
    )

    if len(df_agg) < 30:
        st.error(
            f"Not enough data points after aggregation ({len(df_agg)} found). "
            "Time-series forecasting works best with at least ~30 periods."
        )
        st.info(
            "Try choosing a different frequency (e.g., Weekly or Monthly), "
            "or select a different date column."
        )
        return

    prophet_df = df_agg.rename(columns={date_col: "ds", value_col: "y"})

    # Store the aggregated historical actuals for the Insights page (recommended)
    st.session_state.timeseries_actuals = prophet_df.copy()

    # -------------------------
    # Historical chart
    # -------------------------
    st.markdown(heading_html(f"Historical {value_col} ({freq})", "trend_up", level=3), unsafe_allow_html=True)
    fig_hist = plot_historical(prophet_df)
    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------
    # Forecast
    # -------------------------
    if st.button("Generate 90-Day Forecast", type="primary"):
        with st.spinner("Building forecast..."):
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=(freq != "Monthly"),
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=90, freq=freq_map[freq])
            forecast = m.predict(future)

            # Build monthly summary for downloads/insights (even if freq is weekly/monthly)
            try:
                export_monthly = (
                    forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
                            .resample("ME").mean()
                            .reset_index()
                            .rename(columns={"ds": date_col})
                )

                hist_monthly = (
                    prophet_df.set_index("ds")[["y"]]
                             .resample("ME").sum()
                             .reset_index()
                             .rename(columns={"ds": date_col, "y": f"historical_{value_col}"})
                )

                export_monthly = export_monthly.merge(hist_monthly, on=date_col, how="left")
            except Exception:
                export_monthly = None

            # Store results for insights (compatible with both new and old Insights code)
            st.session_state.timeseries_results = {
                # New-style keys
                "forecast": forecast,
                "value_col": value_col,
                "date_col": date_col,
                "frequency": freq,

                # Backwards-compatible keys (older insights expects these)
                "forecast_daily": forecast,
                "forecast_monthly": export_monthly if export_monthly is not None else forecast
            }

    if "timeseries_results" in st.session_state:
        forecast = st.session_state.timeseries_results["forecast"]
        st.markdown(heading_html("Forecast", "trend_up", level=3), unsafe_allow_html=True)
        fig = plot_forecast(prophet_df, forecast)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Continue -> Insights Summary", type="primary", key="timeseries_continue"):
            st.session_state.current_analysis = "timeseries"
            st.session_state.step = 5
            st.rerun()

    # Bottom navigation
    st.markdown("---")
    back_button(2)

    st.session_state.analysis_complete = True
