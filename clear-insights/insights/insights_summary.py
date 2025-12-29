import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.icons import heading_html
from ui.navigation import back_button


# ------------------------------------------------------------
# STEP 4 - INSIGHTS SUMMARY (Supports Churn + Regression + Time-Series + Clustering)
# ------------------------------------------------------------
def step_4_insights():
    st.markdown(heading_html("Step 5: Insights & Recommendations", "bulb", level=1), unsafe_allow_html=True)

    # Determine which analysis was performed
    analysis_type = st.session_state.get("current_analysis", "churn")

    # ============================================================
    # CHURN INSIGHTS (Your original - unchanged)
    # ============================================================
    if analysis_type == "churn":
        if "churn_results" not in st.session_state:
            st.warning("No results found. Please complete the churn analysis first.")
            if st.button("<- Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.churn_results
        st.success("Churn Analysis Complete! Here's your actionable summary.")

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Churn Rate", f"{r['historical_churn_rate']:.1%}")
        col2.metric("Active Customers", f"{len(r['active_customers']):,}")
        col3.metric(
            f"High-Risk Customers (top {r['high_risk_threshold_pct']}%)",
            f"{len(r['high_risk_customers']):,}"
        )

        # Drivers
        st.markdown(heading_html("Top Predictive Drivers", "chart", level=4), unsafe_allow_html=True)
        st.info("""
These are the features your model found most important for predicting churn.
- Higher = stronger influence
- Positive = increases churn risk
- Negative = reduces churn risk
""")

        if r["importances"] is not None:
            df_importance = pd.DataFrame({
                "Feature": r["feature_names"],
                "Importance": r["importances"]
            }).sort_values("Importance", ascending=False).head(15)

            col_name = "Abs Coefficient" if "Logistic" in r["best_model_name"] else "Importance"
            df_importance = df_importance.rename(columns={"Importance": col_name})

            st.dataframe(
                df_importance.style.format({col_name: "{:.4f}"}),
                use_container_width=True
            )
        else:
            st.info("Feature importance not available for this model.")

        # Recommendations
        st.markdown(heading_html("Actionable Recommendations", "bulb", level=4), unsafe_allow_html=True)
        st.write("""
1. Focus immediate outreach on high-risk customers.
2. Reinforce onboarding for new customers (<90 days).
3. Investigate top drivers -> these are your biggest churn levers.
4. Re-run this analysis monthly to measure improvements.
5. Export results to your CRM to activate retention workflows.
""")

        # Export
        st.markdown(heading_html("Export Options", "download", level=3), unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        export_active_only = st.toggle(
            "Export active customers only (recommended for retention)",
            value=True
        )

        with col1:
            export_df = r["active_customers"] if export_active_only else r["df_with_preds"]
            csv_full = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Full Dataset",
                csv_full,
                file_name=(
                    "clear_insights_active_customers_with_predictions.csv"
                    if export_active_only
                    else "clear_insights_all_customers_with_predictions.csv"
                ),
                mime="text/csv"
            )

        with col2:
            csv_risk = (
                r["high_risk_customers"]
                .sort_values("churn_probability", ascending=False)
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                f" Download High-Risk (Top {r['high_risk_threshold_pct']}%)",
                csv_risk,
                file_name="clear_insights_high_risk_customers.csv",
                mime="text/csv"
            )

    # ============================================================
    # REGRESSION INSIGHTS
    # ============================================================
    elif analysis_type == "regression":
        if "regression_results" not in st.session_state:
            st.warning("No regression results found. Please complete the regression analysis first.")
            if st.button("<- Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.regression_results
        target = r["target"]

        st.success(f"Regression Analysis Complete: Predicting {target}")

        st.markdown(heading_html("Model Performance", "chart", level=4), unsafe_allow_html=True)
        st.write(f"**Best Model:** {r['best_model_name']}")

        st.markdown(heading_html("Top Predictive Drivers", "chart", level=4), unsafe_allow_html=True)
        st.info(f"""
These features most strongly influence predicted {target}.
- Higher importance = bigger impact on the prediction
- Use this to understand what drives higher/lower values
""")

        if r["importances"] is not None:
            top_drivers = pd.Series(r["importances"], index=r["feature_names"]).sort_values(ascending=False).head(15)
            df_drivers = pd.DataFrame({
                "Feature": top_drivers.index,
                "Importance": top_drivers.values.round(4)
            })
            st.dataframe(df_drivers, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

        st.markdown(heading_html("Export Predictions", "download", level=3), unsafe_allow_html=True)
        csv_full = r["df_with_preds"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Dataset with Predictions",
            data=csv_full,
            file_name=f"regression_predictions_{target}.csv",
            mime="text/csv",
            use_container_width=True
        )

        pred_col = f"{target}_predicted"
        if pred_col in r["df_with_preds"].columns:
            top_pred = r["df_with_preds"].nlargest(100, pred_col)
            csv_top = top_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Top 100 Highest Predicted Values",
                data=csv_top,
                file_name=f"top_100_predicted_{target}.csv",
                mime="text/csv"
            )

        st.markdown(heading_html("Actionable Recommendations", "bulb", level=4), unsafe_allow_html=True)
        st.write(f"""
1. Use predicted {target} to prioritize customers, properties, or opportunities
2. Focus on improving top predictive drivers to increase outcomes
3. Monitor model performance over time - retrain with fresh data
4. Export predictions to your CRM, sales, or operations tools
""")

    # ============================================================
    # TIME SERIES INSIGHTS (UPDATED - properly indented)
    # ============================================================
    elif analysis_type == "timeseries":
        if "timeseries_results" not in st.session_state:
            st.warning("No forecast results found. Please generate a forecast first.")
            if st.button("<- Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.timeseries_results

        forecast = r.get("forecast")
        if forecast is None:
            forecast = r.get("forecast_daily")

        if forecast is None:
            st.error("Forecast results are missing. Please re-run the forecast.")
            return

        actuals = st.session_state.get("timeseries_actuals")  # optional but recommended

        st.success("Time-Series Forecast Complete! Here's your summary.")
        st.markdown(heading_html("Forecast + Historical", "trend_up", level=4), unsafe_allow_html=True)

        fig = go.Figure()

        # Historical actuals (if available)
        if actuals is not None and {"ds", "y"}.issubset(actuals.columns):
            fig.add_trace(go.Scatter(
                x=actuals["ds"],
                y=actuals["y"],
                mode="lines+markers",
                name="Historical Actuals"
            ))
            last_actual_value = float(actuals["y"].iloc[-1])
        else:
            # Fallback: approximate last actual using forecast
            last_actual_value = float(forecast["yhat"].iloc[-91]) if len(forecast) >= 91 else float(forecast["yhat"].iloc[0])

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast"
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence interval"
        ))

        fig.update_layout(title="Historical + Forecast", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        forecast_end = float(forecast["yhat"].iloc[-1])
        change = forecast_end - last_actual_value

        col1, col2 = st.columns(2)
        col1.metric("Current Value", f"{last_actual_value:,.0f}")
        col2.metric("Forecast at End", f"{forecast_end:,.0f}")

        if abs(change) > 100:
            if change > 0:
                st.success(f" Expected increase of {change:,.0f} over the forecast horizon")
            else:
                st.warning(f" Expected decrease of {abs(change):,.0f} over the forecast horizon")
        else:
            st.info("Stable value expected")

        # Export
        st.markdown(heading_html("Export Options", "download", level=3), unsafe_allow_html=True)
        st.caption("Both exports include historical values plus the forecast (with confidence bounds).")

        col1, col2 = st.columns(2)

        # Build Daily "Historical + Forecast" export (single button)
        date_name = r.get("date_col", "Date")
        metric_name = r.get("value_col", "metric")

        export_daily = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        export_daily = export_daily.rename(columns={
            "ds": date_name,
            "yhat": "forecast",
            "yhat_lower": "forecast_lower",
            "yhat_upper": "forecast_upper"
        })

        # Add historical actuals aligned to the same date column (if available)
        if actuals is not None and {"ds", "y"}.issubset(actuals.columns):
            hist_daily = actuals[["ds", "y"]].copy()
            hist_daily = hist_daily.rename(columns={"ds": date_name, "y": f"historical_{metric_name}"})
            export_daily = export_daily.merge(hist_daily, on=date_name, how="left")
        else:
            # Keep file explicit even if actuals aren't available
            export_daily[f"historical_{metric_name}"] = np.nan

        with col1:
            csv_daily = export_daily.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Daily Detail (History + Forecast)",
                csv_daily,
                file_name=f"daily_detail_{metric_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Keep your existing monthly export logic (already includes history + forecast in your case)
            monthly = r.get("forecast_monthly")
            if monthly is None:
                monthly = (
                    forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
                    .resample("ME").mean()
                    .reset_index()
                    .rename(columns={"ds": date_name})
                )

            csv_monthly = monthly.to_csv(index=False).encode("utf-8")
            st.download_button(
                " Download Monthly Summary (History + Forecast)",
                csv_monthly,
                file_name=f"monthly_summary_{metric_name}.csv",
                mime="text/csv",
                use_container_width=True
            )


        st.markdown(heading_html("Recommendations", "bulb", level=4), unsafe_allow_html=True)
        st.write(
            "- Compare actuals vs forecast as new data arrives\n"
            "- Use this forecast for planning (budget, staffing, inventory)\n"
            "- Re-run regularly to keep projections current"
        )

    # ============================================================
    # CLUSTERING INSIGHTS
    # ============================================================
    elif analysis_type == "clustering":
        if "clustering_results" not in st.session_state:
            st.warning("No clustering results found. Please run clustering first.")
            if st.button("<- Back to Analysis"):
                st.session_state.step = 3
                st.rerun()
            return

        r = st.session_state.clustering_results
        n_clusters = r["n_clusters"]

        st.success(f"Clustering Complete: Discovered {n_clusters} Customer Segments")

        st.markdown(heading_html("Cluster Profiles", "cluster", level=4), unsafe_allow_html=True)
        st.dataframe(r["profile"], use_container_width=True)

        st.markdown(heading_html("Recommendations", "bulb", level=4), unsafe_allow_html=True)
        st.write(f"""
- Name and describe each cluster based on key traits
- Tailor marketing, product, or support strategies to each segment
- Monitor how cluster sizes and behaviors change over time
- Export data to activate segment-specific campaigns
""")

        st.markdown(heading_html("Export Segmented Data", "download", level=3), unsafe_allow_html=True)
        csv = r["df_clustered"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Customers with Cluster Labels",
            data=csv,
            file_name="customer_segments_with_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ============================================================
    # COMMON FOOTER
    # ============================================================
    st.markdown("---")

    if st.button("Start New Analysis", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.step = 0
        st.rerun()

    back_button(3)
