import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_historical(prophet_df: pd.DataFrame):
    return px.line(prophet_df, x="ds", y="y")


def plot_forecast(prophet_df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    forecast_plot = forecast.copy()
    if "yhat_lower" not in forecast_plot.columns:
        forecast_plot["yhat_lower"] = forecast_plot["yhat"]
    if "yhat_upper" not in forecast_plot.columns:
        forecast_plot["yhat_upper"] = forecast_plot["yhat"]

    if not prophet_df.empty and not forecast_plot.empty:
        last_actual_date = prophet_df["ds"].iloc[-1]
        if forecast_plot["ds"].min() > last_actual_date:
            anchor = pd.DataFrame({
                "ds": [last_actual_date],
                "yhat": [prophet_df["y"].iloc[-1]],
                "yhat_lower": [prophet_df["y"].iloc[-1]],
                "yhat_upper": [prophet_df["y"].iloc[-1]]
            })
            forecast_plot = pd.concat([anchor, forecast_plot], ignore_index=True).sort_values("ds")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast_plot["ds"], y=forecast_plot["yhat"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=forecast_plot["ds"], y=forecast_plot["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_plot["ds"], y=forecast_plot["yhat_lower"],
        mode="lines", fill="tonexty", line=dict(width=0),
        name="Confidence interval"
    ))
    fig.update_layout(hovermode="x unified")
    return fig


def plot_elbow(K, inertias):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, inertias, "bx-")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    return fig
