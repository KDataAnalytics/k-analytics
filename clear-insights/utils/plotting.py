import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_historical(prophet_df: pd.DataFrame):
    return px.line(prophet_df, x="ds", y="y")


def plot_forecast(prophet_df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
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
