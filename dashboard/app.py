"""
Streamlit dashboard for network capacity forecasting and anomaly detection.

Run with:
    streamlit run dashboard/app.py

Three views:
1. Network overview — current and historical traffic per PoP
2. Forecasting — run any model on any PoP, compare against ground truth
3. Anomaly detection — flagged events with operator-friendly context
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from network_forecast import (
    ResidualAnomalyDetector,
    SeasonalNaiveForecaster,
    build_forecaster,
    walk_forward_eval,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(
    page_title="Network Capacity Forecasting",
    page_icon="📡",
    layout="wide",
)


@st.cache_data
def load_traffic() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "network_traffic.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "anomalies_ground_truth.csv")
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
    return df


def render_overview(traffic: pd.DataFrame) -> None:
    st.header("Network overview")

    col1, col2, col3, col4 = st.columns(4)
    latest_24h = traffic[traffic["timestamp"] >= traffic["timestamp"].max() - pd.Timedelta("24h")]
    col1.metric("Total PoPs", traffic["pop"].nunique())
    col2.metric("Observations", f"{len(traffic):,}")
    col3.metric("Peak (last 24h, Gbps)", f"{latest_24h['traffic_gbps'].max():.0f}")
    col4.metric("Avg (last 24h, Gbps)", f"{latest_24h['traffic_gbps'].mean():.0f}")

    st.subheader("Traffic per PoP — last 30 days")
    cutoff = traffic["timestamp"].max() - pd.Timedelta(days=30)
    recent = traffic[traffic["timestamp"] >= cutoff]
    fig = px.line(
        recent, x="timestamp", y="traffic_gbps", color="pop",
        labels={"traffic_gbps": "Traffic (Gbps)", "timestamp": "Time"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily peak traffic by PoP")
    daily = (
        traffic.assign(date=traffic["timestamp"].dt.date)
        .groupby(["date", "pop"])["traffic_gbps"].max().reset_index()
    )
    fig2 = px.line(
        daily, x="date", y="traffic_gbps", color="pop",
        labels={"traffic_gbps": "Daily peak (Gbps)", "date": "Date"},
    )
    fig2.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)


def render_forecast(traffic: pd.DataFrame) -> None:
    st.header("Forecasting")

    pops = sorted(traffic["pop"].unique())
    col1, col2, col3 = st.columns(3)
    pop = col1.selectbox("PoP", pops)
    model_name = col2.selectbox("Model", ["seasonal_naive", "lightgbm"])
    horizon = col3.slider("Forecast horizon (hours)", 24, 168, 48, 24)

    series = (
        traffic[traffic["pop"] == pop]
        .set_index("timestamp")["traffic_gbps"]
        .sort_index()
    )

    # Train on everything except the last `horizon` hours so we can show
    # the actual values alongside the forecast for honest visual diagnosis.
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    with st.spinner(f"Fitting {model_name}…"):
        model = build_forecaster(model_name).fit(train)
        forecast = model.predict(horizon)

    forecast_df = forecast.to_dataframe()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index[-168:], y=train.values[-168:],
        mode="lines", name="Training (last 7d)", line=dict(color="#7f8c8d"),
    ))
    fig.add_trace(go.Scatter(
        x=test.index, y=test.values,
        mode="lines", name="Actual (held out)", line=dict(color="#2c3e50", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"], y=forecast_df["forecast"],
        mode="lines", name=f"Forecast ({model_name})",
        line=dict(color="#e74c3c", dash="dash", width=2.5),
    ))
    fig.update_layout(
        height=460, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time", yaxis_title="Traffic (Gbps)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show error metrics
    from network_forecast.evaluation import mae, rmse, smape
    y_true, y_pred = test.values, forecast_df["forecast"].values
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (Gbps)", f"{mae(y_true, y_pred):.2f}")
    c2.metric("RMSE (Gbps)", f"{rmse(y_true, y_pred):.2f}")
    c3.metric("sMAPE (%)", f"{smape(y_true, y_pred):.2f}")


def render_anomalies(traffic: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
    st.header("Anomaly detection")

    pops = sorted(traffic["pop"].unique())
    col1, col2 = st.columns(2)
    pop = col1.selectbox("PoP", pops, key="anomaly_pop")
    threshold = col2.slider("Detection threshold (MAD-σ)", 2.0, 5.0, 3.5, 0.1)

    series = (
        traffic[traffic["pop"] == pop]
        .set_index("timestamp")["traffic_gbps"]
        .sort_index()
    )

    # Use seasonal naive as the "expected" baseline — fast and stakeholder-friendly.
    # Production would use the champion model from the comparison table.
    train = series.iloc[: -24 * 14]  # hold out last 14 days
    test = series.iloc[-24 * 14 :]
    model = SeasonalNaiveForecaster(season_length=168).fit(train)

    # Generate "expected" by repeatedly predicting one season and shifting forward.
    expected_values = []
    history = train.copy()
    for _ in range((len(test) // 168) + 1):
        f = SeasonalNaiveForecaster(season_length=168).fit(history).predict(168)
        expected_values.extend(f.point_forecast)
        next_idx = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h", tz="UTC",
        )
        history = pd.concat([history, pd.Series(f.point_forecast, index=next_idx)])
    expected = pd.Series(expected_values[: len(test)], index=test.index)

    residuals = (test - expected).values
    detector = ResidualAnomalyDetector(threshold=threshold).fit(residuals)
    detection = detector.detect(test.index, test.values, expected.values)
    detection_df = detection.to_dataframe()

    n_anomalies = int(detection_df["is_anomaly"].sum())
    n_outages = int((detection_df["anomaly_type"] == "outage").sum())
    n_surges = int((detection_df["anomaly_type"] == "surge").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total anomalies", n_anomalies)
    c2.metric("Outages", n_outages)
    c3.metric("Surges", n_surges)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=detection_df["timestamp"], y=detection_df["expected"],
        mode="lines", name="Expected", line=dict(color="#7f8c8d", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=detection_df["timestamp"], y=detection_df["observed"],
        mode="lines", name="Observed", line=dict(color="#2c3e50", width=2),
    ))
    flagged = detection_df[detection_df["is_anomaly"]]
    if not flagged.empty:
        outages = flagged[flagged["anomaly_type"] == "outage"]
        surges = flagged[flagged["anomaly_type"] == "surge"]
        if not outages.empty:
            fig.add_trace(go.Scatter(
                x=outages["timestamp"], y=outages["observed"],
                mode="markers", name="Outage",
                marker=dict(color="#e74c3c", size=10, symbol="triangle-down"),
            ))
        if not surges.empty:
            fig.add_trace(go.Scatter(
                x=surges["timestamp"], y=surges["observed"],
                mode="markers", name="Surge",
                marker=dict(color="#f39c12", size=10, symbol="triangle-up"),
            ))
    fig.update_layout(
        height=480, margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time", yaxis_title="Traffic (Gbps)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_anomalies > 0:
        st.subheader("Flagged events")
        st.dataframe(
            flagged[["timestamp", "observed", "expected", "z_score", "anomaly_type"]]
            .round(2)
            .reset_index(drop=True),
            use_container_width=True,
        )


def main() -> None:
    st.title("📡 Network capacity forecasting & anomaly detection")
    st.caption("Synthetic UK broadband telemetry · Built for the Sky Network Data Scientist role · Naveed Abbas")

    if not (DATA_DIR / "network_traffic.parquet").exists():
        st.error(
            "Data not found. Run `python -m network_forecast.data_generator` "
            "from the project root first."
        )
        st.stop()

    traffic = load_traffic()
    ground_truth = load_anomalies()

    tab1, tab2, tab3 = st.tabs(["Overview", "Forecasting", "Anomaly detection"])
    with tab1:
        render_overview(traffic)
    with tab2:
        render_forecast(traffic)
    with tab3:
        render_anomalies(traffic, ground_truth)


if __name__ == "__main__":
    main()
