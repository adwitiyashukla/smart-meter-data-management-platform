from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from meterdm import db
from meterdm.config import load_config

CONFIG_PATH = os.environ.get("METERDM_CONFIG", "config.toml")
TEMPLATE = "plotly_dark"
ACCENT = "#4cc9f0"
WARN = "#f4a261"
BAD = "#e76f51"
GOOD = "#2a9d8f"
STATUS_COLOURS = {"actual": ACCENT, "estimated": WARN, "edited": BAD, "missing": "#6b7280"}

st.set_page_config(page_title="Smart Meter Data Management", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def configuration():
    return load_config(CONFIG_PATH)


@st.cache_data(show_spinner=False)
def query(sql: str, parameters: tuple = ()) -> pd.DataFrame:
    config = configuration()
    handle = db.read_only(config)
    try:
        return handle.execute(sql, list(parameters)).df()
    finally:
        handle.close()


def styled(figure, height=340):
    figure.update_layout(
        template=TEMPLATE,
        height=height,
        margin={"l": 10, "r": 10, "t": 52, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "right", "x": 1},
        title={"x": 0.0, "xanchor": "left"},
    )
    return figure


def overview(fleet: str):
    totals = query(
        """
        SELECT count(*) AS meters, sum(readings) AS readings, min(first_local_date) AS first_day,
               max(last_local_date) AS last_day, sum(total_kwh) AS total_kwh
        FROM dim_meter WHERE fleet = ?
        """,
        (fleet,),
    ).iloc[0]
    quality = query("SELECT stage, check_name, subject, value, threshold, passed FROM fact_quality")
    columns = st.columns(5)
    columns[0].metric("Meters", f"{int(totals['meters']):,}")
    columns[1].metric("Readings", f"{int(totals['readings']):,}")
    columns[2].metric("Energy", f"{totals['total_kwh'] / 1e6:,.2f} GWh")
    columns[3].metric("First day", str(totals["first_day"])[:10])
    columns[4].metric("Last day", str(totals["last_day"])[:10])

    daily = query(
        """
        SELECT local_date, sum(kwh) AS kwh, count(DISTINCT meter_id) AS meters
        FROM fact_daily_read WHERE fleet = ? GROUP BY 1 ORDER BY 1
        """,
        (fleet,),
    )
    daily["kwh_per_meter"] = daily["kwh"] / daily["meters"].clip(lower=1)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=daily["local_date"], y=daily["kwh_per_meter"], name="kWh per meter", line={"color": ACCENT})
    )
    figure.update_layout(title="Average daily consumption per meter", yaxis_title="kWh")
    st.plotly_chart(styled(figure), width="stretch")

    weather = query(
        "SELECT local_date, tavg_c FROM fact_weather_day WHERE fleet = ? ORDER BY 1",
        (fleet,),
    )
    merged = daily.merge(weather, on="local_date", how="inner")
    scatter = px.scatter(
        merged,
        x="tavg_c",
        y="kwh_per_meter",
        color_discrete_sequence=[ACCENT],
        opacity=0.55,
        labels={"tavg_c": "Daily mean temperature (C)", "kwh_per_meter": "kWh per meter"},
        title="Fleet response to temperature",
    )
    st.plotly_chart(styled(scatter), width="stretch")

    failures = quality[~quality["passed"]]
    if failures.empty:
        st.success(f"All {len(quality)} warehouse quality checks passed")
    else:
        st.error(f"{len(failures)} of {len(quality)} quality checks failed")
    st.dataframe(quality, width="stretch", hide_index=True)


def vee_audit(fleet: str):
    summary = query(
        """
        SELECT rule_id, method, count(*) AS readings
        FROM fact_vee_edit WHERE fleet = ? AND rule_id IS NOT NULL GROUP BY 1, 2 ORDER BY 3 DESC
        """,
        (fleet,),
    )
    statuses = query(
        "SELECT status, count(*) AS readings FROM fact_interval_sample WHERE fleet = ? GROUP BY 1 ORDER BY 2 DESC",
        (fleet,),
    )
    left, right = st.columns([3, 2])
    with left:
        figure = px.bar(
            summary,
            x="readings",
            y="rule_id",
            color="method",
            orientation="h",
            title="Readings changed by validation rule",
            color_discrete_sequence=[ACCENT, WARN, BAD],
        )
        st.plotly_chart(styled(figure, 380), width="stretch")
    with right:
        pie = px.pie(
            statuses,
            names="status",
            values="readings",
            title="Reading status",
            color="status",
            color_discrete_map=STATUS_COLOURS,
            hole=0.5,
        )
        pie.update_traces(textposition="inside", texttemplate="%{percent:.2%}", insidetextorientation="horizontal")
        st.plotly_chart(styled(pie, 380), width="stretch")

    flags = query(
        """
        SELECT rule_id, count(*) AS days, count(DISTINCT meter_id) AS meters
        FROM fact_day_flag WHERE fleet = ? GROUP BY 1 ORDER BY 2 DESC
        """,
        (fleet,),
    )
    st.subheader("Day level review flags")
    st.dataframe(flags, width="stretch", hide_index=True)

    st.subheader("Most recent edits")
    edits = query(
        """
        SELECT meter_id, ts_utc, rule_id, method, original_kwh, new_kwh
        FROM fact_vee_edit WHERE fleet = ? ORDER BY ts_utc DESC LIMIT 200
        """,
        (fleet,),
    )
    st.dataframe(edits, width="stretch", hide_index=True)


def meter_view(fleet: str):
    meters = query(
        "SELECT meter_id FROM fact_interval_sample WHERE fleet = ? GROUP BY 1 ORDER BY 1",
        (fleet,),
    )["meter_id"].tolist()
    if not meters:
        st.info("No sampled meters in this fleet")
        return
    meter = st.selectbox("Meter", meters)
    detail = query(
        """
        SELECT m.tariff_group, m.acorn_group, m.transformer_id, m.readings, m.total_kwh,
               c.model, c.baseline_kwh_day, c.heat_slope, c.cool_slope, c.heat_cp, c.cool_cp, c.r2, c.cvrmse,
               c.baseload_kwh_year, c.heating_kwh_year, c.cooling_kwh_year, k.cluster
        FROM dim_meter m
        LEFT JOIN fact_changepoint c USING (fleet, meter_id)
        LEFT JOIN fact_cluster k USING (fleet, meter_id)
        WHERE m.fleet = ? AND m.meter_id = ?
        """,
        (fleet, meter),
    )
    if detail.empty:
        st.warning("No warehouse record for this meter")
        return
    row = detail.iloc[0]
    columns = st.columns(5)
    columns[0].metric("Tariff", str(row["tariff_group"]))
    columns[1].metric("Segment", str(row["acorn_group"]))
    columns[2].metric("Model", str(row["model"]))
    columns[3].metric("Fit R2", "n/a" if pd.isna(row["r2"]) else f"{row['r2']:.3f}")
    columns[4].metric("Cluster", "n/a" if pd.isna(row["cluster"]) else str(int(row["cluster"])))

    readings = query(
        """
        SELECT ts_local, kwh, status FROM fact_interval_sample
        WHERE fleet = ? AND meter_id = ? ORDER BY ts_local DESC LIMIT 3360
        """,
        (fleet, meter),
    ).sort_values("ts_local")
    figure = px.scatter(
        readings,
        x="ts_local",
        y="kwh",
        color="status",
        color_discrete_map=STATUS_COLOURS,
        title="Last 70 days of interval reads",
        labels={"ts_local": "Local time", "kwh": "kWh per interval"},
    )
    figure.update_traces(marker={"size": 3})
    st.plotly_chart(styled(figure), width="stretch")

    daily = query(
        """
        SELECT d.local_date, d.kwh, w.tavg_c
        FROM fact_daily_read d LEFT JOIN fact_weather_day w USING (fleet, local_date)
        WHERE d.fleet = ? AND d.meter_id = ? AND d.kwh IS NOT NULL ORDER BY 1
        """,
        (fleet, meter),
    )
    if not daily.empty and not pd.isna(row["baseline_kwh_day"]):
        temperature = daily["tavg_c"].to_numpy()
        fitted = float(row["baseline_kwh_day"]) + 0.0 * temperature
        if not pd.isna(row["heat_cp"]):
            fitted = fitted + float(row["heat_slope"]) * (float(row["heat_cp"]) - temperature).clip(min=0)
        if not pd.isna(row["cool_cp"]):
            fitted = fitted + float(row["cool_slope"]) * (temperature - float(row["cool_cp"])).clip(min=0)
        daily["fitted"] = fitted
        scatter = go.Figure()
        scatter.add_trace(
            go.Scatter(
                x=daily["tavg_c"], y=daily["kwh"], mode="markers", name="Observed",
                marker={"color": ACCENT, "size": 5, "opacity": 0.55},
            )
        )
        ordered = daily.sort_values("tavg_c")
        scatter.add_trace(
            go.Scatter(x=ordered["tavg_c"], y=ordered["fitted"], mode="lines", name="Changepoint fit",
                       line={"color": WARN, "width": 3})
        )
        scatter.update_layout(
            title=f"Daily energy against temperature, {row['model']} model",
            xaxis_title="Daily mean temperature (C)",
            yaxis_title="kWh per day",
        )
        st.plotly_chart(styled(scatter), width="stretch")
        parts = pd.DataFrame(
            {
                "component": ["Baseload", "Heating", "Cooling"],
                "kwh_year": [row["baseload_kwh_year"], row["heating_kwh_year"], row["cooling_kwh_year"]],
            }
        )
        bar = px.bar(
            parts, x="component", y="kwh_year", title="Annual energy split from the fitted model",
            color="component", color_discrete_sequence=[ACCENT, BAD, GOOD],
        )
        st.plotly_chart(styled(bar, 300), width="stretch")


def load_shapes(fleet: str):
    profiles = query(
        """
        SELECT cluster, label, meters, is_weekend, period, share
        FROM dim_cluster_profile WHERE fleet = ? ORDER BY cluster, is_weekend, period
        """,
        (fleet,),
    )
    if profiles.empty:
        st.info("Clustering has not been run for this fleet")
        return
    weekend = st.radio("Day type", ["Weekday", "Weekend"], horizontal=True) == "Weekend"
    subset = profiles[profiles["is_weekend"] == weekend].copy()
    subset["name"] = subset["cluster"].astype(str) + " " + subset["label"] + " (" + subset["meters"].astype(str) + ")"
    figure = px.line(
        subset, x="period", y="share", color="name",
        title="Normalised load shape by cluster",
        labels={"period": "Interval of the local day", "share": "Share of daily energy"},
    )
    st.plotly_chart(styled(figure, 420), width="stretch")
    sizes = profiles.groupby(["cluster", "label"], as_index=False)["meters"].max()
    st.dataframe(sizes, width="stretch", hide_index=True)


def revenue_protection(fleet: str):
    scores = query(
        """
        SELECT meter_id, transformer_id, score, label, kind, level_ratio, night_ratio, residual_ratio,
               cusum_negative, transformer_loss, holdout
        FROM fact_theft_score WHERE fleet = ? ORDER BY score DESC
        """,
        (fleet,),
    )
    if scores.empty:
        st.info("Revenue protection has not been run for this fleet")
        return
    holdout = scores[scores["holdout"]]
    columns = st.columns(4)
    columns[0].metric("Meters scored", f"{len(scores):,}")
    columns[1].metric("Tampered", f"{int(scores['label'].sum()):,}")
    columns[2].metric("Holdout meters", f"{len(holdout):,}")
    columns[3].metric("Holdout tampered", f"{int(holdout['label'].sum()):,}")

    figure = px.histogram(
        scores, x="score", color=scores["label"].map({0: "clean", 1: "tampered"}), nbins=60, barmode="overlay",
        opacity=0.7, color_discrete_map={"clean": ACCENT, "tampered": BAD},
        title="Score distribution by true state", labels={"score": "Model score", "color": "State"},
    )
    st.plotly_chart(styled(figure), width="stretch")

    ranked = holdout.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["precision"] = ranked["label"].cumsum() / ranked["rank"]
    curve = px.line(
        ranked.head(400), x="rank", y="precision", title="Precision as the inspection queue deepens",
        labels={"rank": "Meters inspected", "precision": "Share that were tampered"},
        color_discrete_sequence=[WARN],
    )
    st.plotly_chart(styled(curve, 300), width="stretch")
    st.subheader("Inspection queue")
    st.dataframe(ranked.head(50).drop(columns=["holdout"]), width="stretch", hide_index=True)


def tariff_trial():
    estimate = query("SELECT metric, value FROM fact_dtou_estimate")
    if estimate.empty:
        st.info("No tariff trial in this dataset")
        return
    values = dict(zip(estimate["metric"], estimate["value"], strict=True))
    columns = st.columns(4)
    columns[0].metric("Effect on peak window", f"{values.get('beta', 0.0):.4f} kWh")
    columns[1].metric("Relative effect", f"{values.get('relative_effect', 0.0) * 100:.2f} percent")
    columns[2].metric("Standard error", f"{values.get('standard_error', 0.0):.4f}")
    columns[3].metric("t statistic", f"{values.get('t_statistic', 0.0):.2f}")
    st.dataframe(estimate, width="stretch", hide_index=True)

    shape = query(
        """
        SELECT tariff_group, event_day, avg(window_kwh) AS window_kwh, count(*) AS observations
        FROM fact_trial_window WHERE window_kwh IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2
        """
    )
    shape["day type"] = shape["event_day"].map({True: "High price day", False: "Normal day"})
    figure = px.bar(
        shape, x="tariff_group", y="window_kwh", color="day type", barmode="group",
        title="Mean consumption in the peak window",
        labels={"tariff_group": "Tariff group", "window_kwh": "kWh in the peak window"},
        color_discrete_sequence=[ACCENT, BAD],
    )
    st.plotly_chart(styled(figure), width="stretch")


def forecasting(fleet: str):
    metrics = query("SELECT metric, value FROM fact_forecast_metric WHERE fleet = ?", (fleet,))
    if metrics.empty:
        st.info("Forecasting has not been run for this fleet")
        return
    values = dict(zip(metrics["metric"], metrics["value"], strict=True))
    columns = st.columns(4)
    columns[0].metric("MAPE", f"{values.get('mape', 0.0) * 100:.2f} percent")
    columns[1].metric("Seasonal naive MAPE", f"{values.get('naive_mape', 0.0) * 100:.2f} percent")
    columns[2].metric("Skill against naive", f"{values.get('skill_vs_naive', 0.0) * 100:.1f} percent")
    columns[3].metric("Interval coverage", f"{values.get('interval_coverage', 0.0) * 100:.1f} percent")

    origins = query(
        "SELECT DISTINCT origin, origin_ts FROM fact_forecast_backtest WHERE fleet = ? ORDER BY origin",
        (fleet,),
    )
    if origins.empty:
        return
    chosen = st.selectbox("Backtest origin", origins["origin_ts"].tolist())
    origin = int(origins[origins["origin_ts"] == chosen]["origin"].iloc[0])
    frame = query(
        """
        SELECT horizon, actual, predicted, naive, q10, q90
        FROM fact_forecast_backtest WHERE fleet = ? AND origin = ? ORDER BY horizon
        """,
        (fleet, origin),
    )
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame["horizon"], y=frame["q90"], line={"width": 0}, showlegend=False))
    figure.add_trace(
        go.Scatter(
            x=frame["horizon"], y=frame["q10"], fill="tonexty", fillcolor="rgba(76,201,240,0.18)",
            line={"width": 0}, name="10 to 90 band",
        )
    )
    figure.add_trace(
        go.Scatter(x=frame["horizon"], y=frame["actual"], name="Actual", line={"color": ACCENT, "width": 3})
    )
    figure.add_trace(go.Scatter(x=frame["horizon"], y=frame["predicted"], name="Forecast", line={"color": WARN}))
    figure.add_trace(
        go.Scatter(x=frame["horizon"], y=frame["naive"], name="Seasonal naive", line={"color": BAD, "dash": "dot"})
    )
    figure.update_layout(title="Day ahead forecast", xaxis_title="Hours ahead", yaxis_title="kWh per meter")
    st.plotly_chart(styled(figure, 400), width="stretch")
    st.dataframe(metrics, width="stretch", hide_index=True)


def main():
    config = configuration()
    st.sidebar.title("Smart Meter Data Management")
    st.sidebar.caption("Meter data management and revenue protection over smart meter interval reads")
    try:
        available = query("SELECT DISTINCT fleet FROM dim_meter ORDER BY 1")["fleet"].tolist()
    except FileNotFoundError:
        st.error("The gold warehouse has not been built yet. Run the pipeline first.")
        return
    fleet = st.sidebar.selectbox("Fleet", available or list(config.enabled_fleets))
    tabs = st.tabs(
        ["Overview", "Validation", "Meter", "Load shapes", "Revenue protection", "Tariff trial", "Forecast"]
    )
    with tabs[0]:
        overview(fleet)
    with tabs[1]:
        vee_audit(fleet)
    with tabs[2]:
        meter_view(fleet)
    with tabs[3]:
        load_shapes(fleet)
    with tabs[4]:
        revenue_protection(fleet)
    with tabs[5]:
        tariff_trial()
    with tabs[6]:
        forecasting(fleet)


main()
