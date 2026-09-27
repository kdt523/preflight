"""Preflight dashboard.   streamlit run apps/dashboard/app.py

Every number is computed from the database at view time; nothing is typed in.
Set PREFLIGHT_INSTALLATION_ID to show one tenant's data under row-level security.
"""

import math
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from apps.dashboard import data
from preflight_core.db.engine import get_engine
from preflight_core.db.tenancy import tenant_connection

# validated palette: status colours for risk bands (always labelled), single-hue ramp for rates
BAND_COLORS = {"low": "#0ca30c", "watch": "#fab219", "high": "#d03b3b", "skipped": "#8a8987"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
INK_2 = "#52514e"

st.set_page_config(page_title="Preflight", page_icon="🛫", layout="wide")


@contextmanager
def conn():
    inst = os.environ.get("PREFLIGHT_INSTALLATION_ID")
    if inst:
        with tenant_connection(get_engine(), int(inst)) as c:
            yield c
    else:
        with get_engine().connect() as c:
            yield c


@st.cache_data(ttl="5m")
def load(window_days: int, repo_id: int | None):
    since = datetime.now(UTC) - timedelta(days=window_days)
    with conn() as c:
        return data.checks(c, since, repo_id), data.outcomes(c, since, repo_id)


@st.cache_data(ttl="15m")
def load_repos():
    with conn() as c:
        return data.repos(c)


@st.cache_data(ttl="15m")
def load_hotspots(repo_id: int, days: int):
    with conn() as c:
        return data.hotspots(c, repo_id, days)


def fmt(v, pattern="{:.0%}", empty="—"):
    return empty if v is None or (isinstance(v, float) and math.isnan(v)) else pattern.format(v)


# -- sidebar ---------------------------------------------------------------------------------
repo_df = load_repos()
with st.sidebar:
    st.title("🛫 Preflight")
    names = ["All repos", *repo_df["full_name"].tolist()] if len(repo_df) else ["All repos"]
    choice = st.selectbox("Repository", names)
    repo_id = (
        None
        if choice == "All repos"
        else int(repo_df.loc[repo_df["full_name"] == choice, "id"].iloc[0])
    )
    window = (
        st.segmented_control("Window", [7, 30, 90], default=30, format_func=lambda d: f"{d} days")
        or 30
    )
    st.caption("Preflight never blocks a merge. Checks are advisory.")

chk, out = load(window, repo_id)
k = data.kpis(chk, out)

overview, hot, calib, drift = st.tabs(["Overview", "Hotspots", "Calibration", "Drift"])

# -- overview --------------------------------------------------------------------------------
with overview:
    if not len(chk):
        st.info("No checks in this window yet. Install the GitHub App on a repo and open a PR.")
    with st.container(horizontal=True):
        st.metric(
            "Checks", f"{k['checks']:,}", border=True, chart_data=data.weekly(chk), chart_type="bar"
        )
        st.metric(
            "Live PR-AUC",
            fmt(k.get("live_pr_auc"), "{:.3f}"),
            border=True,
            help=f"On {k['labelled']} predictions whose CI outcome is known",
        )
        st.metric(
            "High-band precision",
            fmt(k.get("high_band_precision")),
            border=True,
            help="Of builds flagged high, the share that really failed (code defect)",
        )
        st.metric(
            "Found useful",
            fmt(k["useful_share"]),
            border=True,
            help=f"{k['votes']} votes on Check Runs",
        )
        st.metric("Fail-open rate", fmt(k["fail_open_rate"], "{:.1%}"), border=True)
        st.metric(
            "p95 latency",
            fmt(k["p95_latency_ms"], "{:.0f} ms"),
            border=True,
            help="SLO: under 300 ms",
        )

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Checks by risk band")
        if len(chk):
            counts = chk["band"].value_counts()
            order = [b for b in ("low", "watch", "high", "skipped") if b in counts]
            fig = go.Figure(
                go.Bar(
                    x=[counts[b] for b in order],
                    y=order,
                    orientation="h",
                    marker_color=[BAND_COLORS[b] for b in order],
                    text=[f"{b}: {counts[b]:,}" for b in order],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:,} checks<extra></extra>",
                )
            )
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=40, t=10, b=0),
                showlegend=False,
                xaxis_title=None,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
    with right, st.container(border=True):
        st.subheader("CI minutes the adaptive plan could save")
        share = st.slider(
            "Share of a full run skipped by lint + affected tests",
            0.0,
            1.0,
            0.6,
            0.05,
            help="An assumption, not a measurement. Adjust it.",
        )
        m = data.ci_minutes(out, share)
        st.metric(
            "Estimated minutes saved",
            f"{m['estimated_saved']:,.0f}",
            help=f"of {m['total_minutes']:,.0f} CI minutes on scored builds",
        )
        st.caption(
            f"Cost side: {m['low_risk_failures']} low-risk builds still failed and would "
            "have run a lighter plan first. Pushes to the default branch always run the "
            "full matrix."
        )

# -- hotspots --------------------------------------------------------------------------------
with hot, st.container(border=True):
    st.subheader("Hotspots: churn × failure rate (90 days)")
    if repo_id is None:
        st.info("Pick a repository in the sidebar.")
    else:
        h = load_hotspots(repo_id, 90).head(300)
        if not len(h):
            st.info("No builds with diffs in the last 90 days.")
        else:
            labels = [choice, *h["dir"].unique(), *h["path"]]
            parents = ["", *[choice] * h["dir"].nunique(), *h["dir"]]
            dir_churn = h.groupby("dir")["churn"].sum()
            values = [0, *[dir_churn[d] for d in h["dir"].unique()], *h["churn"]]
            dir_rate = h.groupby("dir").apply(
                lambda g: (g["fail_rate"] * g["churn"]).sum() / max(1, g["churn"].sum()),
                include_groups=False,
            )
            colors = [0, *[dir_rate[d] for d in h["dir"].unique()], *h["fail_rate"]]
            fig = go.Figure(
                go.Treemap(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues="total",
                    marker=dict(
                        colors=colors,
                        colorscale=list(
                            zip(
                                [i / (len(BLUE_RAMP) - 1) for i in range(len(BLUE_RAMP))],
                                BLUE_RAMP,
                                strict=True,
                            )
                        ),
                        colorbar=dict(title="fail rate", tickformat=".0%"),
                        line=dict(width=2),
                    ),
                    hovertemplate="<b>%{label}</b><br>churn %{value:,} lines<br>"
                    "smoothed fail rate %{color:.0%}<extra></extra>",
                )
            )
            fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                h[["path", "churn", "builds", "fails", "fail_rate"]],
                hide_index=True,
                column_config={
                    "fail_rate": st.column_config.NumberColumn(
                        "smoothed fail rate", format="percent"
                    )
                },
            )

# -- calibration -----------------------------------------------------------------------------
with calib, st.container(border=True):
    st.subheader("Does 70% mean 7 in 10?")
    cal = data.calibration(out)
    if not len(cal):
        st.info("Needs served predictions whose CI outcome is known.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="perfect",
                line=dict(color=INK_2, dash="dash", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cal["predicted"],
                y=cal["observed"],
                mode="lines+markers",
                name="Preflight",
                line=dict(color="#2a78d6", width=2),
                marker=dict(
                    size=[max(8, min(24, n**0.5)) for n in cal["n"]],
                    line=dict(color="white", width=2),
                ),
                customdata=cal["n"],
                hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>%{customdata} builds"
                "<extra></extra>",
            )
        )
        fig.update_layout(
            height=420,
            xaxis=dict(title="Predicted P(fail)", tickformat=".0%", range=[0, 1]),
            yaxis=dict(title="Observed failure rate", tickformat=".0%", range=[0, 1]),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"ECE {fmt(k.get('live_ece'), '{:.3f}')} on {k['labelled']} builds (target ≤ 0.05)."
        )
        st.dataframe(cal, hide_index=True)

# -- drift -----------------------------------------------------------------------------------
with drift, st.container(border=True):
    st.subheader("Feature drift (served vs training)")
    rep = data.monitor_report()
    if rep is None:
        st.info("No monitor run yet (scripts/monitor.py runs hourly).")
    else:
        status = (
            "🔴 shift"
            if rep["psi_max"] > 0.25
            else ("🟡 moderate" if rep["psi_max"] > 0.1 else "🟢 stable")
        )
        with st.container(horizontal=True):
            st.metric("Max PSI", f"{rep['psi_max']:.2f}", border=True, help="> 0.25 = shift")
            st.metric("Status", status, border=True)
            st.metric("Served (7 d)", f"{rep['served']:,}", border=True)
        if rep["retrain_reasons"]:
            st.warning("Retrain triggered: " + "; ".join(rep["retrain_reasons"]))
        st.dataframe(rep["psi_top"], hide_index=True)
