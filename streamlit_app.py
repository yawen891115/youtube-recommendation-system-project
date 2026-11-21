"""English interactive client for the recommendation API."""

import os
from io import StringIO

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REPLAY_TIME = os.getenv("REPLAY_TIME", "2022-05-08T16:00:00Z")
ACCENT, NEUTRAL = "#2a78d6", "#c3c2b7"
st.set_page_config(page_title="YouTube Shorts Recommendation", page_icon="▶", layout="wide")
st.title("YouTube Shorts Recommendation")
st.caption(
    "Real Kuaishou plays (KuaiRand-Pure) · point-in-time watch-history features · "
    "LightGBM LambdaRank ranker · monitored inference"
)


def call_api(path, payload=None):
    try:
        with httpx.Client(timeout=60) as client:
            response = (
                client.get(API_URL + path)
                if payload is None
                else client.post(API_URL + path, json=payload)
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API returned {exc.response.status_code}: {exc.response.text[:500]}")
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")
    return None


def bar_chart(frame, x, y, title, highlight=None, error=None, percent=False):
    """Horizontal bars in one color; an optional highlighted row; values at the bar tips."""
    colors = [ACCENT if highlight is None or row == highlight else NEUTRAL for row in frame[y]]
    figure = go.Figure(
        go.Bar(
            x=frame[x],
            y=frame[y],
            orientation="h",
            marker_color=colors,
            width=0.5,
            error_x=error,
            text=[f"{value:.1%}" if percent else f"{value:.3f}" for value in frame[x]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title, height=max(260, 34 * len(frame) + 90), margin=dict(l=10, r=40, t=50, b=30)
    )
    if percent:
        figure.update_xaxes(tickformat=".0%")
    return figure


def recommendation_table(result):
    frame = pd.DataFrame(result["recommendations"])
    if frame.empty:
        return frame
    return frame[["video_id", "category", "creator_id", "score", "duration_sec"]]


with st.sidebar:
    st.header("Service")
    health = call_api("/health")
    if health:
        st.success("Model ready" if health["model_loaded"] else "Model unavailable")
        st.caption(f"Cache: {health['cache_backend']}")
    st.caption(
        "Like probabilities and confidence labels are uncalibrated. Ranking scores are "
        "relative within one list. This demo does not measure online engagement lift."
    )

single, batch, recommendation, insights, monitoring = st.tabs(
    ["Single prediction", "Batch prediction", "Recommendations", "Model insights", "Monitoring"]
)
with single:
    st.subheader("Predict engagement after observed watch time")
    with st.form("prediction"):
        left, right = st.columns(2)
        user = left.text_input("User ID", "user_16")
        video = right.text_input("Video ID", "video_1")
        watch = left.number_input("Observed watch time (seconds)", 0.0, 3600.0, 20.0)
        hour = right.number_input("Hour of day (UTC)", 0, 23, 12)
        submitted = st.form_submit_button("Predict like probability", type="primary")
    if submitted:
        result = call_api(
            "/predict",
            {"user_id": user, "video_id": video, "watch_time": watch, "hour_of_day": hour},
        )
        if result:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["probability"] * 100,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT}},
                )
            )
            st.plotly_chart(gauge, width="stretch")
            columns = st.columns(3)
            columns[0].metric("Confidence heuristic", result["confidence"])
            columns[1].metric("Response time", f"{result['response_time_ms']:.1f} ms")
            columns[2].metric("Cache hit", str(result["cached"]))
            if result["cold_start"]:
                st.info(
                    "At least one ID has no historical interactions; metadata and fallback features were used."
                )
            st.json(result, expanded=False)

with batch:
    st.subheader("Score a CSV file")
    st.caption(
        "Required columns: user_id, video_id, watch_time. Optional: hour_of_day, timestamp (ISO 8601 with timezone). Up to 5,000 rows; requests are sent in groups of 256."
    )
    st.download_button(
        "Download example CSV",
        "user_id,video_id,watch_time,hour_of_day\nuser_16,video_1,20,12\nuser_18,video_2,35,18\n",
        "batch-example.csv",
        "text/csv",
    )
    upload = st.file_uploader("Interaction CSV", type=["csv"])
    if upload and st.button("Run batch prediction", type="primary"):
        try:
            data = pd.read_csv(StringIO(upload.getvalue().decode("utf-8-sig")))
            required = {"user_id", "video_id", "watch_time"}
            if not required.issubset(data.columns) or not 1 <= len(data) <= 5000:
                raise ValueError("Provide 1–5,000 rows and all three required columns.")
            if not set(data.columns).issubset(required | {"hour_of_day", "timestamp"}):
                raise ValueError("The CSV contains unsupported columns.")
            if data.isna().any().any():
                raise ValueError("Missing cells are not accepted.")
            results = []
            bar = st.progress(0)
            for start in range(0, len(data), 256):
                response = call_api(
                    "/predict/batch",
                    {"instances": data.iloc[start : start + 256].to_dict("records")},
                )
                if response is None:
                    raise ValueError(
                        "Batch stopped because the API returned an error; no partial output is presented as complete."
                    )
                results.extend(response["predictions"])
                bar.progress(min((start + 256) / len(data), 1.0))
            output = pd.concat(
                [data.reset_index(drop=True), pd.DataFrame(results).add_prefix("prediction_")],
                axis=1,
            )
            st.session_state["batch_result"] = output
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(str(exc))
    if "batch_result" in st.session_state:
        output = st.session_state["batch_result"]
        st.dataframe(output, width="stretch")
        st.plotly_chart(
            px.histogram(
                output,
                x="prediction_probability",
                nbins=25,
                title="Like probability distribution",
                color_discrete_sequence=[ACCENT],
            ),
            width="stretch",
        )
        st.download_button(
            "Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv"
        )

with recommendation:
    st.subheader("Rank unseen videos before playback")
    st.caption(
        "Retrieval takes up to 300 eligible unseen videos (the user's most-played categories "
        "first, then popularity); the LightGBM LambdaRank model orders them using pre-watch "
        "features only. The data ends in May 2022, so requests replay a time inside it."
    )
    with st.form("recommendations"):
        left, right = st.columns(2)
        rec_user = left.text_input(
            "User ID (try a new ID such as user_999999 for cold start)", "user_16"
        )
        when = right.text_input("Request time (UTC, ISO 8601)", REPLAY_TIME)
        top_k = st.slider("Number of videos", 1, 50, 10)
        rec_submit = st.form_submit_button("Recommend videos", type="primary")
    if rec_submit:
        result = call_api("/recommend", {"user_id": rec_user, "top_k": top_k, "timestamp": when})
        if result:
            columns = st.columns(4)
            columns[0].metric("Cold start", str(result["cold_start"]))
            columns[1].metric("Earlier plays", result["history_events"])
            columns[2].metric("Response time", f"{result['response_time_ms']:.1f} ms")
            columns[3].metric("Cache hit", str(result["cached"]))
            if result["top_categories"]:
                st.caption("Most-played categories: " + ", ".join(result["top_categories"]))
            else:
                st.info("No watch history: the ranker falls back to video-side signals.")
            st.dataframe(recommendation_table(result), width="stretch", hide_index=True)

with insights:
    st.subheader("Offline ranking benchmark and feature importance")
    info = call_api("/model/info")
    if info and info.get("ranking_benchmark"):
        ranker = info.get("ranking_model") or {}
        protocol = ranker.get("protocol") or {}
        benchmark = pd.DataFrame(info["ranking_benchmark"]).sort_values("test_ndcg@10")
        error = None
        if {"test_ndcg@10_low", "test_ndcg@10_high"}.issubset(benchmark.columns):
            error = dict(
                type="data",
                symmetric=False,
                array=benchmark["test_ndcg@10_high"] - benchmark["test_ndcg@10"],
                arrayminus=benchmark["test_ndcg@10"] - benchmark["test_ndcg@10_low"],
                color="#52514e",
                thickness=1,
            )
        st.plotly_chart(
            bar_chart(
                benchmark,
                "test_ndcg@10",
                "model",
                "Test NDCG@10 by model (95% bootstrap CI)",
                highlight=ranker.get("name"),
                error=error,
            ),
            width="stretch",
        )
        significance = ranker.get("significance_vs_best_baseline") or {}
        st.caption(
            f"Protocol: {protocol.get('query', 'one first-time play')}, "
            f"{protocol.get('candidates_per_query', 100)} candidates per query. Best baseline "
            f"(chosen on validation): {ranker.get('best_baseline', 'n/a')}."
            + (
                f" Paired bootstrap: NDCG@10 difference {significance['mean_difference']:+.4f} "
                f"[{significance['low']:+.4f}, {significance['high']:+.4f}], "
                f"p = {significance['p_value']:.4g}."
                if significance
                else ""
            )
        )
        st.dataframe(
            benchmark.sort_values("test_ndcg@10", ascending=False),
            width="stretch",
            hide_index=True,
        )
        left, right = st.columns(2)
        importance = pd.DataFrame(info.get("ranking_feature_importance") or [])
        if not importance.empty:
            importance["share"] = importance["gain"] / max(importance["gain"].sum(), 1e-12)
            left.plotly_chart(
                bar_chart(
                    importance.head(20).iloc[::-1],
                    "share",
                    "feature",
                    "Ranker: top 20 features by gain share",
                    percent=True,
                ),
                width="stretch",
            )
        groups = pd.DataFrame(info.get("ranking_feature_group_importance") or [])
        if not groups.empty:
            right.plotly_chart(
                bar_chart(
                    groups.iloc[::-1],
                    "share",
                    "group",
                    "Ranker: gain share by feature family",
                    percent=True,
                ),
                width="stretch",
            )
        if info.get("ranking_ablations"):
            st.markdown("**Ablations** (same tuned parameters, one change each)")
            rows = [
                {
                    "variant": name,
                    "features_removed": row.get("removed_features"),
                    "test_ndcg@10": row.get("test_ndcg@10"),
                    "final_minus_variant": (row.get("difference_vs_final") or {}).get(
                        "mean_difference"
                    ),
                    "p_value": (row.get("difference_vs_final") or {}).get("p_value"),
                }
                for name, row in info["ranking_ablations"].items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Gain counts every split, including splits on noise features; ablations are the "
            "causal check."
        )
    elif info:
        st.info("This model bundle has no ranking benchmark. Run python -m src.train.")

with monitoring:
    st.subheader("Live service dashboard")
    if st.button("Refresh monitoring"):
        st.rerun()
    metrics = call_api("/monitoring")
    info = call_api("/model/info")
    if metrics:
        columns = st.columns(5)
        columns[0].metric("Recent requests", metrics["window_requests"])
        columns[1].metric("Mean latency", f"{metrics['mean_latency_ms']:.1f} ms")
        columns[2].metric("p95 latency", f"{metrics.get('p95_latency_ms', 0):.1f} ms")
        columns[3].metric("Error rate", f"{metrics['error_rate']:.1%}")
        columns[4].metric("Cache hit ratio", f"{metrics.get('cache_hit_ratio', 0):.0%}")
        events = pd.DataFrame(metrics["events"])
        if not events.empty:
            events["timestamp"] = pd.to_datetime(events.timestamp, utc=True)
            trend = (
                events.set_index("timestamp")
                .resample("1min")
                .size()
                .rename("requests")
                .reset_index()
            )
            st.plotly_chart(
                px.line(
                    trend,
                    x="timestamp",
                    y="requests",
                    title="Requests per minute",
                    color_discrete_sequence=[ACCENT],
                ),
                width="stretch",
            )
            st.plotly_chart(
                px.histogram(
                    events,
                    x="latency_ms",
                    nbins=25,
                    title="Response latency distribution",
                    color_discrete_sequence=[ACCENT],
                ),
                width="stretch",
            )
        else:
            st.info("Run a prediction to populate the request dashboard.")
    if info:
        importance = pd.DataFrame(info["feature_importance"]).head(15).sort_values("importance")
        st.plotly_chart(
            px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Like classifier feature importance",
                color_discrete_sequence=[ACCENT],
            ),
            width="stretch",
        )
        st.caption(
            f"Model: {info['model_name']} · version {info['model_version']} · training cutoff {info['training_cutoff']}"
        )
