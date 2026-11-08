"""English interactive client for the recommendation API."""

import os
from io import StringIO

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="YouTube Shorts Recommendation", page_icon="▶", layout="wide")
st.title("YouTube Shorts Recommendation")
st.caption("Point-in-time features · model comparison · monitored inference")


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


with st.sidebar:
    st.header("Service")
    health = call_api("/health")
    if health:
        st.success("Model ready" if health["model_loaded"] else "Model unavailable")
        st.caption(f"Cache: {health['cache_backend']}")
    st.caption(
        "Like probabilities and confidence labels are uncalibrated. This demo does not measure online engagement lift."
    )

single, batch, recommendation, monitoring = st.tabs(
    ["Single prediction", "Batch prediction", "Recommendations", "Monitoring"]
)
with single:
    st.subheader("Predict engagement after observed watch time")
    with st.form("prediction"):
        left, right = st.columns(2)
        user = left.text_input("User ID", "user_1")
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
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E84855"}},
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
        "user_id,video_id,watch_time,hour_of_day\nuser_1,video_1,20,12\nuser_2,video_2,35,18\n",
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
                output, x="prediction_probability", nbins=25, title="Like probability distribution"
            ),
            width="stretch",
        )
        st.download_button(
            "Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv"
        )

with recommendation:
    st.subheader("Rank unseen videos before playback")
    st.caption(
        "A separate ranking model excludes observed watch duration and completion features. Up to 300 eligible candidates are ranked."
    )
    with st.form("recommendations"):
        rec_user = st.text_input("Recommendation user ID", "user_1")
        top_k = st.slider("Number of videos", 1, 50, 10)
        rec_submit = st.form_submit_button("Recommend videos", type="primary")
    if rec_submit:
        result = call_api("/recommend", {"user_id": rec_user, "top_k": top_k})
        if result:
            st.dataframe(pd.DataFrame(result["recommendations"]), width="stretch")

with monitoring:
    st.subheader("Live service dashboard")
    if st.button("Refresh monitoring"):
        st.rerun()
    metrics = call_api("/monitoring")
    info = call_api("/model/info")
    if metrics:
        columns = st.columns(3)
        columns[0].metric("Recent requests", metrics["window_requests"])
        columns[1].metric("Mean latency", f"{metrics['mean_latency_ms']:.1f} ms")
        columns[2].metric("Error rate", f"{metrics['error_rate']:.1%}")
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
                px.line(trend, x="timestamp", y="requests", title="Requests per minute"),
                width="stretch",
            )
            st.plotly_chart(
                px.histogram(
                    events, x="latency_ms", nbins=25, title="Response latency distribution"
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
                title="Selected model feature importance",
            ),
            width="stretch",
        )
        st.caption(
            f"Model: {info['model_name']} · version {info['model_version']} · training cutoff {info['training_cutoff']}"
        )
