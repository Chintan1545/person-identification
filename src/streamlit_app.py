"""
streamlit_app.py
=================
Interactive dashboard for the face-clustering pipeline.

Run with:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import settings
from main import FaceClusteringPipeline
from visualization import cluster_summary_table

st.set_page_config(page_title="Face Clustering Dashboard", page_icon="🧑‍🤝‍🧑", layout="wide")

# --------------------------------------------------------------------------- #
# Minimal dark-mode-friendly styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(127,127,127,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧑‍🤝‍🧑 Face Clustering Dashboard")
st.caption("Upload a folder of unlabeled face images and automatically group them by identity.")

# --------------------------------------------------------------------------- #
# Sidebar: dataset upload + run controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. Dataset")
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
    )

    st.header("2. Clustering settings")
    eps = st.slider("DBSCAN eps (cosine distance)", 0.1, 0.8, settings.clustering.eps, 0.01)
    min_samples = st.slider("DBSCAN min_samples", 1, 10, settings.clustering.min_samples, 1)

    run_button = st.button("🚀 Run clustering", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.output_dir = None


def run_pipeline(files, eps: float, min_samples: int) -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="face_cluster_"))
    dataset_dir = work_dir / "dataset"
    output_dir = work_dir / "output"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        (dataset_dir / f.name).write_bytes(f.getbuffer())

    progress = st.progress(0.0, text="Initializing pipeline...")

    pipeline = FaceClusteringPipeline(dataset_dir=dataset_dir, output_dir=output_dir)
    # Override clustering params from the sidebar sliders
    pipeline.clusterer.eps = eps
    pipeline.clusterer.min_samples = min_samples

    progress.progress(0.2, text="Detecting faces & extracting embeddings...")
    pipeline.run()
    progress.progress(1.0, text="Done!")

    csv_path = settings.paths.results_csv
    if csv_path.exists():
        st.session_state.results_df = pd.read_csv(csv_path)
        st.session_state.output_dir = output_dir
    else:
        st.session_state.results_df = None
        st.error("No results were generated - check that faces were detected.")


if run_button:
    if not uploaded_files:
        st.warning("Please upload at least one image first.")
    else:
        with st.spinner("Running face clustering pipeline..."):
            run_pipeline(uploaded_files, eps, min_samples)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
df = st.session_state.results_df

if df is not None:
    st.success("Clustering complete!")

    n_clusters = df.loc[df["cluster_id"] != -1, "cluster_id"].nunique()
    n_noise = (df["cluster_id"] == -1).sum()
    avg_conf = df["confidence"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total faces", len(df))
    c2.metric("People found", n_clusters)
    c3.metric("Unclustered (noise)", int(n_noise))
    c4.metric("Avg. confidence", f"{avg_conf:.1f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📁 Clusters", "📊 Statistics", "🔍 Search"])

    with tab1:
        for cluster_id in sorted(df["cluster_id"].unique()):
            label = "Unclustered / Noise" if cluster_id == -1 else f"Person {cluster_id}"
            subset = df[df["cluster_id"] == cluster_id]
            with st.expander(f"{label} ({len(subset)} images)"):
                cols = st.columns(6)
                for i, (_, row) in enumerate(subset.iterrows()):
                    with cols[i % 6]:
                        img_path = Path(row["image_path"])
                        if img_path.exists():
                            st.image(str(img_path), caption=f"{row['confidence']:.0f}%")

    with tab2:
        summary = cluster_summary_table(df)
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(df["cluster_id"].value_counts().sort_index())
        st.write("Confidence distribution")
        st.bar_chart(df["confidence"])

    with tab3:
        query = st.text_input("Search by image filename")
        if query:
            matches = df[df["image_name"].str.contains(query, case=False, na=False)]
            st.dataframe(matches, use_container_width=True)

    st.divider()
    st.download_button(
        "⬇️ Download results.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )
else:
    st.info("Upload images and click **Run clustering** to get started.")
