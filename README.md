# 🧑‍🤝‍🧑 Face Clustering — Person Identification System

Automatically group unlabeled face images by identity using **ArcFace embeddings**
and **DBSCAN clustering** — no training required, no labels required, no known
number of people required.

---

## Overview

Given a folder of mixed, unlabeled photos (different people, lighting, angles,
and expressions), this system:

1. Detects every face in every image (RetinaFace via InsightFace)
2. Aligns and embeds each face into a 512-d ArcFace vector
3. Clusters embeddings by identity using DBSCAN (cosine distance)
4. Assigns a 0–100% confidence score to every face
5. Writes one output folder per person + a `results.csv` report

Built for a Computer Vision Engineer interview assignment, but structured like
a real production service: modular, tested, containerized, and deployable via
both a REST API and an interactive dashboard.

---

## Features

- ✅ Unknown number of people — no need to specify K
- ✅ Multi-face-per-image support
- ✅ Automatic noise/outlier detection (DBSCAN `-1` label)
- ✅ Blur detection & near-duplicate filtering before clustering
- ✅ Per-image confidence scoring (centroid or k-NN based)
- ✅ Embedding disk cache (skip re-computation on repeated runs)
- ✅ Streamlit dashboard with live progress, search, and CSV export
- ✅ FastAPI REST service with Swagger docs
- ✅ Cluster visualizations: t-SNE/UMAP projection, collages, stat charts
- ✅ Dockerized, with CPU/GPU toggle
- ✅ Unit-tested core modules

---

## Architecture

```
┌──────────────┐
│ dataset/       │
└──────┬───────┘
       ▼
┌────────────────────┐
│ Face Detection          │  RetinaFace (InsightFace)
└──────┬─────────────┘
       ▼
┌────────────────────┐
│ Face Alignment           │  5-point landmark warp → 112×112
└──────┬─────────────┘
       ▼
┌────────────────────┐
│ Embedding Extraction      │  ArcFace → 512-d, L2-normalized
└──────┬─────────────┘
       ▼
┌────────────────────┐
│ DBSCAN Clustering          │  cosine distance, unknown K
└──────┬─────────────┘
       ▼
┌────────────────────┐
│ Confidence Scoring          │  cosine similarity to centroid
└──────┬─────────────┘
       ▼
┌────────────────────┐
│ output/cluster_N/ + CSV      │
└────────────────────┘
```

See `src/` for the module that implements each stage — every file maps 1:1 to
a pipeline stage (single responsibility principle).

---

## Pipeline

```
load_images → detect_faces → quality_filter (blur/duplicate)
  → align → extract_embeddings → DBSCAN clustering
  → confidence scoring → save CSV → save cluster folders → visualizations
```

---

## Installation

```bash
git clone https://github.com/<your-username>/person-identification.git
cd person-identification

python -m venv .venv
source .venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> First run downloads the InsightFace `buffalo_l` model pack (~300MB) into `models/`.

### Requirements
- Python 3.11+
- ~2GB RAM minimum for CPU inference on small batches
- Optional: CUDA-capable GPU + `onnxruntime-gpu` for faster inference

---

## Dataset Format

```
dataset/
├── img001.jpg
├── img002.png
├── subfolder_ok_too/
│   └── img003.jpg
```

- Any mix of `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- No folder structure or labeling required
- Subfolders are scanned recursively

---

## Usage

### CLI (batch pipeline)

```bash
python src/main.py --dataset dataset/ --output output/
```

Options:
- `--dataset PATH` — input folder (default: `dataset/`)
- `--output PATH` — output folder (default: `output/`)
- `--no-visualize` — skip t-SNE/collage generation for faster runs

### Streamlit dashboard

```bash
streamlit run src/streamlit_app.py
```
Then open `http://localhost:8501`, upload images, tune `eps`/`min_samples`, and
run clustering interactively.

### FastAPI service

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
Swagger docs at `http://localhost:8000/docs`. Endpoints:

| Method | Path          | Description                              |
|--------|---------------|-------------------------------------------|
| GET    | `/health`     | Liveness check                            |
| POST   | `/predict`    | Embeddings for a single uploaded image    |
| POST   | `/cluster`    | Full clustering run on an uploaded batch  |
| GET    | `/clusters`   | Most recent clustering results, grouped   |
| GET    | `/statistics` | Summary stats of the most recent run      |

### Docker

```bash
docker compose up --build
```
Starts the API on `:8000` and the dashboard on `:8501`, with `dataset/`,
`output/`, `models/`, and `logs/` mounted as volumes.

---

## Output

```
output/
├── cluster_0/
│   ├── img1_face0.jpg
│   └── img4_face0.jpg
├── cluster_1/
│   └── img2_face0.jpg
├── noise/
│   └── img7_face0.jpg
├── results.csv
└── _visualizations/
    ├── tsne_projection.png
    ├── cluster_statistics.png
    ├── collage_cluster_0.jpg
    └── cluster_summary.csv
```

### `results.csv` columns

| Column            | Description                                   |
|-------------------|------------------------------------------------|
| image_name        | Source filename                                |
| image_path        | Full path to the source image                  |
| face_index        | Index of this face within the source image      |
| cluster_id        | Assigned cluster (`-1` = unclustered/noise)     |
| confidence        | 0–100 confidence score                          |
| embedding_dim     | 512 (ArcFace output dimension)                  |
| detection_score   | RetinaFace detection confidence (0–1)           |
| bbox              | `x1,y1,x2,y2` pixel coordinates                 |
| is_noise          | Boolean, `True` if `cluster_id == -1`           |

---

## Testing

```bash
pytest --cov=src tests/
```

Covers detection filtering/error-handling, embedding normalization & caching,
DBSCAN cluster separation & noise handling, and both confidence-scoring
strategies.

---

## Limitations

- Confidence scores are geometric heuristics (similarity to centroid/kNN),
  not calibrated probabilities.
- DBSCAN's `eps`/`min_samples` are dataset-dependent; the defaults are tuned
  for typical ArcFace similarity distributions but may need adjustment for
  very small or very noisy datasets.
- Extreme pose (>60° yaw) or heavy occlusion (masks, sunglasses) reduces
  detection and embedding quality, as with any 2D face-recognition system.
- Not designed for real-time video; this is a batch/offline clustering tool.

---

## Future Improvements

- HDBSCAN as an alternative to DBSCAN for varying-density clusters
- Active-learning loop: let a human correct low-confidence clusters, feed
  corrections back to auto-tune `eps`
- Face-quality-aware weighting in centroid computation
- GPU batch inference for large datasets (see Optimization notes below)
- Cluster merging suggestions based on inter-cluster centroid similarity

---

## Optimization Notes

- **GPU acceleration**: set `INSIGHTFACE_CTX_ID=0` and install
  `onnxruntime-gpu` to run detection + embedding on CUDA.
- **Batch inference**: process images in batches through the ONNX session
  rather than one at a time to amortize overhead (see `embedding.py` for the
  single-image path to extend).
- **ONNX optimization**: use `onnxruntime`'s graph optimization level
  `ORT_ENABLE_ALL` and consider quantized models for CPU-only deployments.
- **Parallel processing**: detection/embedding across images is embarrassingly
  parallel — a `multiprocessing.Pool` or async worker queue scales linearly
  with cores for CPU inference.
- **Memory optimization**: stream embeddings to disk (e.g. memory-mapped
  numpy arrays) instead of holding all embeddings in RAM for datasets beyond
  ~100K images; DBSCAN itself can also be swapped for a mini-batch / FAISS-
  index-backed approximate neighbor approach at that scale.
- **100K+ images**: precompute and cache embeddings first (embarrassingly
  parallel, cache-friendly), then run clustering as a separate, much cheaper
  second pass — this is exactly why `embedding.py`'s disk cache is keyed
  independently of the clustering step.

---

## License

MIT — see [LICENSE](LICENSE).
