# app.py
import os
import re
import glob
import zipfile
import pathlib
import tempfile
import io
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

# --- Summary agent import (with safe fallback) ---
try:
    from agents.summary_agent import generate_run_summary
except Exception as _import_err:
    def generate_run_summary(_state):
        return f"# AutoML Run Summary\n\n_Summary agent unavailable: {_import_err}_"

# --- Cost tracking import ---
from cost_tracker import APIUsageTracker

load_dotenv()  # load KAGGLE_USERNAME / KAGGLE_KEY from .env if present

import pandas as pd
import streamlit as st
from agents.graph_orchestrator import create_graph

# ====== PDF export (reportlab) ======
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Preformatted, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# =========================
# Runtime flags (for LLM-generated code execution guardrails)
# =========================
os.environ.setdefault("ALLOW_IO", "1")
os.environ.setdefault("ALLOW_TUNING", "0")
os.environ.setdefault("ALLOWED_DATA_DIR", str(pathlib.Path(__file__).parent.resolve()))

# CSV size limit (approx). You can raise/lower this via env.
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", "150000000"))  # ~150 MB

st.set_page_config(page_title="AutoML Assistant with RAG ", layout="wide")

# ---------- Session state defaults ----------
defaults = {
    "df": None,
    "source_name": None,
    "target_column": None,
    "final_state": None,   # full graph output
    "metrics": None,
    "model": None,
    "profile": None,
    "code_map": None,
    "steps": None,
    "summary_md": None,
    "cost_tracker": APIUsageTracker(),
    "pipeline_progress": None,  # For progress tracking
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

st.title("AutoML Assistant with RAG")
st.markdown("""
### Overview
This application builds a supervised machine-learning pipeline for a tabular CSV dataset.

**What happens under the hood**
- LLM-generated pipeline: The system synthesizes preprocessing and model-training code that follows a strict code contract. Execution is sandboxed with targeted auto-repairs for common issues.
- Deterministic baseline: A BaselineSelector trains conventional, strong models (Logistic Regression / Random Forest for classification; Ridge / RandomForestRegressor for regression) with robust preprocessing.
- Selection and safety: Both approaches are evaluated on a fresh hold-out split. The better performing model is used. If LLM code cannot produce a valid model, the baseline is used automatically. Heavy tuning and file I/O in generated code are disabled by default.

**What you'll see**
- Dataset profile and inferred task type
- Hold-out metrics (Accuracy/F1 for classification, RMSE/R² for regression) and cross-validation details for the baseline (when available)
- On-demand confusion matrix
- The generated step plan and code map
- Any charts emitted by the generated code
- Real-time API cost tracking
- Step-by-step progress display

**Data size guidance**
By default this app reads up to **~150 MB** per CSV (configurable via `MAX_CSV_BYTES`). Larger files may be slow or exhaust memory.
""")

# =========================
# Progress Display Functions
# =========================

def display_progress_ui():
    """Display real-time progress of AutoML pipeline"""
    if 'pipeline_progress' not in st.session_state or st.session_state.pipeline_progress is None:
        return
    
    progress_data = st.session_state.pipeline_progress
    
    # Progress bar
    total_steps = len(progress_data.get('all_steps', []))
    completed_steps = len(progress_data.get('completed_steps', []))
    current_step = progress_data.get('current_step', '')
    
    if total_steps > 0:
        progress_pct = min(completed_steps / total_steps, 1.0)
        st.progress(progress_pct)
        
        if completed_steps < total_steps:
            st.caption(f"Step {completed_steps + 1} of {total_steps}: {current_step}")
        else:
            st.caption(f"Complete! All {total_steps} steps finished.")
    
    # Step checklist
    with st.expander("Pipeline Progress", expanded=True):
        for i, step in enumerate(progress_data.get('all_steps', [])):
            if i < completed_steps:
                st.success(f"✅ {step}")
            elif i == completed_steps and completed_steps < total_steps:
                st.info(f"🔄 {step} (In Progress)")
            else:
                st.write(f"⏳ {step}")

def update_progress(step_name: str):
    """Update progress and refresh UI"""
    if 'pipeline_progress' not in st.session_state or st.session_state.pipeline_progress is None:
        return
        
    st.session_state.pipeline_progress['current_step'] = step_name
    
    # Mark previous step as complete if we're moving to next step
    completed = st.session_state.pipeline_progress['completed_steps']
    all_steps = st.session_state.pipeline_progress['all_steps']
    
    # Find which step we're currently on and mark previous ones complete
    current_step_index = None
    for i, step in enumerate(all_steps):
        if step_name.lower() in step.lower() or step.lower() in step_name.lower():
            current_step_index = i
            break
    
    if current_step_index is not None:
        # Mark all steps up to current as completed
        for i in range(current_step_index):
            if all_steps[i] not in completed:
                completed.append(all_steps[i])

# =========================
# Utilities
# =========================

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(raw))

@st.cache_data(show_spinner=False)
def _read_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# pretty-print LLM code in the UI
_BULLET_LINE = re.compile(r'^\s*[-•]\s+')
# strip inline <img src="data:..."> before PDF/markdown to avoid massive base64 blobs
_IMG_DATAURI = re.compile(r'<img[^>]+src\s*=\s*"data:image/[^"]+"[^>]*>', re.I)

def _clean_code_for_display(snippet: str) -> str:
    """Normalize LLM code for Streamlit display."""
    if not isinstance(snippet, str):
        return str(snippet)
    s = snippet.replace("\r\n", "\n").replace("\r", "\n").strip()
    # strip accidental triple-backtick fences
    s = re.sub(r"^```(?:python)?\n?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n```$", "", s)
    # remove leading bullets sometimes added to code lines
    lines = []
    for ln in s.split("\n"):
        lines.append(_BULLET_LINE.sub("", ln.rstrip()))
    s = "\n".join(lines).strip()
    if not s.endswith("\n"):
        s += "\n"
    return s

def _friendly_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.0f} PB"

def load_from_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    try:
        size = getattr(uploaded_file, "size", None)
        if size and size > MAX_CSV_BYTES:
            return None, None, f"File is {_friendly_size(size)} which exceeds the limit ({_friendly_size(MAX_CSV_BYTES)})."
        raw = uploaded_file.read()
        if len(raw) > MAX_CSV_BYTES:
            return None, None, f"File is {_friendly_size(len(raw))} which exceeds the limit ({_friendly_size(MAX_CSV_BYTES)})."
        df = _read_csv_from_bytes(raw)
        return df, f"Upload: {uploaded_file.name}", None
    except Exception as e:
        return None, None, f"Failed to parse uploaded CSV: {e}"

def _parse_kaggle_input(text: str, explicit_csv: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Accepts either a full Kaggle URL like
      https://www.kaggle.com/datasets/OWNER/DATASET?select=file.csv
    or a plain slug 'OWNER/DATASET'. Returns (slug, csv_name or None).
    """
    s = (text or "").strip()
    csv_name = (explicit_csv or "").strip() or None
    if not s:
        raise ValueError("Provide a Kaggle dataset slug or URL.")
    if s.startswith(("http://", "https://")):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(s)
        parts = [p for p in u.path.split("/") if p]
        try:
            i = parts.index("datasets")
            owner, dataset = parts[i + 1], parts[i + 2]
        except Exception:
            raise ValueError("Could not parse Kaggle dataset from URL.")
        slug = f"{owner}/{dataset}"
        q_csv = parse_qs(u.query).get("select", [None])[0]
        csv_name = q_csv or csv_name
        return slug, csv_name
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", s):
        return s, csv_name
    raise ValueError("Invalid Kaggle dataset format. Use 'owner/dataset' or a Kaggle dataset URL.")

def load_from_kaggle(slug_or_url: str, csv_name: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Downloads a CSV from a Kaggle dataset using the Kaggle API and returns a DataFrame.
    Accepts either a slug 'owner/dataset' or a full Kaggle URL.
    Requires:
      - pip install kaggle
      - env KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return None, None, "Kaggle package not installed. Run: pip install kaggle"

    try:
        slug, csv_wanted = _parse_kaggle_input(slug_or_url, csv_name)
    except Exception as e:
        return None, None, f"{e}"

    try:
        api = KaggleApi()
        api.authenticate()

        with tempfile.TemporaryDirectory() as tmpdir:
            if csv_wanted:
                api.dataset_download_file(slug, csv_wanted, path=tmpdir, quiet=True)
            else:
                api.dataset_download_files(slug, path=tmpdir, quiet=True)

            # unzip downloaded archives
            for z in glob.glob(os.path.join(tmpdir, "*.zip")):
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(tmpdir)

            csvs = glob.glob(os.path.join(tmpdir, "*.csv"))
            if not csvs:
                return None, None, "No CSV files found in the Kaggle dataset."

            chosen = None
            if csv_wanted:
                for c in csvs:
                    if os.path.basename(c) == csv_wanted:
                        chosen = c
                        break
                if chosen is None:
                    chosen = csvs[0]
            else:
                chosen = csvs[0]

            df = _read_csv_from_path(chosen)
            label = f"Kaggle: {slug} ({os.path.basename(chosen)})"
            return df, label, None
    except Exception as e:
        return None, None, f"Kaggle load failed: {e}"

# ---------- PDF builder ----------
def _build_pdf_from_markdown(summary_md: str, plots: list) -> bytes:
    """Markdown -> PDF for headings, bullets, code; plots are page-broken and size-restricted."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    base_code_parent = styles["BodyText"] if "Code" not in styles.byName else styles["Code"]
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceAfter=8)
    body = ParagraphStyle("body", parent=styles["BodyText"], leading=14, spaceAfter=6)
    code_style = ParagraphStyle(
        "code", parent=base_code_parent, fontName="Courier",
        fontSize=9, leading=11, backColor=colors.whitesmoke, borderPadding=4,
    )

    flow = []
    in_code = False
    code_lines: list[str] = []

    # Strip any inline <img src="data:..."> so it doesn't dump base64 into the PDF
    text = _IMG_DATAURI.sub("", summary_md or "")

    for raw in text.splitlines():
        line = raw.rstrip()

        # toggle code block
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                flow.append(Preformatted("\n".join(code_lines), code_style))
                flow.append(Spacer(1, 6))
            continue

        if in_code:
            code_lines.append(line)
            continue

        # headings
        if line.startswith("# "):
            flow.append(Paragraph(line[2:].strip(), h1)); flow.append(Spacer(1, 6)); continue
        if line.startswith("## "):
            flow.append(Paragraph(line[3:].strip(), h2)); flow.append(Spacer(1, 4)); continue

        # bullets
        if line.strip().startswith("- "):
            flow.append(Paragraph("• " + line.strip()[2:], body)); continue

        # blank
        if line.strip() == "":
            flow.append(Spacer(1, 6)); continue

        # default paragraph
        flow.append(Paragraph(line, body))

    # ----- Append images with size restriction -----
    if plots:
        max_w = A4[0] - doc.leftMargin - doc.rightMargin - 6
        max_h = A4[1] - doc.topMargin - doc.bottomMargin - 12

        flow.append(Spacer(1, 12))
        flow.append(Paragraph("Generated Visualizations", h2))

        first = True
        for p in plots:
            path = p.get("path")
            label = p.get("label", "Visualization")
            if not path or not os.path.exists(path):
                continue

            # Only page-break before images AFTER the first one
            if not first:
                flow.append(PageBreak())
            first = False

            flow.append(Paragraph(label, body))

            img = RLImage(path)
            try:
                img._restrictSize(max_w, max_h)  # keep aspect ratio, fit on page
            except Exception:
                pass

            flow.append(img)
            flow.append(Spacer(1, 10))

    doc.build(flow)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================
# Data source UI
# =========================
st.subheader("1) Load Data")

source_mode = st.radio(
    "Choose a source",
    ["Upload CSV", "Kaggle Dataset"],
    horizontal=True
)

df_loaded = False

if source_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
    if uploaded_file is not None:
        df, label, err = load_from_upload(uploaded_file)
        if err:
            st.error(err)
        else:
            st.session_state.df = df
            st.session_state.source_name = label
            df_loaded = True

else:  # Kaggle Dataset
    st.caption("Provide a Kaggle dataset **slug** like `uciml/iris` or paste the **full Kaggle dataset URL**. Optionally specify a CSV file name.")
    slug_or_url = st.text_input("Kaggle dataset (slug or URL)")
    csv_name = st.text_input("Specific CSV file name (optional)")
    if st.button("Load from Kaggle"):
        df, label, err = load_from_kaggle(slug_or_url.strip(), csv_name.strip() or None)
        if err:
            st.error(err)
        else:
            st.session_state.df = df
            st.session_state.source_name = label
            df_loaded = True

# If we already had data in session (from a previous run) and nothing new loaded, reuse it
if st.session_state.df is not None and not df_loaded:
    df_loaded = True

if not df_loaded:
    st.info(f"Awaiting data. Max recommended CSV size: {_friendly_size(MAX_CSV_BYTES)}.")
    st.stop()

# Show a small preview and shape
df = st.session_state.df
st.success(f"Loaded {st.session_state.source_name or 'dataset'} with shape {df.shape[0]} rows × {df.shape[1]} columns.")
st.dataframe(df.head(15), use_container_width=True)

# =========================
# 2) Select target
# =========================
st.subheader("2) Select Target Column")
# Safe target column selection
target_options = list(df.columns)
try:
    default_index = 0 if st.session_state.target_column is None else target_options.index(st.session_state.target_column)
except ValueError:
    # If cached target doesn't exist in new dataset, reset to first column
    default_index = 0
    st.session_state.target_column = None

target_column = st.selectbox(
    "Target column",
    options=target_options,
    index=default_index
)
st.session_state.target_column = target_column

# =========================
# 3) Run pipeline
# =========================
st.subheader("3) Run AutoML")

# Display current API costs
if 'cost_tracker' in st.session_state:
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    with cost_col1:
        current_cost = st.session_state.cost_tracker.get_current_run_total()
        st.metric("Current Run Cost", f"${current_cost:.4f}")
    with cost_col2:
        session_cost = st.session_state.cost_tracker.get_session_total()
        st.metric("Session Total", f"${session_cost:.4f}")
    with cost_col3:
        # Estimate cost based on dataset size (rough approximation)
        estimated_cost = min(0.02 + (len(df) / 10000) * 0.01, 0.10)  # Cap at $0.10
        st.metric("Estimated Cost", f"${estimated_cost:.4f}")

run_clicked = st.button("Run AutoML Pipeline", use_container_width=False)

if run_clicked:
    # Initialize progress tracking
    st.session_state.pipeline_progress = {
        'all_steps': [
            'Profiling Dataset', 
            'Retrieving Rules', 
            'Planning Pipeline', 
            'Building Pipeline', 
            'Evaluating Model', 
            'Generating Summary'
        ],
        'completed_steps': [],
        'current_step': 'Starting...'
    }
    
    # Progress container
    progress_container = st.empty()
    
    # Show initial progress state
    with progress_container.container():
        display_progress_ui()
    
    # Run the actual pipeline (this happens quickly behind the scenes)
    with st.spinner("Running AI-generated + Fallback pipeline..."):
        app = create_graph()
        input_state = {
            "df": df,
            "target_column": target_column,
            "profile": {"target": target_column},
            "rules": {},
            "steps": [],
            "fallback": False,
            "code_map": {},
            "model": None,
            "evaluation": {}
            # Remove progress_callback - we'll simulate it
        }
        
        final_state = app.invoke(input_state)
    
    # Now simulate step-by-step progress for visual effect
    steps = [
        'Profiling Dataset', 
        'Retrieving Rules', 
        'Planning Pipeline', 
        'Building Pipeline', 
        'Evaluating Model', 
        'Generating Summary'
    ]
    
    for i, step in enumerate(steps):
        st.session_state.pipeline_progress['current_step'] = step
        st.session_state.pipeline_progress['completed_steps'] = steps[:i+1]
        
        with progress_container.container():
            display_progress_ui()
        
        time.sleep(0.8)  # Visual delay between steps
    
    # Mark as complete
    st.session_state.pipeline_progress['current_step'] = 'Complete'
    st.session_state.pipeline_progress['completed_steps'] = steps.copy()
    
    with progress_container.container():
        display_progress_ui()
        
        # Mark as complete
        st.session_state.pipeline_progress['current_step'] = 'Complete'
        st.session_state.pipeline_progress['completed_steps'] = st.session_state.pipeline_progress['all_steps'].copy()
        
        # Final progress update
        with progress_container.container():
            display_progress_ui()
        
        # Mark run as complete for cost tracking
        if 'cost_tracker' in st.session_state:
            st.session_state.cost_tracker.finish_run()

    # Persist results for future reruns
    st.session_state.final_state = final_state
    st.session_state.metrics = final_state.get("evaluation", {}) or {}
    st.session_state.model   = final_state.get("model", None)
    st.session_state.profile = final_state.get("profile", {}) or {}
    st.session_state.code_map = final_state.get("code_map", {}) or {}
    st.session_state.steps    = final_state.get("steps", []) or []
    # If graph created a summary already, capture it
    if "summary_md" in final_state and final_state["summary_md"]:
        st.session_state.summary_md = final_state["summary_md"]

    st.success("Pipeline execution complete.")

# ---------- Render results if we have them in session ----------
if st.session_state.final_state is not None:
    profile = st.session_state.profile
    metrics = st.session_state.metrics
    model   = st.session_state.model

    # Display final cost breakdown
    if 'cost_tracker' in st.session_state:
        st.subheader("Cost Breakdown")
        breakdown = st.session_state.cost_tracker.get_breakdown()
        if breakdown:
            cost_df = pd.DataFrame([
                {"Operation": op, "Cost ($)": f"${cost:.4f}"} 
                for op, cost in breakdown.items()
            ])
            st.table(cost_df)
            total_run_cost = sum(breakdown.values())
            st.caption(f"Total pipeline cost: ${total_run_cost:.4f}")

    # Profile
    st.subheader("Dataset Profile")
    st.json(profile)

    # Evaluation
    st.subheader("Evaluation Metrics")
    st.caption("Scores for the final chosen model on a fresh hold-out split. These can differ from validation (CV) scores because they use different splits/metrics.")

    # Model origin + badge
    origin = getattr(model, "_origin", "unknown") if model is not None else "unknown"
    origin_label = "AI-generated (LLM)" if origin == "ai" else ("Baseline" if origin == "baseline" else "unknown")
    st.markdown(
        f"**Model origin:** <span style='padding:2px 6px; border-radius:6px; background:#1f6feb22; color:#58a6ff;'>{origin_label}</span>",
        unsafe_allow_html=True
    )

    # Numeric metrics table
    numeric_rows = [{"Metric": k, "Score": float(v)} for k, v in (metrics or {}).items() if isinstance(v, (int, float))]
    if numeric_rows:
        st.table(pd.DataFrame(numeric_rows).set_index("Metric"))
    else:
        st.info("No numeric metrics available.")

    split = (metrics or {}).get("split", {})
    if (metrics or {}).get("n_test") is not None:
        st.caption(
            f"Hold-out split • test_size={split.get('test_size', 0.2)} • "
            f"stratified={bool(split.get('stratified', False))} • "
            f"n_train={(metrics or {}).get('n_train')} • n_test={(metrics or {}).get('n_test')}"
        )

    # BaselineSelector summary (if present)
    bs = getattr(model, "_baseline_selector", None) if model is not None else None
    if bs:
        st.subheader("Validation Results (Cross-Validation)")
        st.caption(
            f"Cross-validated means used to select the baseline model. "
            f"Problem: {bs.get('problem')} • Primary metric: {bs.get('primary_metric')} • Folds: {bs.get('cv_folds')}"
        )
        res = pd.DataFrame(bs.get("results", {})).T
        if not res.empty:
            if bs.get("problem") == "classification":
                cols = [c for c in ["accuracy", "f1", "fit_time_s"] if c in res.columns]
                st.caption("Note: validation uses f1_weighted; your hold-out table may show a different F1 variant.")
            else:
                cols = [c for c in ["rmse", "r2", "fit_time_s"] if c in res.columns]
            st.table(res[cols])
        st.success(f"Winner: {bs.get('winner')}")

    # Generated visualizations
    plot_info = getattr(model, "_plot_info", {}) if model is not None else {}
    plot_images = [p for p in sorted(glob.glob("step_*_plot.png")) if re.match(r"^step_\d+_plot\.png$", os.path.basename(p))]

    if st.button("Show Confusion Matrix (final model)"):
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
            from sklearn.model_selection import train_test_split

            # Prefer matrix from evaluation if present
            if "confusion_matrix" in (metrics or {}):
                cm = np.array(metrics["confusion_matrix"])
                labels = metrics.get("labels")
                fig, ax = plt.subplots(figsize=(6, 4))
                disp = ConfusionMatrixDisplay(cm, display_labels=labels) if labels is not None else ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax, colorbar=False)
                st.pyplot(fig, clear_figure=True, use_container_width=False)
            else:
                # ad-hoc computation for classification
                df_cm = st.session_state.df
                target_column_cm = st.session_state.target_column
                X = df_cm.drop(columns=[target_column_cm])
                y = df_cm[target_column_cm]
                is_classification = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique() <= max(20, int(0.05 * len(y))))
                if not is_classification:
                    st.info("Confusion matrix applies to classification problems only.")
                elif model is None:
                    st.warning("No trained model available.")
                else:
                    stratify = y if (y.nunique() <= 20 and y.value_counts().min() >= 2) else None
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
                    y_pred = model.predict(X_te)
                    labels = np.unique(y_te)
                    cm = confusion_matrix(y_te, y_pred, labels=labels)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                    disp.plot(ax=ax, colorbar=False)
                    st.pyplot(fig, clear_figure=True, use_container_width=False)
        except Exception as e:
            st.error(f"Confusion matrix unavailable: {e}")

    if plot_images:
        st.subheader("Generated Visualizations")
        for img_path in plot_images:
            fname = os.path.basename(img_path)
            m = re.match(r"^step_(\d+)_plot\.png$", fname)
            step_idx = int(m.group(1)) if m else None
            label = plot_info.get(step_idx, "Visualization")
            st.markdown(f"**{label} (Step {step_idx})**" if step_idx else f"**{label}**")
            st.image(img_path, width=650)

    # ---------- Generated Steps & Code Map ----------
    st.subheader("Generated Steps")
    steps = st.session_state.steps or []
    if steps:
        for step in steps:
            st.markdown(f"- {step}")
    else:
        st.info("No steps generated.")

    st.subheader("Generated Code Map")
    code_map = st.session_state.code_map or {}
    if code_map:
        for step, code in code_map.items():
            with st.expander(f"Step: {step}"):
                st.code(_clean_code_for_display(code), language="python")
    else:
        st.info("No generated code.")

    # ---------- Run Summary (professional Markdown) ----------
    # Collect plot metadata so the summary can list them
    plot_meta = []
    if plot_images:
        for img_path in plot_images:
            fname = os.path.basename(img_path)
            m = re.match(r"^step_(\d+)_plot\.png$", fname)
            step_idx = int(m.group(1)) if m else None
            label = plot_info.get(step_idx, "Visualization")
            plot_meta.append({"step": step_idx, "label": label, "path": img_path})

    # Helper: extract final estimator name + params (works for sklearn Pipelines)
    def _extract_model_info(mdl):
        name = "UnknownModel"
        params = {}
        try:
            est = mdl
            if hasattr(mdl, "steps") and mdl.steps:
                est = mdl.steps[-1][1]  # take last step of sklearn Pipeline
                name = type(est).__name__
            if hasattr(est, "get_params"):
                try:
                    params = est.get_params(deep=False)
                except Exception:
                    params = est.get_params()
        except Exception:
            pass
        return name, params

    model_name, model_params = _extract_model_info(model)

    # Build a rich payload; the summary agent will ignore extra keys it doesn't use
    payload = {
        # Basic dataset info
        "dataset_name": st.session_state.get("source_name") or "Uploaded dataset",
        "n_rows": int(st.session_state.df.shape[0]) if st.session_state.df is not None else None,
        "n_cols": int(st.session_state.df.shape[1]) if st.session_state.df is not None else None,
        "target": st.session_state.target_column,

        # Origin / selection info
        "origin": getattr(model, "_origin", "unknown"),
        "baseline": getattr(model, "_baseline_selector", None),

        # Final estimator details
        "model_name": model_name,
        "model_params": model_params,

        # Full state for summary
        "profile": profile or {},
        "metrics": metrics or {},
        "plan": st.session_state.steps or [],
        "code_map": st.session_state.code_map or {},
        "plots": plot_meta,
        "confusion": {
            "matrix": (metrics or {}).get("confusion_matrix"),
            "labels": (metrics or {}).get("labels"),
        },
    }

    # Generate summary (persist so it survives reruns)
    try:
        st.session_state.summary_md = generate_run_summary(payload)
    except Exception as e:
        st.session_state.summary_md = f"# AutoML Run Summary\n\n_Summary agent error:_ {e}"

    # ---- Sanitize summary: remove inline data-URI <img> tags
    _summary_text = st.session_state.summary_md or ""
    if isinstance(_summary_text, bytes):
        _summary_text = _summary_text.decode("utf-8", errors="replace")
    _summary_text = _IMG_DATAURI.sub("[embedded image omitted]", _summary_text)

    # ---- Ensure images appear in MD as file links (for downloads); Streamlit will also show them inline next block
    if plot_meta:
        lines = [_summary_text, "\n## Embedded Plots\n"]
        for p in plot_meta:
            path = p.get("path")
            label = p.get("label", "Visualization")
            if path:
                lines.append(f"![{label}]({path})")
        _summary_text = "\n".join(lines)

    st.subheader("Run Summary")
    with st.expander("Show summary (Markdown)", expanded=True):
        st.markdown(_summary_text, unsafe_allow_html=False)

    # Show the same plots inline (reliable in Streamlit regardless of Markdown rendering)
    if plot_meta:
        st.markdown("### Embedded Plots (inline)")
        for p in plot_meta:
            if os.path.exists(p["path"]):
                st.image(p["path"], caption=p.get("label") or "Visualization", use_container_width=True)

    # ---- Markdown download ----
    st.download_button(
        label="Download summary (.md)",
        data=_summary_text.encode("utf-8"),
        file_name="run_summary.md",
        mime="text/markdown",
        use_container_width=False,
    )

    # ---- PDF download ----
    try:
        pdf_bytes = _build_pdf_from_markdown(_summary_text, plot_meta)
        st.download_button(
            label="Download summary (.pdf)",
            data=pdf_bytes,
            file_name="run_summary.pdf",
            mime="application/pdf",
            use_container_width=False,
        )
    except Exception as e:
        st.caption(f"PDF export unavailable: {e}")