"""
app.py
------
Streamlit UI for the Blood Cell Classification pipeline.

Tabs:
  1. Model Status      — uptime, model metadata, scheduler status
  2. Predict           — single image upload and prediction
  3. Visualisations    — 3 dataset feature interpretations
  4. Upload & Retrain  — bulk upload + manual + autonomous retrain controls
  5. History           — past retraining run logs
"""

import io
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
PALETTE = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

st.set_page_config(
    page_title="Blood Cell Classifier",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🩸 Blood Cell Classifier")
st.sidebar.markdown(
    "African Leadership University · BSE\n\n"
    "White blood cell differential counting using deep learning.\n\n"
    f"**API:** `{API_URL}`"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Status",
    "Predict",
    "Visualisations",
    "Upload & Retrain",
    "History"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Model Status & Uptime")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh Status"):
            st.rerun()

    try:
        r    = requests.get(f"{API_URL}/status", timeout=5)
        data = r.json()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Uptime", data.get("uptime_human", "—"))
        with col2:
            acc = data.get("accuracy")
            st.metric("Test Accuracy", f"{acc}%" if acc else "—")
        with col3:
            f1 = data.get("f1")
            st.metric("Test F1 Score", f"{f1}%" if f1 else "—")
        with col4:
            pending = data.get("pending_images_for_retrain", "—")
            threshold = data.get("retrain_threshold", 50)
            st.metric(
                "Images Pending Retrain",
                f"{pending} / {threshold}",
                delta=f"{pending} uploaded" if isinstance(pending, int) else None
            )

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Model Information")
            st.json({
                "Model"      : data.get("model_name", "Custom CNN"),
                "Classes"    : data.get("class_names", CLASSES),
                "Input Size" : data.get("img_size", [96, 96]),
                "Model File" : data.get("model_file"),
                "File Exists": data.get("model_exists", False),
            })
        with col_b:
            st.subheader("Autonomous Retraining Scheduler")
            sched_running = data.get("scheduler_running", False)
            st.success("Scheduler Running") if sched_running else st.error("Scheduler Stopped")
            st.json({
                "Retrain Threshold" : f"{threshold} new images",
                "Check Interval"    : "Every 1 hour",
                "Next Auto-Retrain" : data.get("next_auto_retrain", "—"),
                "Triggered By"      : "Threshold OR hourly check",
            })

        # Progress bar showing pending images vs threshold
        if isinstance(pending, int) and threshold:
            progress = min(pending / threshold, 1.0)
            st.progress(progress, text=f"Pending images: {pending} / {threshold} (auto-retrain threshold)")

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Is the server running?")
    except Exception as e:
        st.error(f"Error fetching status: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Single Image Prediction")
    st.markdown(
        "Upload a microscope image of a white blood cell to receive an "
        "automated classification. Supported formats: JPEG, PNG."
    )

    uploaded = st.file_uploader(
        "Choose a cell image",
        type=["jpg", "jpeg", "png"],
        help="Wright-Giemsa stained white blood cell microscope image"
    )

    if uploaded:
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(uploaded, caption=f"Uploaded: {uploaded.name}", use_column_width=True)

        with col_result:
            with st.spinner("Classifying..."):
                try:
                    r = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=30
                    )
                    result = r.json()

                    if r.status_code == 200:
                        pred    = result["prediction"]
                        label   = pred["label"]
                        conf    = pred["confidence"]
                        scores  = pred["all_scores"]

                        cls_idx = CLASSES.index(label)
                        color   = PALETTE[cls_idx]

                        st.markdown(f"### Prediction: :{color.replace('#','')}[{label}]")
                        st.markdown(f"**Confidence:** {conf:.1f}%")
                        st.progress(conf / 100)

                        st.divider()
                        st.subheader("Scores for All Classes")

                        # Horizontal bar chart
                        fig, ax = plt.subplots(figsize=(6, 3))
                        bars = ax.barh(
                            list(scores.keys()),
                            list(scores.values()),
                            color=[PALETTE[CLASSES.index(c)] for c in scores.keys()]
                        )
                        ax.set_xlabel("Confidence (%)")
                        ax.set_xlim(0, 100)
                        for bar, val in zip(bars, scores.values()):
                            ax.text(
                                bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                                f"{val:.1f}%", va="center", fontsize=9
                            )
                        ax.spines[["top","right"]].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    else:
                        st.error(f"API error {r.status_code}: {result.get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to API at {API_URL}.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    st.divider()
    st.caption(
        "**Supported cell types:** Eosinophil · Lymphocyte · Monocyte · Neutrophil. "
        "Upload a clear microscope image for best results."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALISATIONS (3 required by rubric, with interpretations)
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Dataset Visualisations")
    st.markdown(
        "Three feature-level visualisations from the blood cell dataset with "
        "clinical interpretations. Each visual answers a specific analytical question."
    )

    # ── Visualisation 1: Model Performance Comparison ─────────────────────────
    st.subheader("1. Model Performance Comparison Across Experiments")
    st.markdown(
        "**What this shows:** The weighted F1 score and test accuracy achieved by each "
        "of the three models trained in this pipeline.\n\n"
        "**Interpretation:** The Custom CNN trained from scratch achieved 99.84% F1, "
        "outperforming both transfer learning models on this dataset. This is explained "
        "by the fact that blood cell microscopy images are visually very different from "
        "the natural images that MobileNetV2 and EfficientNetB0 were pre-trained on. "
        "The custom CNN, trained entirely on cell morphology, learns exactly the features "
        "that distinguish the four cell types — granule texture, nuclear lobe count, and "
        "cytoplasm-to-nucleus ratio."
    )

    try:
        r = requests.get(f"{API_URL}/results", timeout=5)
        if r.status_code == 200:
            metadata = r.json().get("metadata", {})
            results_dict = metadata.get("all_results", {})
            if results_dict:
                df_results = pd.DataFrame(results_dict)
                models     = df_results.get("Model", {})
                f1_scores  = df_results.get("F1", {})
                acc_scores = df_results.get("Accuracy", {})

                fig, ax = plt.subplots(figsize=(10, 4))
                x  = np.arange(len(models))
                w  = 0.35
                b1 = ax.bar(x - w/2, list(f1_scores.values()), w,
                            label="F1 Score (%)", color="#3498DB", alpha=0.85)
                b2 = ax.bar(x + w/2, list(acc_scores.values()), w,
                            label="Accuracy (%)", color="#E74C3C", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(list(models.values()), rotation=10)
                ax.set_ylabel("Score (%)")
                ax.set_ylim(90, 101)
                ax.legend()
                ax.axhline(97, color="gray", linestyle="--", lw=1.5, label="97% target")
                ax.spines[["top","right"]].set_visible(False)
                for bar in list(b1) + list(b2):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{bar.get_height():.1f}",
                        ha="center", fontsize=8
                    )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No results data available yet.")
        else:
            st.warning("Could not load results from API.")
    except Exception:
        # Fallback static visualisation if API is offline
        fig, ax = plt.subplots(figsize=(10, 4))
        models  = ["Custom CNN", "MobileNetV2 FT", "EfficientNetB0 FT"]
        f1s     = [99.84, 98.50, 97.80]
        accs    = [99.84, 98.50, 97.80]
        x       = np.arange(len(models))
        w       = 0.35
        ax.bar(x - w/2, f1s,  w, label="F1 Score (%)",  color="#3498DB", alpha=0.85)
        ax.bar(x + w/2, accs, w, label="Accuracy (%)", color="#E74C3C", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10)
        ax.set_ylim(90, 101)
        ax.legend()
        ax.axhline(97, color="gray", linestyle="--", lw=1.5)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("(Static fallback — API offline)")

    st.divider()

    # ── Visualisation 2: Class Distribution ───────────────────────────────────
    st.subheader("2. Class Distribution in the Combined Dataset")
    st.markdown(
        "**What this shows:** The number of images per white blood cell type across "
        "the full 12,444-image dataset after combining original TRAIN and TEST folders.\n\n"
        "**Interpretation:** The dataset is near-perfectly balanced with approximately "
        "3,100 images per class (balance ratio = 1.01). This confirms that no class "
        "weighting is required during training and that the model is not biased towards "
        "any particular cell type by sheer image volume. In clinical practice, neutrophils "
        "are the most common WBC (50–70% of all white cells), but the training dataset "
        "deliberately equalises class frequency to ensure the model learns all four types equally."
    )

    class_counts = {
        "EOSINOPHIL" : 3120,
        "LYMPHOCYTE" : 3103,
        "MONOCYTE"   : 3098,
        "NEUTROPHIL" : 3123
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        class_counts.keys(),
        class_counts.values(),
        color=PALETTE, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars, class_counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(val), ha="center", fontweight="bold", fontsize=10
        )
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, 3500)
    ax.axhline(
        np.mean(list(class_counts.values())),
        color="gray", linestyle="--", lw=1.5,
        label=f"Mean: {np.mean(list(class_counts.values())):.0f}"
    )
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Class Distribution — Balance Ratio: 1.01x", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Visualisation 3: Per-class Performance (from training results) ─────────
    st.subheader("3. Per-Class Recall — Clinical Difficulty of Each Cell Type")
    st.markdown(
        "**What this shows:** The recall (sensitivity) of the best model on each "
        "white blood cell type — i.e., what percentage of each class it correctly identifies.\n\n"
        "**Interpretation:** Lymphocytes achieve near-perfect recall (≈ 100%) because "
        "their large, dense, round nucleus with minimal cytoplasm is visually distinctive. "
        "Eosinophils are identifiable by their bright pink granules (eosin affinity). "
        "Monocytes and neutrophils are the hardest pair — both are granulocytes with "
        "irregular lobed nuclei. This confusion is not a model failure; it reflects "
        "genuine biological morphological similarity that even trained haematologists "
        "can find challenging at high magnification."
    )

    per_class_recall = {
        "EOSINOPHIL" : 98.9,
        "LYMPHOCYTE" : 100.0,
        "MONOCYTE"   : 99.6,
        "NEUTROPHIL" : 99.6
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        per_class_recall.keys(),
        per_class_recall.values(),
        color=PALETTE, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars, per_class_recall.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 1.5,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11,
            color="white"
        )
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(95, 101)
    ax.axhline(97, color="gray", linestyle="--", lw=1.5, label="97% target threshold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Per-Class Recall — Custom CNN (Test Set)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — UPLOAD & RETRAIN
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Upload Data & Retrain")

    col_upload, col_retrain = st.columns([1, 1])

    # ── Bulk Upload ────────────────────────────────────────────────────────────
    with col_upload:
        st.subheader("Bulk Image Upload")
        st.markdown(
            "Upload new labelled cell images. These are stored in Supabase and "
            "will be used in the next retraining cycle — either triggered manually "
            "below or automatically when the threshold is reached."
        )

        label_choice = st.selectbox(
            "Cell type label for all uploaded images",
            options=CLASSES,
            help="All files in this upload batch will receive this label."
        )

        uploaded_files = st.file_uploader(
            "Select images to upload",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} file(s) selected — label: **{label_choice}**")

            if st.button("Upload to Supabase", type="primary"):
                with st.spinner(f"Uploading {len(uploaded_files)} images..."):
                    try:
                        files = [
                            ("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                            for f in uploaded_files
                        ]
                        r = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            params={"label": label_choice},
                            timeout=120
                        )
                        result = r.json()
                        if r.status_code == 200:
                            st.success(
                                f"Uploaded {result['uploaded']} images successfully. "
                                f"Failed: {result['failed']}."
                            )
                            st.info(
                                f"Pending images for auto-retrain: "
                                f"**{result['pending_total']} / {result['retrain_threshold']}**"
                            )
                            if result.get("auto_retrain_triggered"):
                                st.success(
                                    "Threshold reached — autonomous retraining has been triggered automatically."
                                )
                        else:
                            st.error(f"Upload failed: {result.get('detail')}")
                    except Exception as e:
                        st.error(f"Upload error: {e}")

    # ── Manual Retrain ─────────────────────────────────────────────────────────
    with col_retrain:
        st.subheader("Manual Retraining Trigger")
        st.markdown(
            "Press the button below to immediately trigger retraining on all "
            "uploaded images that have not yet been used in a retraining run. "
            "The model is only saved if weighted F1 improves."
        )

        st.info(
            "**Autonomous retraining** also runs automatically every hour. "
            "When 50 or more new images accumulate, the scheduler triggers "
            "retraining with no human action required."
        )

        if st.button("Trigger Retraining Now", type="primary"):
            with st.spinner("Retraining in progress... this may take several minutes."):
                try:
                    r      = requests.post(f"{API_URL}/retrain", timeout=600)
                    result = r.json()

                    if result.get("status") == "skipped":
                        st.warning(f"Retraining skipped: {result.get('reason')}")
                    elif result.get("status") == "complete" or "f1_before" in result:
                        if result.get("improved"):
                            st.success(
                                f"Model improved! F1: {result['f1_before']} → {result['f1_after']}"
                            )
                        else:
                            st.info(
                                f"No improvement. F1: {result['f1_before']} → {result['f1_after']}. "
                                "Original model retained."
                            )
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Images Used",  result.get("images_used"))
                        col_b.metric("Epochs Run",   result.get("epochs_run"))
                        col_c.metric("Duration",     f"{result.get('duration_s')}s")
                    else:
                        st.error(f"Unexpected response: {result}")

                except requests.exceptions.Timeout:
                    st.error("Retraining timed out. Check API logs.")
                except Exception as e:
                    st.error(f"Retrain error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Retraining History")
    st.markdown(
        "All past retraining runs — both manual (triggered by button) and "
        "autonomous (triggered by the scheduler) — are logged here."
    )

    if st.button("Refresh History"):
        st.rerun()

    try:
        r = requests.get(f"{API_URL}/history", timeout=10)
        if r.status_code == 200:
            runs = r.json().get("runs", [])
            if runs:
                df = pd.DataFrame(runs)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Runs",       len(df))
                col2.metric("Improved Runs",    int(df["improved"].sum()) if "improved" in df else "—")
                col3.metric("Images Processed", int(df["images_used"].sum()) if "images_used" in df else "—")

                st.divider()

                # Table
                display_cols = [c for c in [
                    "triggered_at", "triggered_by", "images_used",
                    "f1_before", "f1_after", "improved", "epochs_run", "duration_s"
                ] if c in df.columns]
                st.dataframe(
                    df[display_cols].rename(columns={
                        "triggered_at" : "Time",
                        "triggered_by" : "Trigger",
                        "images_used"  : "Images",
                        "f1_before"    : "F1 Before",
                        "f1_after"     : "F1 After",
                        "improved"     : "Improved",
                        "epochs_run"   : "Epochs",
                        "duration_s"   : "Duration (s)"
                    }),
                    use_container_width=True
                )

                # F1 trend chart if multiple runs
                if len(df) > 1 and "f1_after" in df.columns:
                    st.subheader("F1 Score Trend Across Retraining Runs")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(range(1, len(df)+1), df["f1_before"][::-1],
                            marker="o", label="F1 Before", color="#E74C3C", lw=2)
                    ax.plot(range(1, len(df)+1), df["f1_after"][::-1],
                            marker="s", label="F1 After",  color="#3498DB", lw=2)
                    ax.set_xlabel("Run Number")
                    ax.set_ylabel("Weighted F1")
                    ax.set_ylim(0.8, 1.02)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.spines[["top","right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("No retraining runs recorded yet. Upload images and trigger retraining.")
        else:
            st.error(f"API error: {r.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")
