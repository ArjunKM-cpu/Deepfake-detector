"""
AI Image Authenticity Checker & Deepfake Detection System.
Professional Streamlit Web Application providing real-time AI classification,
confidence scoring, dynamic Grad-CAM XAI heatmaps, batch gallery, and model analytics.
"""

import os
import json
import io
import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

# Custom module imports
from src.utils import get_device, validate_image_file, generate_synthetic_dataset, ensure_dir, IDX_TO_CLASS
from src.predict import predict_image
from src.explain import GradCAM, overlay_heatmap, explain_prediction
from src.train import train_model
from src.evaluate import evaluate_model
from src.data_preprocessing import get_transforms

# Page configuration
st.set_page_config(
    page_title="AI Image Authenticity Checker | Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Premium UI Styling (CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-hero {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E1B4B 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        max-width: 850px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .disclaimer-card {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        color: #78350F;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }

    .result-box-real {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    .result-box-fake {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .badge-label-real {
        font-size: 2.2rem;
        font-weight: 800;
        color: #047857;
        margin-bottom: 0.2rem;
    }
    .badge-label-fake {
        font-size: 2.2rem;
        font-weight: 800;
        color: #B91C1C;
        margin-bottom: 0.2rem;
    }
    .confidence-score {
        font-size: 1.25rem;
        font-weight: 600;
        color: #334155;
    }

    .section-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Hardware and Model Initialization
device = get_device()
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Execution Engine"
model_path = os.path.join("models", "deepfake_detector.pth")
model_exists = os.path.exists(model_path)

# Header Banner
st.markdown("""
    <div class="main-hero">
        <div class="hero-title">🛡️ AI Image Authenticity Checker</div>
        <div class="hero-subtitle">
            Enterprise-grade deepfake and digital image manipulation screening system powered by fine-tuned 
            <strong>EfficientNet-B0</strong> transfer learning and explainable <strong>Grad-CAM</strong> heatmaps.
        </div>
    </div>
""", unsafe_allow_html=True)

# Forensic Disclaimer Banner
st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Forensic Screening Notice:</strong> This application is an AI-assisted manipulation screening tool designed for investigative guidance. 
        Model outputs and Grad-CAM visual heatmaps provide probabilistic assessments and do not constitute absolute digital forensic proof of authenticity.
    </div>
""", unsafe_allow_html=True)

# KPI Metrics Header Bar
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Neural Network</div>
            <div class="kpi-value">EfficientNet-B0</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Compute Device</div>
            <div class="kpi-value">{"GPU CUDA" if torch.cuda.is_available() else "CPU Mode"}</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Status</div>
            <div class="kpi-value" style="color: {'#10B981' if model_exists else '#EF4444'};">
                {"Loaded ✅" if model_exists else "Missing ⚠️"}
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    metrics_path = os.path.join("evaluation_results", "metrics.json")
    acc_text = "N/A"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                m_data = json.load(f)
                acc_text = f"{m_data.get('accuracy', 0)*100:.1f}%"
        except Exception:
            pass
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Benchmark Accuracy</div>
            <div class="kpi-value" style="color: #6366F1;">{acc_text}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar System Control Panel
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("💻 Hardware Info")
st.sidebar.info(f"**Device:** {device_name}")

st.sidebar.subheader("📦 Model Checkpoint")
if model_exists:
    st.sidebar.success("`models/deepfake_detector.pth` ready.")
else:
    st.sidebar.warning("Model checkpoint missing.")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Quick Benchmark Tools")

if st.sidebar.button("🎨 Generate Synthetic Dataset"):
    with st.spinner("Generating sample dataset in dataset/..."):
        generate_synthetic_dataset()
        st.sidebar.success("Synthetic dataset generated!")

if st.sidebar.button("🚀 Quick Train (5 Epochs)"):
    with st.spinner("Training EfficientNet-B0 model..."):
        try:
            train_model(epochs=5, batch_size=16)
            st.sidebar.success("Model trained successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Main Application Multi-Tab Interface
tab_scanner, tab_gallery, tab_metrics, tab_train = st.tabs([
    "🔍 Image Authenticity Scanner",
    "🖼️ Test Sample Gallery",
    "📊 Model Analytics & Metrics",
    "⚙️ Model Training Center"
])

# ==============================================================================
# TAB 1: IMAGE AUTHENTICITY SCANNER & XAI
# ==============================================================================
with tab_scanner:
    st.markdown("### 📤 Upload Image for Deepfake & Manipulation Screening")
    
    col_u1, col_u2 = st.columns([1.2, 1])

    with col_u1:
        uploaded_file = st.file_uploader(
            "Upload image file (Supported formats: JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Select an image to evaluate whether it is authentic or manipulated."
        )

    with col_u2:
        st.markdown("#### ⚙️ Grad-CAM Visualizer Settings")
        colormap_choice = st.selectbox(
            "Heatmap Color Palette",
            options=["JET (Classic Red/Blue)", "VIRIDIS (Perceptual)", "INFERNO (High Contrast)", "PLASMA", "HOT"],
            index=0
        )
        alpha_val = st.slider(
            "Heatmap Opacity / Alpha Overlay",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )

    colormap_dict = {
        "JET (Classic Red/Blue)": cv2.COLORMAP_JET,
        "VIRIDIS (Perceptual)": cv2.COLORMAP_VIRIDIS,
        "INFERNO (High Contrast)": cv2.COLORMAP_INFERNO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "HOT": cv2.COLORMAP_HOT
    }
    selected_colormap = colormap_dict[colormap_choice]

    # Target Image to analyze
    target_image_bytes = None
    target_image_name = "Uploaded Image"

    if uploaded_file is not None:
        target_image_bytes = uploaded_file.read()
        target_image_name = uploaded_file.name
    elif "selected_sample_bytes" in st.session_state:
        target_image_bytes = st.session_state["selected_sample_bytes"]
        target_image_name = st.session_state.get("selected_sample_name", "Gallery Sample")

    if not model_exists:
        st.error(
            "⚠️ **No trained model found!** Please train a model first using the 'Model Training Center' tab "
            "or click 'Quick Train' in the sidebar."
        )

    if target_image_bytes is not None:
        try:
            is_valid, err_msg = validate_image_file(target_image_bytes)
            if not is_valid:
                st.error(f"❌ **Invalid image file:** {err_msg}")
            else:
                pil_img = Image.open(io.BytesIO(target_image_bytes)).convert("RGB")

                st.markdown("---")
                p1, p2 = st.columns([1, 1])

                with p1:
                    st.markdown(f"#### 🖼️ Image Preview: `{target_image_name}`")
                    st.image(pil_img, use_container_width=True)
                    st.caption(
                        f"**Resolution:** {pil_img.width} × {pil_img.height} px | "
                        f"**Color Mode:** {pil_img.mode} | **Format:** RGB"
                    )

                with p2:
                    st.markdown("#### ⚡ Run Analysis")
                    st.write("Click below to pass the image through EfficientNet-B0 and generate Grad-CAM heatmaps.")
                    analyze_btn = st.button("🔍 Run Authenticity Analysis", type="primary", disabled=not model_exists)

                    if analyze_btn or st.session_state.get("last_analyzed_name") == target_image_name:
                        with st.spinner("Analyzing image features & computing Grad-CAM gradients..."):
                            try:
                                # Prediction
                                pred_res = predict_image(target_image_bytes, model_path=model_path, device=device)
                                cls_name = pred_res["class_name"]
                                confidence = pred_res["confidence"]
                                probs = pred_res["probabilities"]

                                # Grad-CAM
                                target_class_idx = 1 if cls_name == "MANIPULATED" else 0
                                explain_res = explain_prediction(
                                    target_image_bytes,
                                    model_path=model_path,
                                    target_class=target_class_idx,
                                    device=device
                                )

                                # Re-overlay with selected colormap & alpha
                                overlay_custom, heatmap_custom = overlay_heatmap(
                                    pil_img,
                                    explain_res["heatmap_raw"],
                                    alpha=alpha_val,
                                    colormap=selected_colormap
                                )

                                st.session_state["last_analyzed_name"] = target_image_name
                                st.session_state["pred_res"] = pred_res
                                st.session_state["overlay_custom"] = overlay_custom
                                st.session_state["heatmap_custom"] = heatmap_custom
                                st.session_state["pil_img"] = pil_img

                            except Exception as e:
                                st.error(f"Analysis failed: {e}")

                # Display Results Card & Visualizers
                if "pred_res" in st.session_state and st.session_state.get("last_analyzed_name") == target_image_name:
                    st.markdown("---")
                    st.markdown("### 📊 Classification & Explainability Output")

                    pred_res = st.session_state["pred_res"]
                    cls_name = pred_res["class_name"]
                    confidence = pred_res["confidence"]
                    probs = pred_res["probabilities"]

                    res_left, res_right = st.columns([1, 1.2])

                    with res_left:
                        if cls_name == "REAL":
                            st.markdown(f"""
                                <div class="result-box-real">
                                    <div style="font-weight:600; color:#065F46;">PREDICTED STATUS</div>
                                    <div class="badge-label-real">✅ REAL IMAGE</div>
                                    <div class="confidence-score">Confidence: <strong>{confidence:.2f}%</strong></div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-box-fake">
                                    <div style="font-weight:600; color:#991B1B;">PREDICTED STATUS</div>
                                    <div class="badge-label-fake">🚨 MANIPULATED / DEEPFAKE</div>
                                    <div class="confidence-score">Confidence: <strong>{confidence:.2f}%</strong></div>
                                </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### Softmax Class Probabilities")
                        st.write(f"**REAL:** `{probs['REAL']:.2f}%`")
                        st.progress(probs['REAL'] / 100.0)
                        st.write(f"**MANIPULATED / DEEPFAKE:** `{probs['MANIPULATED']:.2f}%`")
                        st.progress(probs['MANIPULATED'] / 100.0)

                        st.info(
                            "💡 **Interpretation:** The model evaluates high-level feature activations. "
                            "High confidence scores for MANIPULATED indicate anomalous frequency or boundary artifacts."
                        )

                    with res_right:
                        st.markdown("#### 🧠 Grad-CAM Explainable AI Visualizer")
                        st.caption(
                            "Grad-CAM computes gradients of the target class score with respect to feature maps "
                            "in the final convolutional layer of EfficientNet-B0."
                        )

                        t1, t2, t3 = st.tabs(["🔥 Heatmap Overlay", "🎯 Heatmap Only", "↔️ Side-by-Side"])

                        with t1:
                            st.image(st.session_state["overlay_custom"], caption="Grad-CAM Heatmap Overlay", use_container_width=True)

                        with t2:
                            st.image(st.session_state["heatmap_custom"], caption="Raw Activation Heatmap", use_container_width=True)

                        with t3:
                            s1, s2 = st.columns(2)
                            with s1:
                                st.image(st.session_state["pil_img"], caption="Original Image", use_container_width=True)
                            with s2:
                                st.image(st.session_state["overlay_custom"], caption="Grad-CAM Overlay", use_container_width=True)

        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.info("👆 Upload an image or select a sample from the 'Test Sample Gallery' tab to run authenticity screening.")

# ==============================================================================
# TAB 2: TEST SAMPLE GALLERY
# ==============================================================================
with tab_gallery:
    st.markdown("### 🖼️ Benchmark Test Sample Gallery")
    st.write("Click any sample image below to instantly load it into the Authenticity Scanner.")

    test_dir = os.path.join("dataset", "test")
    if not os.path.exists(test_dir):
        st.warning("No dataset directory found. Click 'Generate Synthetic Dataset' in the sidebar to create test samples.")
    else:
        real_samples = [os.path.join(test_dir, "real", f) for f in os.listdir(os.path.join(test_dir, "real")) if f.endswith(('.jpg', '.png', '.jpeg'))] if os.path.exists(os.path.join(test_dir, "real")) else []
        fake_samples = [os.path.join(test_dir, "fake", f) for f in os.listdir(os.path.join(test_dir, "fake")) if f.endswith(('.jpg', '.png', '.jpeg'))] if os.path.exists(os.path.join(test_dir, "fake")) else []

        st.markdown("#### 🟢 Authentic / Real Samples")
        if real_samples:
            r_cols = st.columns(min(len(real_samples), 5))
            for idx, r_path in enumerate(real_samples[:5]):
                with r_cols[idx]:
                    img_r = Image.open(r_path)
                    st.image(img_r, caption=os.path.basename(r_path), use_container_width=True)
                    if st.button(f"Scan {idx+1}", key=f"btn_real_{idx}"):
                        with open(r_path, "rb") as f:
                            st.session_state["selected_sample_bytes"] = f.read()
                            st.session_state["selected_sample_name"] = os.path.basename(r_path)
                            st.rerun()
        else:
            st.caption("No real test samples found.")

        st.markdown("#### 🚨 Manipulated / Fake Samples")
        if fake_samples:
            f_cols = st.columns(min(len(fake_samples), 5))
            for idx, f_path in enumerate(fake_samples[:5]):
                with f_cols[idx]:
                    img_f = Image.open(f_path)
                    st.image(img_f, caption=os.path.basename(f_path), use_container_width=True)
                    if st.button(f"Scan {idx+1}", key=f"btn_fake_{idx}"):
                        with open(f_path, "rb") as f:
                            st.session_state["selected_sample_bytes"] = f.read()
                            st.session_state["selected_sample_name"] = os.path.basename(f_path)
                            st.rerun()
        else:
            st.caption("No fake test samples found.")

# ==============================================================================
# TAB 3: MODEL ANALYTICS & METRICS
# ==============================================================================
with tab_metrics:
    st.markdown("### 📊 Deepfake Detector Evaluation Analytics")

    metrics_json_path = os.path.join("evaluation_results", "metrics.json")
    cm_path = os.path.join("evaluation_results", "confusion_matrix.png")
    roc_path = os.path.join("evaluation_results", "roc_curve.png")
    history_path = os.path.join("models", "training_history.json")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, "r") as f:
                metrics_data = json.load(f)

            st.markdown("#### 📈 Benchmark Metrics Summary")
            sub_m1, sub_m2, sub_m3, sub_m4 = st.columns(4)
            sub_m1.metric("Accuracy", f"{metrics_data.get('accuracy', 0)*100:.1f}%")
            sub_m2.metric("Precision", f"{metrics_data.get('precision', 0)*100:.1f}%")
            sub_m3.metric("Recall", f"{metrics_data.get('recall', 0)*100:.1f}%")
            sub_m4.metric("F1-Score", f"{metrics_data.get('f1_score', 0)*100:.1f}%")

            st.write(f"**ROC-AUC Score:** `{metrics_data.get('roc_auc', 0):.4f}`")
            st.write(f"**Total Samples Evaluated:** `{metrics_data.get('num_samples', 0)}`")
        else:
            st.info("No saved evaluation metrics found. Click 'Run Evaluation' below.")

        if st.button("🔄 Run Full Model Evaluation"):
            with st.spinner("Evaluating model on test dataset..."):
                try:
                    evaluate_model()
                    st.success("Evaluation complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    with m_col2:
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                hist = json.load(f)
            st.markdown("#### 📉 Epoch Loss & Accuracy Curves")
            st.line_chart({
                "Train Loss": hist.get("train_loss", []),
                "Validation Loss": hist.get("val_loss", [])
            })
            st.line_chart({
                "Train Accuracy": [a * 100 for a in hist.get("train_acc", [])],
                "Validation Accuracy": [a * 100 for a in hist.get("val_acc", [])]
            })

    st.markdown("---")
    st.markdown("#### 🖼️ Diagnostic Plots")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        else:
            st.caption("Confusion matrix plot not generated yet.")

    with p_col2:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve", use_container_width=True)
        else:
            st.caption("ROC curve plot not generated yet.")

# ==============================================================================
# TAB 4: MODEL TRAINING CENTER
# ==============================================================================
with tab_train:
    st.markdown("### ⚙️ Model Hyperparameter Fine-Tuning & Training Center")
    st.write("Configure transfer learning hyperparameters and initiate model training on your dataset.")

    tc1, tc2 = st.columns([1, 1])

    with tc1:
        epochs_input = st.slider("Training Epochs", min_value=1, max_value=30, value=5, step=1)
        batch_size_input = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1)
        lr_input = st.select_slider(
            "Learning Rate (AdamW)",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4
        )

    with tc2:
        st.markdown("#### 📋 Data Split Status")
        tr_count = len(os.listdir("dataset/train/real")) + len(os.listdir("dataset/train/fake")) if os.path.exists("dataset/train/real") else 0
        val_count = len(os.listdir("dataset/validation/real")) + len(os.listdir("dataset/validation/fake")) if os.path.exists("dataset/validation/real") else 0
        te_count = len(os.listdir("dataset/test/real")) + len(os.listdir("dataset/test/fake")) if os.path.exists("dataset/test/real") else 0

        st.write(f"- **Train Set:** `{tr_count}` images")
        st.write(f"- **Validation Set:** `{val_count}` images")
        st.write(f"- **Test Set:** `{te_count}` images")

        if tr_count == 0:
            st.warning("Dataset is currently empty. Click below to generate demo synthetic data.")
            if st.button("Generate Synthetic Data"):
                generate_synthetic_dataset()
                st.rerun()

    start_train_btn = st.button("🚀 Start Model Training", type="primary")

    if start_train_btn:
        with st.spinner(f"Training EfficientNet-B0 for {epochs_input} epochs..."):
            try:
                model, save_p = train_model(
                    epochs=epochs_input,
                    batch_size=batch_size_input,
                    lr=lr_input,
                    device=device
                )
                st.success(f"🎉 Training finished! Best model saved to `{save_p}`.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
