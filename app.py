import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Ensure src modules are importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import EmotionEngine, EMOTION_COLORS
from src.models import EMOTION_DICT

# Page configuration
st.set_page_config(
    page_title="EmoVision AI | Facial Expression Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and modern dark UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .emotion-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine(model_type, detector_type, weights_path):
    return EmotionEngine(
        model_path=weights_path,
        model_type=model_type,
        detector_type=detector_type,
        cascade_path="haarcascade_frontalface_default.xml"
    )

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800;">🎭 EmoVision AI</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 1.1rem;">
            Real-Time Deep Learning Facial Emotion Detection & Multi-Class Confidence Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_choice = st.selectbox(
            "Neural Architecture",
            options=["CNN (4-Block Custom)", "MobileNetV3-Small (Transfer)"],
            index=0
        )
        model_type = "cnn" if "CNN" in model_choice else "mobilenet"

        detector_choice = st.selectbox(
            "Face Detector Backend",
            options=["MediaPipe Face Detection (Recommended)", "OpenCV Haar Cascade"],
            index=0
        )
        detector_type = "mediapipe" if "MediaPipe" in detector_choice else "haar"

        weights_path = st.text_input("Weights File", value="model.h5")
        conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        st.markdown("---")
        st.markdown("### 📊 Emotion Categories")
        emotions_preview = [
            ("Angry", "🔴 #ef4444"),
            ("Disgusted", "🟢 #10b981"),
            ("Fearful", "🟣 #a855f7"),
            ("Happy", "🟡 #eab308"),
            ("Neutral", "⚪ #94a3b8"),
            ("Sad", "🔵 #3b82f6"),
            ("Surprised", "🌸 #ec4899")
        ]
        for name, col in emotions_preview:
            st.markdown(f"- **{name}**")

    # Load inference engine
    with st.spinner("Initializing Emotion Detection Engine..."):
        engine = load_engine(model_type, detector_type, weights_path)

    # Tabs
    tab_image, tab_camera, tab_info = st.tabs(["🖼️ Image Upload", "📸 Camera Live Snapshot", "ℹ️ Architecture & Info"])

    with tab_image:
        st.markdown("### Upload an Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting faces and classifying expressions..."):
                annotated_bgr, results = engine.process_frame(img_bgr, draw_annotations=True)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.image(annotated_rgb, caption=f"Analyzed Image — {len(results)} Face(s) Detected", use_container_width=True)

            with col2:
                if not results:
                    st.warning("No faces detected in the image. Try another photo or adjust lighting.")
                else:
                    st.markdown(f"#### 👤 Detected Faces ({len(results)})")
                    for i, res in enumerate(results):
                        with st.expander(f"Face #{i+1}: **{res['emotion']}** ({res['confidence']*100:.1f}%)", expanded=True):
                            # Crop face thumbnail
                            x, y, w, h = res["bbox"]
                            h_img, w_img = img_np.shape[:2]
                            crop = img_np[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]
                            
                            c_thumb, c_prob = st.columns([0.4, 0.6])
                            with c_thumb:
                                if crop.size > 0:
                                    st.image(crop, width=120, caption=f"Face #{i+1}")
                                st.metric("Top Emotion", res["emotion"], f"{res['confidence']*100:.1f}%")

                            with c_prob:
                                st.markdown("**Probability Breakdown:**")
                                for emo, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
                                    st.write(f"{emo}: `{prob*100:.1f}%`")
                                    st.progress(min(1.0, float(prob)))

    with tab_camera:
        st.markdown("### Live Camera Emotion Snapshot")
        st.info("Take a photo using your webcam to analyze your facial emotion in real time.")
        camera_img = st.camera_input("Capture Face")

        if camera_img is not None:
            pil_cam = Image.open(camera_img).convert("RGB")
            cam_np = np.array(pil_cam)
            cam_bgr = cv2.cvtColor(cam_np, cv2.COLOR_RGB2BGR)

            annotated_cam_bgr, cam_results = engine.process_frame(cam_bgr, draw_annotations=True)
            annotated_cam_rgb = cv2.cvtColor(annotated_cam_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.image(annotated_cam_rgb, caption="Analyzed Webcam Frame", use_container_width=True)

            with col2:
                if not cam_results:
                    st.warning("No face detected in camera capture. Ensure your face is centered and well-lit.")
                else:
                    for i, res in enumerate(cam_results):
                        st.markdown(f"### Dominant: **{res['emotion']}** ({res['confidence']*100:.1f}%)")
                        st.markdown("**Emotion Distribution:**")
                        for emo, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
                            st.write(f"**{emo}**: {prob*100:.1f}%")
                            st.progress(float(prob))

    with tab_info:
        st.markdown("### 🧠 Technical Pipeline & Architecture")
        st.markdown("""
        - **Preprocessing**: Grayscale conversion $\\to$ Face ROI extraction $\\to$ Resize to $48 \\times 48 \\to$ Normalization $[0.0, 1.0]$.
        - **Face Detectors**:
          - **MediaPipe**: BlazeFace-based SSD topology for robust multi-angle and occluded face detection.
          - **Haar Cascade**: Classical Viola-Jones Haar features detector.
        - **Classifier**: 4-Block CNN with Dropout & Batch Normalization or MobileNetV3-Small Transfer Learning.
        - **Classes**: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.
        """)

if __name__ == '__main__':
    main()
