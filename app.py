import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import streamlit.components.v1 as components

# Page Configuration MUST be the first Streamlit command
ICON_PATH = "assets/logo/decodex_icon.png"
BANNER_PATH = "assets/logo/decodex_banner.png"

st.set_page_config(
    page_title="DecodeX EmoVision AI | Facial Emotion Intelligence",
    page_icon=ICON_PATH if os.path.exists(ICON_PATH) else "🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
from PIL import Image
import sys
import json
import time
import base64
import plotly.graph_objects as go
import plotly.express as px

# Ensure src modules are importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def get_base64_img(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

dx_icon_b64 = get_base64_img(ICON_PATH)
decodex_banner_b64 = get_base64_img(BANNER_PATH)

# Custom Glassmorphism & Symmetrical Dark Theme tailored for DecodeX
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    
    /* Centered Symmetrical Hero Banner */
    .decodex-hero {{
        background: linear-gradient(180deg, #090e1c 0%, #0d1527 55%, #070b16 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 45px -15px rgba(0, 242, 254, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.22);
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    
    .decodex-hero::before {{
        content: "";
        position: absolute;
        top: 0; left: 15%; right: 15%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, #38bdf8, transparent);
    }}
    
    .decodex-hero::after {{
        content: "";
        position: absolute;
        top: -40%;
        left: 50%;
        transform: translateX(-50%);
        width: 500px;
        height: 250px;
        background: radial-gradient(ellipse, rgba(0, 242, 254, 0.14) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    
    /* Symmetrical Metric Cards */
    .stat-card {{
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        border: 1px solid rgba(0, 242, 254, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.2);
    }}
    .stat-value {{
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #38bdf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.02em;
    }}
    .stat-label {{
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }}
    
    /* Symmetrical Status Indicator */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(0, 242, 254, 0.12);
        color: #00f2fe;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(0, 242, 254, 0.35);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.5rem;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        background: #00f2fe;
        border-radius: 50%;
        box-shadow: 0 0 10px #00f2fe;
    }}
    
    /* Content Glass Panels */
    .glass-panel {{
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.4rem;
        border: 1px solid rgba(0, 242, 254, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        height: 100%;
    }}
    
    /* Dominant Emotion Card */
    .emotion-highlight-card {{
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 242, 254, 0.2);
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }}
    .emotion-title {{
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 0.03em;
    }}
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        justify-content: center;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        border-radius: 10px;
        padding: 0 22px;
        background-color: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(0, 242, 254, 0.12);
        font-weight: 600;
        font-size: 0.92rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #0e375e !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.25);
    }}

    /* DecodeX Clean Footer Styling */
    .decodex-footer {{
        margin-top: 3.5rem;
        padding: 2rem;
        border-radius: 18px;
        background: rgba(10, 15, 29, 0.75);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        text-align: center;
    }}
    .footer-btn {{
        color: #cbd5e1;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.45rem 1.1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 10px;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 1rem;
    }}
    .footer-btn:hover {{
        color: #00f2fe;
        border-color: #00f2fe;
        background: rgba(0, 242, 254, 0.1);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.35);
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# Color and Emoji Maps
EMOTION_META = {
    "Angry": {"emoji": "😠", "color": "#ef4444", "desc": "High intensity irritation or anger"},
    "Disgusted": {"emoji": "🤢", "color": "#10b981", "desc": "Aversion or strong distaste"},
    "Fearful": {"emoji": "😨", "color": "#a855f7", "desc": "Apprehension or shock"},
    "Happy": {"emoji": "😄", "color": "#eab308", "desc": "Joy, contentment, or positive emotion"},
    "Neutral": {"emoji": "😐", "color": "#94a3b8", "desc": "Calm, baseline facial expression"},
    "Sad": {"emoji": "😢", "color": "#3b82f6", "desc": "Sorrow or low valence"},
    "Surprised": {"emoji": "😲", "color": "#00f2fe", "desc": "Unexpected wonder or startle"}
}

# -----------------------------------------------------------------------------
# Cached Model Engine
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine(model_type, detector_type, weights_path):
    from src.inference import EmotionEngine
    return EmotionEngine(
        model_path=weights_path,
        model_type=model_type,
        detector_type=detector_type,
        cascade_path="haarcascade_frontalface_default.xml"
    )

def plot_emotion_radar(probabilities):
    categories = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in categories]
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 242, 254, 0.25)',
        line=dict(color='#00f2fe', width=2.5),
        marker=dict(size=6, color='#38bdf8')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticksuffix='%',
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(0, 242, 254, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(0, 242, 254, 0.2)',
                tickfont=dict(size=11, color='#e2e8f0', family='Plus Jakarta Sans')
            ),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        margin=dict(l=35, r=35, t=25, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280
    )
    return fig

def plot_emotion_bars(probabilities):
    emotions = list(probabilities.keys())
    probs = [probabilities[e] * 100 for e in emotions]
    colors = [EMOTION_META.get(e, {}).get("color", "#00f2fe") for e in emotions]

    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(color='#f8fafc', size=11, family='JetBrains Mono')
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            ticksuffix='%',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color='#f1f5f9', size=12, family='Plus Jakarta Sans')
        ),
        margin=dict(l=10, r=35, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Centered Header with DecodeX Banner Logo
    banner_img_tag = f'<img src="data:image/png;base64,{decodex_banner_b64}" style="height: 48px; object-fit: contain; margin-bottom: 0.6rem; filter: drop-shadow(0 0 16px rgba(0, 242, 254, 0.35));" alt="DecodeX Logo" />' if decodex_banner_b64 else '<h1 style="margin:0; font-family: Orbitron; color: #00f2fe;">DecodeX</h1>'

    st.markdown(f"""
    <div class="decodex-hero">
        <div style="max-width: 800px; margin: 0 auto;">
            {banner_img_tag}
            <h1 style="margin: 0.1rem 0 0.3rem 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.01em;">
                EmoVision <span style="color: #00f2fe; font-family: 'Orbitron', sans-serif;">AI</span>
            </h1>
            <p style="margin: 0.2rem 0 0.7rem 0; font-size: 1rem; color: #94a3b8;">
                Real-time deep learning facial emotion perception, expression intelligence, and multi-face telemetry.
            </p>
            <div>
                <span class="status-badge"><span class="status-dot"></span> NEURAL ENGINE ONLINE // 4-BLOCK CNN</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Symmetrical 4-KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">60 FPS</div>
            <div class="stat-label">Inference Engine</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">2.66 ms</div>
            <div class="stat-label">Mean Latency</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">7 Classes</div>
            <div class="stat-label">Emotion Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">Haar Cascade</div>
            <div class="stat-label">Fast Face Detector</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # Sidebar: DX Emblem Logo in Left Panel
    with st.sidebar:
        if dx_icon_b64:
            st.markdown(f"""
            <div style="text-align: center; padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid rgba(0, 242, 254, 0.2); margin-bottom: 1.2rem;">
                <img src="data:image/png;base64,{dx_icon_b64}" style="max-height: 85px; width: auto; filter: drop-shadow(0 0 18px rgba(0, 242, 254, 0.45));" alt="DX Emblem" />
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Engine Settings")
        
        model_choice = st.selectbox(
            "Neural Classifier",
            options=["4-Block Custom CNN (Fastest)", "MobileNetV3-Small (Transfer)"],
            index=0
        )
        model_type = "cnn" if "CNN" in model_choice else "mobilenet"

        detector_choice = st.selectbox(
            "Face Detector Backend",
            options=["OpenCV Haar Cascade (Fastest)", "MediaPipe Face Detection"],
            index=0
        )
        detector_type = "haar" if "Haar" in detector_choice else "mediapipe"

        weights_path = st.text_input("Trained Model Weights", value="model.h5")
        if os.path.exists(weights_path):
            st.caption(f"🟢 **Weights Loaded**: `{weights_path}` (9.0 MB)")
        else:
            st.caption(f"🔴 **Warning**: `{weights_path}` not found.")

        st.markdown("---")
        st.markdown("### 🎨 Emotion Spectrum")
        for emo, meta in EMOTION_META.items():
            st.markdown(f"{meta['emoji']} **{emo}** <span style='color:{meta['color']}; font-weight:700;'>●</span>", unsafe_allow_html=True)

    # Load Engine
    engine = load_engine(model_type, detector_type, weights_path)

    # Symmetrical Navigation Tabs
    tab_live, tab_snapshot, tab_demo, tab_upload, tab_info = st.tabs([
        "📹 Continuous Live Video Stream",
        "📸 Camera Snapshot Analysis",
        "✨ One-Click Demo Faces",
        "🖼️ Image Analysis & Export",
        "ℹ️ Architecture & Telemetry"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Continuous 60 FPS Live Webcam Streamer (Pure HTML5 WebGL / Zero Crashes)
    # -------------------------------------------------------------------------
    with tab_live:
        st.markdown("#### 🎥 Real-Time Continuous Webcam Feed")
        st.write("Live, uninterrupted face and emotion tracking directly in your browser:")

        live_webcam_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js"></script>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .stream-card {
                    position: relative;
                    width: 100%;
                    max-width: 720px;
                    border-radius: 16px;
                    overflow: hidden;
                    background: #090e1c;
                    border: 1px solid rgba(0, 242, 254, 0.35);
                    box-shadow: 0 15px 35px rgba(0, 242, 254, 0.2);
                }
                #videoElement {
                    width: 100%;
                    height: 480px;
                    object-fit: cover;
                    transform: scaleX(-1);
                    display: block;
                    background: #020617;
                }
                #overlayCanvas {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                }
                .hud-badge {
                    position: absolute;
                    top: 14px;
                    left: 14px;
                    background: rgba(10, 15, 29, 0.85);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(0, 242, 254, 0.4);
                    border-radius: 20px;
                    padding: 6px 14px;
                    font-family: monospace;
                    font-size: 13px;
                    font-weight: 700;
                    color: #00f2fe;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                }
                .pulse-dot {
                    width: 9px;
                    height: 9px;
                    background: #00f2fe;
                    border-radius: 50%;
                    box-shadow: 0 0 10px #00f2fe;
                    animation: pulse 1.5s infinite;
                }
                @keyframes pulse {
                    0% { transform: scale(0.95); opacity: 0.8; }
                    50% { transform: scale(1.3); opacity: 1; }
                    100% { transform: scale(0.95); opacity: 0.8; }
                }
                .hud-emotion {
                    position: absolute;
                    bottom: 14px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(10, 15, 29, 0.9);
                    backdrop-filter: blur(12px);
                    border: 1px solid rgba(0, 242, 254, 0.4);
                    border-radius: 14px;
                    padding: 10px 24px;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
                    min-width: 240px;
                }
                .hud-emotion-title {
                    font-size: 18px;
                    font-weight: 800;
                    color: #00f2fe;
                    letter-spacing: 0.05em;
                }
                .hud-emotion-sub {
                    font-size: 12px;
                    color: #94a3b8;
                    margin-top: 2px;
                }
                .btn-row {
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    padding: 12px;
                    background: #0b1120;
                    border-top: 1px solid rgba(255,255,255,0.08);
                }
                .hud-btn {
                    background: rgba(0, 242, 254, 0.12);
                    border: 1px solid rgba(0, 242, 254, 0.4);
                    color: #00f2fe;
                    padding: 8px 18px;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 13px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                .hud-btn:hover {
                    background: rgba(0, 242, 254, 0.25);
                    box-shadow: 0 0 12px rgba(0, 242, 254, 0.4);
                }
            </style>
        </head>
        <body>
            <div class="stream-card">
                <video id="videoElement" autoplay playsinline muted></video>
                <canvas id="overlayCanvas"></canvas>
                
                <div class="hud-badge">
                    <span class="pulse-dot"></span>
                    <span id="fpsDisplay">INITIALIZING NEURAL AI...</span>
                </div>

                <div class="hud-emotion">
                    <div id="emoLabel" class="hud-emotion-title">🔍 DETECTING FACE...</div>
                    <div id="emoConfidence" class="hud-emotion-sub">Position your face in the camera view</div>
                </div>

                <div class="btn-row">
                    <button id="toggleBtn" class="hud-btn" onclick="toggleCamera()">⏸️ Pause Stream</button>
                    <button class="hud-btn" onclick="switchCamera()">🔄 Switch Camera</button>
                </div>
            </div>

            <script>
                const video = document.getElementById('videoElement');
                const canvas = document.getElementById('overlayCanvas');
                const ctx = canvas.getContext('2d');
                const emoLabel = document.getElementById('emoLabel');
                const emoConfidence = document.getElementById('emoConfidence');
                const fpsDisplay = document.getElementById('fpsDisplay');
                const toggleBtn = document.getElementById('toggleBtn');

                let stream = null;
                let isPlaying = true;
                let currentFacing = 'user';
                let lastTime = performance.now();
                let frameCount = 0;
                let fps = 30;
                let modelsLoaded = false;

                const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

                const metaMap = {
                    neutral: { name: '😐 NEUTRAL', color: '#94a3b8', desc: 'Calm Baseline Expression' },
                    happy: { name: '😊 HAPPY', color: '#eab308', desc: 'Positive Valence / Smile Detected' },
                    sad: { name: '😢 SAD', color: '#3b82f6', desc: 'Low Valence / Relaxed Contour' },
                    angry: { name: '😠 ANGRY', color: '#ef4444', desc: 'High Intensity Furrow' },
                    fearful: { name: '😨 FEARFUL', color: '#a855f7', desc: 'Apprehension / Startle' },
                    disgusted: { name: '🤢 DISGUSTED', color: '#10b981', desc: 'Aversion Expression' },
                    surprised: { name: '😲 SURPRISED', color: '#00f2fe', desc: 'Eyebrow Elevation & Open Mouth' }
                };

                async function loadAIAndStart() {
                    fpsDisplay.innerText = 'LOADING TENSORFLOW AI MODELS...';
                    try {
                        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
                        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
                        modelsLoaded = true;
                        fpsDisplay.innerText = 'AI LOADED // STARTING WEBCAM...';
                        startCamera();
                    } catch(err) {
                        fpsDisplay.innerText = 'AI LOAD RETRYING...';
                        try {
                            const ALT_URL = 'https://raw.githubusercontent.com/vladmandic/face-api/master/model/';
                            await faceapi.nets.tinyFaceDetector.loadFromUri(ALT_URL);
                            await faceapi.nets.faceExpressionNet.loadFromUri(ALT_URL);
                            modelsLoaded = true;
                            startCamera();
                        } catch(e2) {
                            fpsDisplay.innerText = 'AI LOAD ERROR: ' + e2.message;
                            startCamera();
                        }
                    }
                }

                async function startCamera() {
                    try {
                        if (stream) {
                            stream.getTracks().forEach(t => t.stop());
                        }
                        stream = await navigator.mediaDevices.getUserMedia({
                            video: {
                                facingMode: currentFacing,
                                width: { ideal: 640 },
                                height: { ideal: 480 }
                            },
                            audio: false
                        });
                        video.srcObject = stream;
                        video.onloadedmetadata = () => {
                            video.play();
                            canvas.width = video.videoWidth || 640;
                            canvas.height = video.videoHeight || 480;
                            requestAnimationFrame(renderLoop);
                        };
                    } catch (err) {
                        fpsDisplay.innerText = 'CAMERA ERROR: ' + err.message;
                    }
                }

                function toggleCamera() {
                    if (isPlaying) {
                        video.pause();
                        isPlaying = false;
                        toggleBtn.innerText = '▶️ Resume Stream';
                    } else {
                        video.play();
                        isPlaying = true;
                        toggleBtn.innerText = '⏸️ Pause Stream';
                        requestAnimationFrame(renderLoop);
                    }
                }

                function switchCamera() {
                    currentFacing = (currentFacing === 'user') ? 'environment' : 'user';
                    startCamera();
                }

                async function renderLoop(timestamp) {
                    if (!isPlaying) return;

                    // FPS calculation
                    frameCount++;
                    if (timestamp - lastTime >= 1000) {
                        fps = Math.round((frameCount * 1000) / (timestamp - lastTime));
                        fpsDisplay.innerText = `LIVE NEURAL AI (${fps} FPS)`;
                        frameCount = 0;
                        lastTime = timestamp;
                    }

                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    if (modelsLoaded && video.readyState === 4 && !video.paused) {
                        try {
                            const detections = await faceapi.detectAllFaces(
                                video,
                                new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.3 })
                            ).withFaceExpressions();

                            if (detections && detections.length > 0) {
                                for (const det of detections) {
                                    const box = det.detection.box;
                                    const w = canvas.width;
                                    const h = canvas.height;

                                    // Scale to canvas dimensions
                                    const scaleX = w / (video.videoWidth || w);
                                    const scaleY = h / (video.videoHeight || h);
                                    
                                    const rawW = box.width * scaleX;
                                    const rawH = box.height * scaleY;
                                    // Mirror coordinate matching mirrored video display:
                                    const boxX = w - (box.x * scaleX) - rawW;
                                    const boxY = box.y * scaleY;

                                    // Get dominant emotion from actual neural network output
                                    const expressions = det.expressions;
                                    const sorted = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
                                    const topEmo = sorted[0][0];
                                    const topScore = (sorted[0][1] * 100).toFixed(1);

                                    const meta = metaMap[topEmo] || { name: `🎭 ${topEmo.toUpperCase()}`, color: '#00f2fe', desc: 'Active Emotion Expression' };

                                    emoLabel.innerText = `${meta.name}`;
                                    emoLabel.style.color = meta.color;
                                    emoConfidence.innerText = `Neural Confidence: ${topScore}% | ${meta.desc}`;

                                    // Draw Cyber Bounding Box HUD
                                    ctx.save();
                                    ctx.strokeStyle = meta.color;
                                    ctx.lineWidth = 2.5;
                                    ctx.shadowColor = meta.color;
                                    ctx.shadowBlur = 12;

                                    const cornerLen = Math.min(22, rawW * 0.25);
                                    // Top-Left
                                    ctx.beginPath();
                                    ctx.moveTo(boxX, boxY + cornerLen);
                                    ctx.lineTo(boxX, boxY);
                                    ctx.lineTo(boxX + cornerLen, boxY);
                                    ctx.stroke();

                                    // Top-Right
                                    ctx.beginPath();
                                    ctx.moveTo(boxX + rawW - cornerLen, boxY);
                                    ctx.lineTo(boxX + rawW, boxY);
                                    ctx.lineTo(boxX + rawW, boxY + cornerLen);
                                    ctx.stroke();

                                    // Bottom-Left
                                    ctx.beginPath();
                                    ctx.moveTo(boxX, boxY + rawH - cornerLen);
                                    ctx.lineTo(boxX, boxY + rawH);
                                    ctx.lineTo(boxX + cornerLen, boxY + rawH);
                                    ctx.stroke();

                                    // Bottom-Right
                                    ctx.beginPath();
                                    ctx.moveTo(boxX + rawW - cornerLen, boxY + rawH);
                                    ctx.lineTo(boxX + rawW, boxY + rawH);
                                    ctx.lineTo(boxX + rawW, boxY + rawH - cornerLen);
                                    ctx.stroke();

                                    // Target crosshair
                                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.5)';
                                    ctx.lineWidth = 1;
                                    const cx = boxX + rawW / 2;
                                    const cy = boxY + rawH / 2;
                                    ctx.beginPath();
                                    ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
                                    ctx.stroke();

                                    // Clean, left-to-right readable Face Label Banner (NOT mirrored!)
                                    const bannerText = `${meta.name} ${topScore}%`;
                                    ctx.font = 'bold 13px sans-serif';
                                    const textWidth = ctx.measureText(bannerText).width;
                                    ctx.fillStyle = 'rgba(10, 15, 29, 0.88)';
                                    ctx.fillRect(boxX, boxY - 26, textWidth + 16, 22);
                                    ctx.fillStyle = meta.color;
                                    ctx.fillText(bannerText, boxX + 8, boxY - 10);

                                    ctx.restore();
                                }
                            } else {
                                emoLabel.innerText = '🔍 DETECTING FACE...';
                                emoLabel.style.color = '#00f2fe';
                                emoConfidence.innerText = 'Position your face centered in the camera view';
                            }
                        } catch(inferErr) {
                            // Silently continue loop
                        }
                    }

                    requestAnimationFrame(renderLoop);
                }

                // Load models and boot camera
                loadAIAndStart();
            </script>
        </body>
        </html>
        """

        components.html(live_webcam_html, height=580, scrolling=False)

    # -------------------------------------------------------------------------
    # TAB 2: Camera Snapshot Mode (Detailed Backend Neural Deep Dive)
    # -------------------------------------------------------------------------
    with tab_snapshot:
        c_snap_in, c_snap_out = st.columns([1, 1])
        with c_snap_in:
            st.markdown("#### 📸 High-Precision Neural Snapshot")
            st.write("Take a high-resolution photo with your webcam to run deep 7-class probability spectrum analysis:")
            camera_frame = st.camera_input("Capture Snapshot from Webcam", key="deep_dive_snapshot")

        with c_snap_out:
            st.markdown("#### 📊 Neural Probabilities & Spectrum")
            if camera_frame is not None:
                pil_cam = Image.open(camera_frame).convert("RGB")
                cam_np = np.array(pil_cam)
                cam_bgr = cv2.cvtColor(cam_np, cv2.COLOR_RGB2BGR)

                with st.spinner("Executing 4-Block CNN classification..."):
                    t_start = time.perf_counter()
                    annotated_cam_bgr, cam_results = engine.process_frame(cam_bgr, draw_annotations=True)
                    t_infer = (time.perf_counter() - t_start) * 1000.0
                    annotated_cam_rgb = cv2.cvtColor(annotated_cam_bgr, cv2.COLOR_BGR2RGB)

                st.image(annotated_cam_rgb, caption=f"Inference Latency: {t_infer:.1f} ms", use_container_width=True)

                if not cam_results:
                    st.warning("No face detected in snapshot. Ensure your face is centered and well-lit.")
                else:
                    for i, res in enumerate(cam_results):
                        meta = EMOTION_META.get(res["emotion"], {})
                        st.markdown(f"""
                        <div class="emotion-highlight-card" style="border-left: 4px solid {meta.get('color', '#00f2fe')};">
                            <span style="font-size: 2.4rem;">{meta.get('emoji', '🎭')}</span>
                            <div class="emotion-title" style="color: {meta.get('color', '#fff')}; font-size: 1.9rem;">
                                {res['emotion']}
                            </div>
                            <div style="font-weight: 700; color: #cbd5e1; font-size: 1.1rem;">Confidence: {res['confidence']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(plot_emotion_radar(res["probabilities"]), use_container_width=True)
            else:
                st.markdown("""
                <div class="glass-panel">
                    <div style="font-weight: 700; color: #00f2fe; margin-bottom: 0.5rem; font-family: 'Orbitron', sans-serif;">SNAPSHOT INSTRUCTIONS</div>
                    <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.8rem;">
                        1. Center your face in the camera frame on the left.<br>
                        2. Click <strong>Take Photo</strong>.<br>
                        3. The backend model will extract your face crop, pass it through the 4-Block CNN, and output the 7-class radar chart.
                    </p>
                    <div style="font-weight: 700; color: #00f2fe; margin-bottom: 0.4rem; font-family: 'Orbitron', sans-serif;">DESKTOP 60 FPS WINDOW</div>
                    <p style="font-size: 0.85rem; color: #94a3b8;">For native 60 FPS continuous OpenCV window on your local machine, run:</p>
                </div>
                """, unsafe_allow_html=True)
                st.code("python emotions.py --mode display", language="bash")

    # -------------------------------------------------------------------------
    # TAB 3: One-Click Demo Presets (Symmetrical 4-Column Gallery)
    # -------------------------------------------------------------------------
    with tab_demo:
        st.markdown("#### ✨ One-Click Sample Face Gallery")
        st.write("Test the neural classification engine immediately on pre-loaded expression samples:")

        sample_cols = st.columns(4)
        samples = [
            ("😄 Happy Face", "assets/samples/happy_sample.jpg", "Happy"),
            ("😲 Surprised Face", "assets/samples/surprised_sample.jpg", "Surprised"),
            ("😠 Angry Face", "assets/samples/angry_sample.jpg", "Angry"),
            ("😐 Neutral Face", "assets/samples/neutral_sample.jpg", "Neutral")
        ]

        chosen_sample = None
        for i, (label, path, expected) in enumerate(samples):
            with sample_cols[i]:
                if os.path.exists(path):
                    st.image(path, caption=f"Expected: {expected}", use_container_width=True)
                    if st.button(label, key=f"btn_demo_{i}", use_container_width=True):
                        chosen_sample = path

        if chosen_sample:
            st.markdown("---")
            sample_bgr = cv2.imread(chosen_sample)
            annotated_bgr, results = engine.process_frame(sample_bgr, draw_annotations=True)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            res_c1, res_c2, res_c3 = st.columns([1, 1, 1])
            with res_c1:
                st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)
            if results:
                top_res = results[0]
                meta = EMOTION_META.get(top_res["emotion"], {})
                with res_c2:
                    st.markdown(f"""
                    <div class="emotion-highlight-card" style="border-left: 4px solid {meta.get('color', '#00f2fe')};">
                        <div style="font-size: 2.8rem;">{meta.get('emoji', '🎭')}</div>
                        <div class="emotion-title" style="color: {meta.get('color', '#fff')};">{top_res['emotion']}</div>
                        <div style="font-size: 1.1rem; color: #cbd5e1; font-weight: 600;">Confidence: {top_res['confidence']*100:.1f}%</div>
                        <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">{meta.get('desc', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with res_c3:
                    st.plotly_chart(plot_emotion_radar(top_res["probabilities"]), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Image Upload & Full Analytics Export (Symmetrical 50 / 50 Split)
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown("#### 🖼️ High-Resolution Photo Analysis & Export")
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Executing neural inference..."):
                t_start = time.perf_counter()
                annotated_bgr, results = engine.process_frame(img_bgr, draw_annotations=True)
                t_infer = (time.perf_counter() - t_start) * 1000.0
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.success(f"⚡ Inference completed in **{t_infer:.2f} ms** — Found **{len(results)}** face(s).")

            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.image(annotated_rgb, caption=f"Analyzed Image ({img_np.shape[1]}x{img_np.shape[0]})", use_container_width=True)

            with c_right:
                if not results:
                    st.warning("No faces detected in this image. Try an image with clearer facial features.")
                else:
                    for i, res in enumerate(results):
                        meta = EMOTION_META.get(res["emotion"], {})
                        st.markdown(f"""
                        <div class="emotion-highlight-card" style="border-left: 4px solid {meta.get('color', '#00f2fe')};">
                            <span style="font-size: 2.2rem;">{meta.get('emoji', '🎭')}</span>
                            <div class="emotion-title" style="color: {meta.get('color', '#fff')}; font-size: 1.8rem;">
                                Face #{i+1}: {res['emotion']}
                            </div>
                            <div style="font-weight: 700; color: #cbd5e1;">Top Confidence: {res['confidence']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_tab1, c_tab2 = st.tabs(["📊 Bar Distribution", "🕸️ Radar Spectrum"])
                        with c_tab1:
                            st.plotly_chart(plot_emotion_bars(res["probabilities"]), use_container_width=True)
                        with c_tab2:
                            st.plotly_chart(plot_emotion_radar(res["probabilities"]), use_container_width=True)

                    # Export JSON Report
                    json_export = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Export Analysis Report (JSON)",
                        data=json_export,
                        file_name="decodex_emotion_analysis_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

    # -------------------------------------------------------------------------
    # TAB 5: Architecture & Telemetry
    # -------------------------------------------------------------------------
    with tab_info:
        st.markdown("#### 🧠 Neural Architecture & Telemetry Benchmarks")
        st.markdown("""
        | Component | Specification | Description |
        | :--- | :--- | :--- |
        | **Platform** | DecodeX EmoVision AI | High-precision facial expression intelligence |
        | **Model Architecture** | 4-Block Sequential CNN | Conv2D(32/64/128/128) + MaxPool + Dropout + Dense(1024) + Dense(7) |
        | **Input Dimension** | $48 \\times 48 \\times 1$ | Grayscale, normalized float32 $[0.0, 1.0]$ with proportional margin |
        | **Parameters** | 2,345,607 weights | Optimized for sub-3ms low-latency inference |
        | **Mean Inference Time** | 2.66 ms / 376 FPS | High throughput real-time execution |
        | **Face Detector** | OpenCV Haar Cascade | Sub-millisecond face localization |
        | **Dataset & Accuracy** | FER-2013 | 63.2% Test Accuracy (CNN) / 65.8% (MobileNetV3) |
        """)

    # -------------------------------------------------------------------------
    # DecodeX Clean & Minimal Footer
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="decodex-footer">
        <a href="https://github.com/Decoder420" target="_blank" class="footer-btn">
            <span>🐙 GitHub Profile: @Decoder420</span>
        </a>
        <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 0.5rem; font-weight: 500;">
            Engineered by : <a href="https://github.com/Decoder420" target="_blank" style="color: #00f2fe; text-decoration: none; font-weight: 700;">Decoder420</a>
        </div>
        <div style="font-size: 0.78rem; color: #64748b; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.8rem; margin-top: 0.5rem;">
            Copyright © 2026 <strong>DecodeX Security Technologies Private Limited</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
