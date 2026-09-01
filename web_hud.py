import base64
import time
import io
import cv2
import numpy as np
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import EmotionEngine
from src.hand_tracker import HandTracker
from src.gesture_engine import GestureEngine
from src.hud_renderer import HUDRenderer

app = FastAPI(title="EmoVision Holo-HUD Web App")

# Initialize AI Engines
engine = EmotionEngine(model_path="model.h5", model_type="cnn", detector_type="haar")
hand_tracker = HandTracker(max_num_hands=2, smooth_alpha=0.65)
gesture_engine = GestureEngine()
hud_renderer = HUDRenderer()

class FramePayload(BaseModel):
    image: str # Base64 encoded JPEG/PNG frame
    mode: str = "holo" # "holo" or "standard"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmoVision AI PRO // Iron Man Holographic HUD</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #070913;
            --panel-bg: rgba(15, 23, 42, 0.75);
            --cyan: #00f2fe;
            --magenta: #ff0844;
            --neon-blue: #4facfe;
            --green: #10b981;
            --purple: #8a2be2;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(255, 8, 68, 0.08) 0%, transparent 40%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Header */
        header {
            padding: 1.2rem 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 242, 254, 0.2);
            background: rgba(7, 9, 19, 0.85);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--cyan), #fff, var(--magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.05em;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            border: 1px solid rgba(16, 185, 129, 0.4);
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.3); }
        }

        /* Main Container */
        .container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 1.8rem 2rem;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 1.8rem;
            flex: 1;
            width: 100%;
        }

        /* Viewport Canvas Card */
        .viewport-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(0, 242, 254, 0.3);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 242, 254, 0.05);
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }

        .viewport-card::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        }

        .canvas-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #hudCanvas, #outputImage {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        #rawVideo {
            display: none;
        }

        .hud-placeholder {
            position: absolute;
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
        }

        .controls-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.2rem;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .btn-start {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #000;
            font-weight: 800;
            border: none;
            padding: 0.8rem 1.8rem;
            border-radius: 12px;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
            transition: all 0.2s ease;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(0, 242, 254, 0.7);
        }

        .btn-stop {
            background: linear-gradient(135deg, #ff0844 0%, #ff4e50 100%);
            color: #fff;
            box-shadow: 0 0 20px rgba(255, 8, 68, 0.4);
        }

        .mode-toggles {
            display: flex;
            background: rgba(30, 41, 59, 0.7);
            border-radius: 10px;
            padding: 4px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .mode-btn {
            background: transparent;
            color: #94a3b8;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .mode-btn.active {
            background: var(--cyan);
            color: #000;
            font-weight: 800;
        }

        /* Sidebar Panels */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }

        .panel-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.4rem;
        }

        .panel-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            color: var(--cyan);
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Dominant Emotion Highlight */
        .dominant-box {
            text-align: center;
            padding: 1.2rem;
            border-radius: 12px;
            background: rgba(0, 242, 254, 0.05);
            border: 1px solid rgba(0, 242, 254, 0.2);
            margin-bottom: 1rem;
        }

        .dominant-emoji {
            font-size: 3.2rem;
            margin-bottom: 0.2rem;
        }

        .dominant-name {
            font-size: 1.6rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: -0.02em;
        }

        .dominant-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: var(--cyan);
            margin-top: 0.2rem;
        }

        /* Emotion Bars */
        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.65rem;
            font-size: 0.85rem;
        }

        .bar-label {
            width: 80px;
            color: #94a3b8;
            font-weight: 600;
        }

        .bar-track {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
            overflow: hidden;
            margin: 0 0.8rem;
        }

        .bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--cyan), var(--magenta));
            border-radius: 4px;
            transition: width 0.15s ease;
        }

        .bar-val {
            width: 45px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #cbd5e1;
        }

        /* Gesture Badges */
        .gesture-item {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.6rem 0.8rem;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.85rem;
        }

        .gesture-item.active {
            border-color: var(--magenta);
            background: rgba(255, 8, 68, 0.15);
            box-shadow: 0 0 10px rgba(255, 8, 68, 0.3);
        }

        .gesture-icon {
            font-size: 1.2rem;
        }

        .telemetry-row {
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.4rem;
        }

        .telemetry-val {
            color: var(--cyan);
            font-weight: 700;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>EMOVISION HOLO-HUD</h1>
            <span class="status-badge" id="statusBadge"><span class="status-dot"></span> <span id="statusText">READY</span></span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #94a3b8;">
            LATENCY: <span id="latencyDisplay" style="color: var(--cyan); font-weight:700;">0.0 ms</span> // FPS: <span id="fpsDisplay" style="color: var(--cyan); font-weight:700;">0.0</span>
        </div>
    </header>

    <main class="container">
        <!-- Viewport Area -->
        <section class="viewport-card">
            <div class="canvas-wrapper">
                <video id="rawVideo" playsinline autoplay muted></video>
                <img id="outputImage" alt="Holo-HUD Stream" style="display: none;">
                <div class="hud-placeholder" id="placeholder">
                    <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">🦾</p>
                    <p style="font-weight: 700; color: #fff;">Camera is currently offline</p>
                    <p style="font-size: 0.9rem; margin-top: 0.3rem;">Click <strong>LAUNCH CAMERA STREAM</strong> below to start real-time Holo-HUD</p>
                </div>
            </div>

            <div class="controls-bar">
                <button class="btn-start" id="btnToggleCam" onclick="toggleCamera()">
                    <span>▶</span> <span id="btnCamText">LAUNCH CAMERA STREAM</span>
                </button>

                <div class="mode-toggles">
                    <button class="mode-btn active" id="btnHolo" onclick="setMode('holo')">Hologram HUD</button>
                    <button class="mode-btn" id="btnStandard" onclick="setMode('standard')">Standard Boxes</button>
                </div>
            </div>
        </section>

        <!-- Right Telemetry Sidebar -->
        <aside class="sidebar">
            <!-- Dominant Card -->
            <div class="panel-card">
                <div class="panel-title">DOMINANT EMOTION</div>
                <div class="dominant-box">
                    <div class="dominant-emoji" id="domEmoji">🎭</div>
                    <div class="dominant-name" id="domName">STANDBY</div>
                    <div class="dominant-conf" id="domConf">Confidence: 0.0%</div>
                </div>

                <!-- Probability Spectrum Bars -->
                <div class="panel-title" style="margin-top: 1rem;">SPECTRUM DISTRIBUTION</div>
                <div id="barsContainer">
                    <!-- Populated dynamically -->
                </div>
            </div>

            <!-- Hand Gestures Active State -->
            <div class="panel-card">
                <div class="panel-title">GESTURE TELEMETRY</div>
                <div class="gesture-item" id="gPinch">
                    <span class="gesture-icon">🤏</span>
                    <div>
                        <div style="font-weight: 700;">PINCH CLICK</div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Thumb & index touch</div>
                    </div>
                </div>
                <div class="gesture-item" id="gPalm">
                    <span class="gesture-icon">🖐️</span>
                    <div>
                        <div style="font-weight: 700;">OPEN PALM</div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Reset active cards</div>
                    </div>
                </div>
                <div class="gesture-item" id="gSwipe">
                    <span class="gesture-icon">↔️</span>
                    <div>
                        <div style="font-weight: 700;">HORIZONTAL SWIPE</div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Cycle interface mode</div>
                    </div>
                </div>

                <div style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem;">
                    <div class="telemetry-row">
                        <span>CONTROL MODE:</span>
                        <span class="telemetry-val" id="controlModeVal">MOUSE FALLBACK</span>
                    </div>
                    <div class="telemetry-row">
                        <span>FACE DETECTED:</span>
                        <span class="telemetry-val" id="faceDetectedVal">NO</span>
                    </div>
                </div>
            </div>
        </aside>
    </main>

    <canvas id="captureCanvas" width="640" height="480" style="display: none;"></canvas>

    <script>
        const EMOTION_LIST = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"];
        const EMOTION_EMOJIS = {
            "Angry": "😠", "Disgusted": "🤢", "Fearful": "😨",
            "Happy": "😄", "Neutral": "😐", "Sad": "😢", "Surprised": "😲"
        };

        let isStreaming = false;
        let streamMode = "holo";
        let videoEl = document.getElementById("rawVideo");
        let outputImg = document.getElementById("outputImage");
        let captureCanvas = document.getElementById("captureCanvas");
        let capCtx = captureCanvas.getContext("2d");
        let placeholder = document.getElementById("placeholder");
        let btnToggle = document.getElementById("btnToggleCam");
        let btnCamText = document.getElementById("btnCamText");

        let lastTime = performance.now();
        let frameCount = 0;
        let isProcessing = false;

        // Initialize bars
        const barsContainer = document.getElementById("barsContainer");
        EMOTION_LIST.forEach(emo => {
            barsContainer.innerHTML += `
                <div class="bar-row">
                    <div class="bar-label">${emo}</div>
                    <div class="bar-track"><div class="bar-fill" id="bar_${emo}"></div></div>
                    <div class="bar-val" id="val_${emo}">0%</div>
                </div>
            `;
        });

        function setMode(mode) {
            streamMode = mode;
            document.getElementById("btnHolo").classList.toggle("active", mode === "holo");
            document.getElementById("btnStandard").classList.toggle("active", mode === "standard");
        }

        async function toggleCamera() {
            if (isStreaming) {
                stopCamera();
            } else {
                await startCamera();
            }
        }

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
                    audio: false
                });
                videoEl.srcObject = stream;
                await videoEl.play();

                isStreaming = true;
                placeholder.style.display = "none";
                outputImg.style.display = "block";
                btnToggle.classList.add("btn-stop");
                btnCamText.innerText = "STOP CAMERA STREAM";
                document.getElementById("statusText").innerText = "LIVE STREAMING";

                requestAnimationFrame(processLoop);
            } catch (err) {
                alert("Camera Access Error: " + err.message);
            }
        }

        function stopCamera() {
            isStreaming = false;
            if (videoEl.srcObject) {
                videoEl.srcObject.getTracks().forEach(track => track.stop());
                videoEl.srcObject = null;
            }
            placeholder.style.display = "block";
            outputImg.style.display = "none";
            btnToggle.classList.remove("btn-stop");
            btnCamText.innerText = "LAUNCH CAMERA STREAM";
            document.getElementById("statusText").innerText = "OFFLINE";
        }

        async function processLoop() {
            if (!isStreaming) return;

            if (!isProcessing && videoEl.videoWidth > 0) {
                isProcessing = true;
                const t0 = performance.now();

                // Capture frame to canvas
                captureCanvas.width = 640;
                captureCanvas.height = 480;
                capCtx.drawImage(videoEl, 0, 0, 640, 480);

                const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);
                const base64Data = dataUrl.split(",")[1];

                try {
                    const resp = await fetch("/api/predict_hud", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: base64Data, mode: streamMode })
                    });

                    if (resp.ok) {
                        const data = await resp.json();
                        outputImg.src = "data:image/jpeg;base64," + data.image;

                        // Update Latency & FPS
                        const dt = performance.now() - t0;
                        document.getElementById("latencyDisplay").innerText = dt.toFixed(1) + " ms";

                        frameCount++;
                        const now = performance.now();
                        if (now - lastTime >= 1000) {
                            document.getElementById("fpsDisplay").innerText = (frameCount * 1000 / (now - lastTime)).toFixed(1);
                            frameCount = 0;
                            lastTime = now;
                        }

                        // Update Telemetry
                        updateTelemetry(data);
                    }
                } catch (e) {
                    console.error("Frame error:", e);
                } finally {
                    isProcessing = false;
                }
            }

            requestAnimationFrame(processLoop);
        }

        function updateTelemetry(data) {
            // Dominant
            if (data.dominant_emotion) {
                document.getElementById("domEmoji").innerText = EMOTION_EMOJIS[data.dominant_emotion] || "🎭";
                document.getElementById("domName").innerText = data.dominant_emotion.toUpperCase();
                document.getElementById("domConf").innerText = `Confidence: ${(data.dominant_confidence * 100).toFixed(1)}%`;
                document.getElementById("faceDetectedVal").innerText = "YES (1 Face)";
            } else {
                document.getElementById("faceDetectedVal").innerText = "SEARCHING...";
            }

            // Probabilities
            if (data.probabilities) {
                for (const [emo, prob] of Object.entries(data.probabilities)) {
                    const fill = document.getElementById("bar_" + emo);
                    const val = document.getElementById("val_" + emo);
                    if (fill && val) {
                        fill.style.width = (prob * 100).toFixed(0) + "%";
                        val.innerText = (prob * 100).toFixed(0) + "%";
                    }
                }
            }

            // Gestures
            const gestures = data.gestures || [];
            document.getElementById("gPinch").classList.toggle("active", gestures.includes("PINCH_CLICK") || gestures.includes("PINCH_HOLD") || data.pinch_active);
            document.getElementById("gPalm").classList.toggle("active", gestures.includes("OPEN_PALM"));
            document.getElementById("gSwipe").classList.toggle("active", gestures.includes("SWIPE_RIGHT") || gestures.includes("SWIPE_LEFT"));

            if (data.fallback_active) {
                document.getElementById("controlModeVal").innerText = "MOUSE FALLBACK";
                document.getElementById("controlModeVal").style.color = "#f59e0b";
            } else {
                document.getElementById("controlModeVal").innerText = "GESTURE ACTIVE";
                document.getElementById("controlModeVal").style.color = "#10b981";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.post("/api/predict_hud")
async def predict_hud(payload: FramePayload):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(payload.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        # Mirror image for natural user interaction
        frame_bgr = cv2.flip(frame_bgr, 1)

        # 1. Run Emotion Detection
        annotated_frame, face_results = engine.process_frame(frame_bgr, draw_annotations=False)

        # 2. Run Hand Tracking
        face_bboxes = [f["bbox"] for f in face_results]
        hands_data = hand_tracker.process_frame(frame_bgr, face_bboxes=face_bboxes)

        # 3. Run Gesture Engine
        gesture_state = gesture_engine.analyze(hands_data)

        # 4. Render Output
        if payload.mode == "holo":
            rendered = hud_renderer.render(frame_bgr, face_results, hands_data, gesture_state)
        else:
            rendered, _ = engine.process_frame(frame_bgr, draw_annotations=True)

        # Encode back to JPEG
        _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_img = base64.b64encode(buffer).decode("utf-8")

        # Telemetry data
        dom_emo = face_results[0]["emotion"] if face_results else None
        dom_conf = face_results[0]["confidence"] if face_results else 0.0
        probs = face_results[0]["probabilities"] if face_results else None

        return {
            "image": encoded_img,
            "dominant_emotion": dom_emo,
            "dominant_confidence": dom_conf,
            "probabilities": probs,
            "gestures": gesture_state.get("gestures", []),
            "pinch_active": gesture_state.get("pinch_active", False),
            "fallback_active": gesture_state.get("fallback_active", True)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
