import base64
import time
import io
import cv2
import numpy as np
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import EmotionEngine

app = FastAPI(title="EmoVision Hologram & Gesture Workspace")

# Initialize AI Engine
engine = EmotionEngine(model_path="model.h5", model_type="cnn", detector_type="haar")

class FaceFramePayload(BaseModel):
    image: str # Base64 JPEG

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iron Man Hologram & Gesture Workspace // EmoVision PRO</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    
    <!-- MediaPipe Hands & Camera Utils -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

    <style>
        :root {
            --bg: #050711;
            --cyan: #00f2fe;
            --magenta: #ff0844;
            --neon-blue: #4facfe;
            --yellow: #f59e0b;
            --green: #10b981;
            --purple: #8a2be2;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 50% 10%, rgba(0, 242, 254, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(255, 8, 68, 0.08) 0%, transparent 50%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px;
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', sans-serif;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Top Header */
        header {
            padding: 0.8rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(0, 242, 254, 0.25);
            background: rgba(5, 7, 17, 0.9);
            backdrop-filter: blur(15px);
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--cyan), #fff, var(--magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.08em;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 242, 254, 0.12);
            color: var(--cyan);
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(0, 242, 254, 0.3);
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--cyan);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--cyan);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.4); }
        }

        /* Main Workspace Container */
        .workspace-area {
            position: relative;
            flex: 1;
            width: 100%;
            height: calc(100vh - 65px);
            overflow: hidden;
        }

        /* Full Screen Interactive Video & Canvas Stage */
        #webcamVideo {
            display: none;
        }

        #holoCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1); /* Mirror view for intuitive control */
        }

        #uiOverlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        /* Gesture Control HUD Banner (Top Left) */
        .hud-top-left {
            position: absolute;
            top: 20px;
            left: 25px;
            background: rgba(10, 15, 30, 0.82);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 242, 254, 0.4);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 242, 254, 0.1);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            z-index: 20;
            width: 290px;
        }

        .hud-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85rem;
            color: var(--cyan);
            letter-spacing: 0.05em;
            margin-bottom: 0.6rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .gesture-pill {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.4rem 0.6rem;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            margin-bottom: 0.4rem;
            font-size: 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: all 0.2s ease;
        }

        .gesture-pill.active {
            border-color: var(--magenta);
            background: rgba(255, 8, 68, 0.2);
            box-shadow: 0 0 12px rgba(255, 8, 68, 0.4);
            color: #fff;
            transform: translateX(4px);
        }

        /* Dominant Emotion HUD Card (Top Right) */
        .hud-top-right {
            position: absolute;
            top: 20px;
            right: 25px;
            background: rgba(10, 15, 30, 0.82);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 8, 68, 0.4);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(255, 8, 68, 0.1);
            border-radius: 14px;
            padding: 1rem 1.4rem;
            z-index: 20;
            width: 320px;
        }

        .emotion-badge-row {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 0.4rem;
        }

        .emotion-big-emoji {
            font-size: 2.8rem;
        }

        .emotion-big-name {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.4rem;
            font-weight: 800;
            color: #fff;
        }

        .emotion-big-conf {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--cyan);
        }

        /* Floating Tool Palette at Bottom Center */
        .hud-bottom-bar {
            position: absolute;
            bottom: 25px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 15, 30, 0.85);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 242, 254, 0.35);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(0, 242, 254, 0.1);
            border-radius: 16px;
            padding: 0.6rem 1.4rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            z-index: 20;
        }

        .hud-btn {
            background: rgba(0, 242, 254, 0.12);
            color: var(--cyan);
            border: 1px solid rgba(0, 242, 254, 0.3);
            padding: 0.55rem 1.2rem;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .hud-btn:hover, .hud-btn.active {
            background: var(--cyan);
            color: #000;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
            transform: translateY(-2px);
        }

        /* Loading Screen */
        .loading-screen {
            position: absolute;
            inset: 0;
            background: rgba(5, 7, 17, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 50;
            gap: 1.2rem;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 3px solid rgba(0, 242, 254, 0.15);
            border-top-color: var(--cyan);
            border-bottom-color: var(--magenta);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>IRON MAN HOLO-WORKSPACE</h1>
            <span class="status-badge" id="hudStatus"><span class="status-dot"></span> <span id="statusText">INITIALIZING AI...</span></span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8;">
            AI TRACKING: <span style="color:var(--cyan); font-weight:700;">21 LANDMARKS</span> // LATENCY: <span id="telemetryFps" style="color:var(--cyan); font-weight:700;">60.0 FPS</span>
        </div>
    </header>

    <main class="workspace-area">
        <!-- Raw video stream -->
        <video id="webcamVideo" playsinline autoplay muted></video>
        <!-- Canvas for rendering Camera + Glowing Finger Connections -->
        <canvas id="holoCanvas"></canvas>
        <!-- UI Canvas for Air Objects & 3D Holograms -->
        <canvas id="uiOverlayCanvas"></canvas>

        <!-- Loading Overlay -->
        <div class="loading-screen" id="loadingScreen">
            <div class="spinner"></div>
            <h2 style="font-family: 'Orbitron', sans-serif; color: var(--cyan); letter-spacing: 0.05em;">INITIALIZING VISION & GESTURE CORE</h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">Please allow camera access when prompted...</p>
        </div>

        <!-- Top Left Gesture Telemetry -->
        <div class="hud-top-left">
            <div class="hud-title">
                <span>GESTURE TELEMETRY</span>
                <span id="handCountBadge" style="color: #94a3b8; font-size: 0.75rem;">0 HANDS</span>
            </div>
            <div class="gesture-pill" id="pillPoint">
                <span>👉</span>
                <div><strong>AIR CURSOR</strong> (Point with Index)</div>
            </div>
            <div class="gesture-pill" id="pillPinch">
                <span>🤏</span>
                <div><strong>PINCH / HOLD</strong> (Grab & Drag Object)</div>
            </div>
            <div class="gesture-pill" id="pillTwoHand">
                <span>👐</span>
                <div><strong>TWO-HAND SCALE</strong> (Resize in Air)</div>
            </div>
            <div class="gesture-pill" id="pillPalm">
                <span>🖐️</span>
                <div><strong>OPEN PALM</strong> (Reset Workspace)</div>
            </div>
        </div>

        <!-- Top Right Dominant Emotion Card -->
        <div class="hud-top-right">
            <div class="hud-title">
                <span>FACIAL EMOTION PERCEPTION</span>
                <span style="color: var(--magenta); font-size: 0.75rem;">LIVE CNN</span>
            </div>
            <div class="emotion-badge-row">
                <div class="emotion-big-emoji" id="domEmoji">🎭</div>
                <div>
                    <div class="emotion-big-name" id="domName">CALIBRATING</div>
                    <div class="emotion-big-conf" id="domConf">Confidence: 0.0%</div>
                </div>
            </div>
        </div>

        <!-- Bottom Controls -->
        <div class="hud-bottom-bar">
            <button class="hud-btn" onclick="spawnHoloObject('cube')">
                <span>🧊</span> SPAWN 3D HOLO CUBE
            </button>
            <button class="hud-btn" onclick="spawnHoloObject('orb')">
                <span>🔮</span> SPAWN EMOTION ORB
            </button>
            <button class="hud-btn" onclick="resetHoloObjects()">
                <span>🔄</span> RESET OBJECTS
            </button>
        </div>
    </main>

    <script>
        const video = document.getElementById("webcamVideo");
        const holoCanvas = document.getElementById("holoCanvas");
        const holoCtx = holoCanvas.getContext("2d");
        const uiCanvas = document.getElementById("uiOverlayCanvas");
        const uiCtx = uiCanvas.getContext("2d");
        const loadingScreen = document.getElementById("loadingScreen");
        const statusText = document.getElementById("statusText");
        const handCountBadge = document.getElementById("handCountBadge");

        let streamWidth = 1280;
        let streamHeight = 720;
        let lastFrameTime = performance.now();
        let fpsCounter = 0;
        let fpsDisplay = 60.0;

        // Emotion data received from backend
        let currentEmotion = "Neutral";
        let currentEmotionConfidence = 0.85;
        const EMOTION_EMOJIS = {
            "Angry": "😠", "Disgusted": "🤢", "Fearful": "😨",
            "Happy": "😄", "Neutral": "😐", "Sad": "😢", "Surprised": "😲"
        };

        // ---------------------------------------------------------------------
        // Interactive 3D Floating Hologram Objects in Air
        // ---------------------------------------------------------------------
        let holoObjects = [
            {
                id: "holo_cube_1",
                type: "cube",
                x: 0.5, // Normalized [0, 1] screen coords
                y: 0.5,
                z: 0.0,
                size: 140, // Base pixel size
                targetSize: 140,
                rotX: 0.3,
                rotY: 0.4,
                rotZ: 0.0,
                isGrabbed: false,
                grabbedHandId: null,
                grabOffsetX: 0,
                grabOffsetY: 0,
                glowColor: "#00f2fe"
            }
        ];

        function spawnHoloObject(type) {
            holoObjects.push({
                id: "holo_" + Date.now(),
                type: type,
                x: 0.35 + Math.random() * 0.3,
                y: 0.35 + Math.random() * 0.3,
                z: 0.0,
                size: 130,
                targetSize: 130,
                rotX: Math.random() * Math.PI,
                rotY: Math.random() * Math.PI,
                rotZ: 0.0,
                isGrabbed: false,
                grabbedHandId: null,
                grabOffsetX: 0,
                grabOffsetY: 0,
                glowColor: type === "cube" ? "#00f2fe" : "#ff0844"
            });
        }

        function resetHoloObjects() {
            holoObjects = [
                {
                    id: "holo_cube_1",
                    type: "cube",
                    x: 0.5,
                    y: 0.5,
                    z: 0.0,
                    size: 140,
                    targetSize: 140,
                    rotX: 0.3,
                    rotY: 0.4,
                    rotZ: 0.0,
                    isGrabbed: false,
                    grabbedHandId: null,
                    grabOffsetX: 0,
                    grabOffsetY: 0,
                    glowColor: "#00f2fe"
                }
            ];
        }

        // ---------------------------------------------------------------------
        // MediaPipe Connections Structure (21 Landmarks)
        // ---------------------------------------------------------------------
        const HAND_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],        // Index
            [5, 9], [9, 10], [10, 11], [11, 12],   // Middle
            [9, 13], [13, 14], [14, 15], [15, 16], // Ring
            [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
            [0, 17]                                // Palm Base
        ];

        // ---------------------------------------------------------------------
        // MediaPipe Hands Initialization (Client-side 60 FPS GPU Pipeline)
        // ---------------------------------------------------------------------
        const hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onHandResults);

        // Resize Canvas dynamically
        function resizeCanvases() {
            holoCanvas.width = window.innerWidth;
            holoCanvas.height = window.innerHeight;
            uiCanvas.width = window.innerWidth;
            uiCanvas.height = window.innerHeight;
        }
        window.addEventListener("resize", resizeCanvases);
        resizeCanvases();

        // ---------------------------------------------------------------------
        // Main Processing & Rendering Loop
        // ---------------------------------------------------------------------
        let lastHandResults = null;

        function onHandResults(results) {
            loadingScreen.style.display = "none";
            statusText.innerText = "AIR GESTURE CORE ACTIVE";
            lastHandResults = results;

            // Calculate FPS
            const now = performance.now();
            fpsCounter++;
            if (now - lastFrameTime >= 1000) {
                fpsDisplay = (fpsCounter * 1000 / (now - lastFrameTime)).toFixed(1);
                document.getElementById("telemetryFps").innerText = `${fpsDisplay} FPS`;
                fpsCounter = 0;
                lastFrameTime = now;
            }

            const w = holoCanvas.width;
            const h = holoCanvas.height;

            // 1. Draw Mirrored Camera Video Feed on holoCanvas
            holoCtx.save();
            holoCtx.clearRect(0, 0, w, h);
            holoCtx.drawImage(results.image, 0, 0, w, h);

            // Darken slightly for vibrant glowing hologram contrast
            holoCtx.fillStyle = "rgba(5, 7, 17, 0.35)";
            holoCtx.fillRect(0, 0, w, h);

            // 2. Draw Glowing Hand Skeletons & Inter-Finger Bone Connections
            if (results.multiHandLandmarks) {
                handCountBadge.innerText = `${results.multiHandLandmarks.length} HAND(S)`;

                results.multiHandLandmarks.forEach((lms, handIdx) => {
                    drawHandHologram(holoCtx, lms, w, h, handIdx);
                });
            } else {
                handCountBadge.innerText = "0 HANDS";
            }
            holoCtx.restore();

            // 3. Clear UI Overlay Canvas
            uiCtx.clearRect(0, 0, w, h);

            // 4. Update & Render Interactive Air Objects & Cursors
            handleAirGesturesAndObjects(results, w, h);
        }

        // ---------------------------------------------------------------------
        // Draw Glowing Hologram Hand Skeletons & Inter-Finger Joints
        // ---------------------------------------------------------------------
        function drawHandHologram(ctx, lms, w, h, handIdx) {
            const isRight = handIdx === 0;
            const baseColor = isRight ? "#00f2fe" : "#ff0844";
            const glowColor = isRight ? "rgba(0, 242, 254, 0.6)" : "rgba(255, 8, 68, 0.6)";

            // Draw Glowing Bone Connections
            ctx.strokeStyle = baseColor;
            ctx.shadowColor = baseColor;
            ctx.shadowBlur = 15;
            ctx.lineWidth = 3.5;
            ctx.lineCap = "round";

            HAND_CONNECTIONS.forEach(([i, j]) => {
                const p1 = lms[i];
                const p2 = lms[j];
                ctx.beginPath();
                ctx.moveTo(p1.x * w, p1.y * h);
                ctx.lineTo(p2.x * w, p2.y * h);
                ctx.stroke();
            });

            // Draw Cross-finger webbing / palm web connections (Iron Man Hologram mesh)
            ctx.strokeStyle = glowColor;
            ctx.lineWidth = 1.2;
            const webPairs = [[4, 8], [8, 12], [12, 16], [16, 20], [5, 17]];
            webPairs.forEach(([i, j]) => {
                ctx.beginPath();
                ctx.moveTo(lms[i].x * w, lms[i].y * h);
                ctx.lineTo(lms[j].x * w, lms[j].y * h);
                ctx.stroke();
            });

            // Draw Glowing Landmark Nodes
            lms.forEach((lm, idx) => {
                const px = lm.x * w;
                const py = lm.y * h;

                // Fingertips (4, 8, 12, 16, 20) get pulsing interactive rings
                if ([4, 8, 12, 16, 20].includes(idx)) {
                    ctx.fillStyle = "#ffffff";
                    ctx.shadowColor = baseColor;
                    ctx.shadowBlur = 20;
                    ctx.beginPath();
                    ctx.arc(px, py, 6, 0, Math.PI * 2);
                    ctx.fill();

                    // Outer pulse ring
                    ctx.strokeStyle = baseColor;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.arc(px, py, 11, 0, Math.PI * 2);
                    ctx.stroke();
                } else {
                    // Regular joints
                    ctx.fillStyle = baseColor;
                    ctx.shadowBlur = 10;
                    ctx.beginPath();
                    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }

        // ---------------------------------------------------------------------
        // Air Gestures: Air Cursor, Pinch Grab, Two-Hand Resize & Object Physics
        // ---------------------------------------------------------------------
        function handleAirGesturesAndObjects(results, w, h) {
            let isPinching = false;
            let isPointing = false;
            let isTwoHandScaling = false;
            let isOpenPalm = false;

            let handPointers = [];

            if (results && results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                results.multiHandLandmarks.forEach((lms, handIdx) => {
                    // Because video is mirrored on canvas, screenX is flipped for UI coordinates
                    const indexTip = lms[8];
                    const thumbTip = lms[4];
                    const wrist = lms[0];

                    const cursorX = (1.0 - indexTip.x) * w;
                    const cursorY = indexTip.y * h;

                    const thumbX = (1.0 - thumbTip.x) * w;
                    const thumbY = thumbTip.y * h;

                    // Calculate Pinch Distance in pixels
                    const pinchDist = Math.hypot(cursorX - thumbX, cursorY - thumbY);
                    const pinching = pinchDist < 45;

                    if (pinching) isPinching = true;
                    isPointing = true;

                    // Check Open Palm (fingers spread)
                    const isPalm = (lms[8].y < lms[6].y && lms[12].y < lms[10].y && lms[16].y < lms[14].y && lms[20].y < lms[18].y);
                    if (isPalm && !pinching) isOpenPalm = true;

                    handPointers.push({
                        handIdx: handIdx,
                        cursorX: cursorX,
                        cursorY: cursorY,
                        thumbX: thumbX,
                        thumbY: thumbY,
                        pinchDist: pinchDist,
                        pinching: pinching
                    });

                    // Draw Air Laser Pointer & Targeting Crosshair
                    drawAirCursor(uiCtx, cursorX, cursorY, pinching, handIdx);
                });

                // Two-Hand Scale Detection
                if (handPointers.length >= 2) {
                    const p1 = handPointers[0];
                    const p2 = handPointers[1];
                    const twoHandDist = Math.hypot(p1.cursorX - p2.cursorX, p1.cursorY - p2.cursorY);

                    // If both hands are active, scale all floating objects dynamically
                    holoObjects.forEach(obj => {
                        obj.targetSize = Math.max(70, Math.min(320, twoHandDist * 0.45));
                    });
                    isTwoHandScaling = true;
                }

                // -------------------------------------------------------------
                // Object Drag / Hold Physics
                // -------------------------------------------------------------
                holoObjects.forEach(obj => {
                    const objPxX = obj.x * w;
                    const objPxY = obj.y * h;
                    const radius = obj.size / 2;

                    // Smooth resize lerp
                    obj.size += (obj.targetSize - obj.size) * 0.15;

                    // Continuous slow 3D rotation
                    obj.rotX += 0.015;
                    obj.rotY += 0.02;

                    // Check if any hand pinches inside/near this object
                    let grabbedBy = null;
                    handPointers.forEach(p => {
                        const distToObject = Math.hypot(p.cursorX - objPxX, p.cursorY - objPxY);
                        if (distToObject < radius + 40 && p.pinching) {
                            grabbedBy = p;
                        }
                    });

                    if (grabbedBy) {
                        obj.isGrabbed = true;
                        obj.x = grabbedBy.cursorX / w;
                        obj.y = grabbedBy.cursorY / h;
                        obj.glowColor = "#ff0844"; // Glow magenta when held in air!
                    } else {
                        obj.isGrabbed = false;
                        obj.glowColor = obj.type === "cube" ? "#00f2fe" : "#4facfe";
                    }
                });
            }

            // Update UI Gesture Pills
            document.getElementById("pillPoint").classList.toggle("active", isPointing);
            document.getElementById("pillPinch").classList.toggle("active", isPinching);
            document.getElementById("pillTwoHand").classList.toggle("active", isTwoHandScaling);
            document.getElementById("pillPalm").classList.toggle("active", isOpenPalm);

            // Render 3D Holographic Objects in Air
            holoObjects.forEach(obj => {
                if (obj.type === "cube") {
                    drawHologram3DCube(uiCtx, obj.x * w, obj.y * h, obj.size, obj.rotX, obj.rotY, obj.glowColor, obj.isGrabbed);
                } else if (obj.type === "orb") {
                    drawHologramOrb(uiCtx, obj.x * w, obj.y * h, obj.size, obj.glowColor, obj.isGrabbed);
                }
            });
        }

        // ---------------------------------------------------------------------
        // Draw Laser Pointer & Pulsing Crosshair Cursor in Mid-Air
        // ---------------------------------------------------------------------
        function drawAirCursor(ctx, x, y, isPinching, handIdx) {
            const col = isPinching ? "#ff0844" : "#00f2fe";

            ctx.save();
            ctx.shadowColor = col;
            ctx.shadowBlur = 15;

            // Concentric target reticle
            ctx.strokeStyle = col;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, isPinching ? 18 : 12, 0, Math.PI * 2);
            ctx.stroke();

            // Center Point
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(x, y, isPinching ? 5 : 3, 0, Math.PI * 2);
            ctx.fill();

            // Corner Aim Crosshairs
            ctx.beginPath();
            ctx.moveTo(x - 20, y); ctx.lineTo(x - 8, y);
            ctx.moveTo(x + 8, y);  ctx.lineTo(x + 20, y);
            ctx.moveTo(x, y - 20); ctx.lineTo(x, y - 8);
            ctx.moveTo(x, y + 8);  ctx.lineTo(x, y + 20);
            ctx.stroke();

            // Holographic Cursor Tag
            ctx.font = "10px 'JetBrains Mono', monospace";
            ctx.fillStyle = col;
            ctx.fillText(isPinching ? "HOLD // GRAB" : `POINT [${Math.round(x)},${Math.round(y)}]`, x + 16, y - 10);

            ctx.restore();
        }

        // ---------------------------------------------------------------------
        // Render 3D Holographic Wireframe Cube in Mid-Air (Iron Man Aesthetic)
        // ---------------------------------------------------------------------
        function drawHologram3DCube(ctx, cx, cy, size, rx, ry, glowColor, isGrabbed) {
            ctx.save();
            const hs = size / 2;

            // 8 Cube 3D Vertices
            const vertices = [
                [-hs, -hs, -hs], [ hs, -hs, -hs], [ hs,  hs, -hs], [-hs,  hs, -hs],
                [-hs, -hs,  hs], [ hs, -hs,  hs], [ hs,  hs,  hs], [-hs,  hs,  hs]
            ];

            // 3D Rotation Matrix Projection
            const cosX = Math.cos(rx), sinX = Math.sin(rx);
            const cosY = Math.cos(ry), sinY = Math.sin(ry);

            const proj = vertices.map(([vx, vy, vz]) => {
                // Rot Y
                let x1 = vx * cosY - vz * sinY;
                let z1 = vx * sinY + vz * cosY;
                // Rot X
                let y2 = vy * cosX - z1 * sinX;
                let z2 = vy * sinX + z1 * cosX;

                // Perspective projection
                const fov = 350;
                const scale = fov / (fov + z2 + 200);
                return [cx + x1 * scale, cy + y2 * scale];
            });

            // Cube 12 Edges
            const edges = [
                [0, 1], [1, 2], [2, 3], [3, 0], // Back Face
                [4, 5], [5, 6], [6, 7], [7, 4], // Front Face
                [0, 4], [1, 5], [2, 6], [3, 7]  // Connecting Struts
            ];

            // Draw Glowing 3D Edges
            ctx.strokeStyle = glowColor;
            ctx.shadowColor = glowColor;
            ctx.shadowBlur = isGrabbed ? 30 : 18;
            ctx.lineWidth = isGrabbed ? 3.5 : 2.2;

            edges.forEach(([i, j]) => {
                ctx.beginPath();
                ctx.moveTo(proj[i][0], proj[i][1]);
                ctx.lineTo(proj[j][0], proj[j][1]);
                ctx.stroke();
            });

            // Draw Glowing Corner Vertex Nodes
            ctx.fillStyle = "#ffffff";
            proj.forEach(([px, py]) => {
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fill();
            });

            // Center Arc Energy Core
            ctx.strokeStyle = glowColor;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, isGrabbed ? 20 : 12, 0, Math.PI * 2);
            ctx.stroke();

            // Holographic Status Tag
            ctx.font = "11px 'Orbitron', sans-serif";
            ctx.fillStyle = glowColor;
            ctx.fillText(isGrabbed ? "⚡ HOLDING OBJECT" : "HOLO-OBJECT // PINCH TO GRAB", cx - 70, cy + hs + 25);

            ctx.restore();
        }

        // ---------------------------------------------------------------------
        // Render 3D Floating Emotion Energy Orb
        // ---------------------------------------------------------------------
        function drawHologramOrb(ctx, cx, cy, size, glowColor, isGrabbed) {
            ctx.save();
            const r = size / 2;

            ctx.shadowColor = glowColor;
            ctx.shadowBlur = isGrabbed ? 35 : 20;

            // Outer Energy Shell
            ctx.strokeStyle = glowColor;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();

            // Orbiting Ring 1
            const time = performance.now() * 0.003;
            ctx.beginPath();
            ctx.ellipse(cx, cy, r, r * 0.35, time, 0, Math.PI * 2);
            ctx.stroke();

            // Orbiting Ring 2
            ctx.beginPath();
            ctx.ellipse(cx, cy, r, r * 0.35, -time * 1.3, 0, Math.PI * 2);
            ctx.stroke();

            // Glowing Core
            ctx.fillStyle = glowColor;
            ctx.beginPath();
            ctx.arc(cx, cy, r * 0.25, 0, Math.PI * 2);
            ctx.fill();

            // Center Emoji
            ctx.font = `${Math.round(r * 0.4)}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(EMOTION_EMOJIS[currentEmotion] || "🔮", cx, cy);

            ctx.restore();
        }

        // ---------------------------------------------------------------------
        // Periodic Facial Emotion Prediction Call to Backend
        // ---------------------------------------------------------------------
        const emotionCaptureCanvas = document.createElement("canvas");
        const emoCtx = emotionCaptureCanvas.getContext("2d");

        async function triggerEmotionInference() {
            if (video.videoWidth > 0) {
                emotionCaptureCanvas.width = 480;
                emotionCaptureCanvas.height = 360;
                emoCtx.drawImage(video, 0, 0, 480, 360);

                const dataUrl = emotionCaptureCanvas.toDataURL("image/jpeg", 0.65);
                const b64 = dataUrl.split(",")[1];

                try {
                    const res = await fetch("/api/predict_face", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: b64 })
                    });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.dominant_emotion) {
                            currentEmotion = data.dominant_emotion;
                            currentEmotionConfidence = data.dominant_confidence;

                            document.getElementById("domEmoji").innerText = EMOTION_EMOJIS[currentEmotion] || "🎭";
                            document.getElementById("domName").innerText = currentEmotion.toUpperCase();
                            document.getElementById("domConf").innerText = `Confidence: ${(currentEmotionConfidence * 100).toFixed(1)}%`;
                        }
                    }
                } catch (e) {
                    // pass
                }
            }
            setTimeout(triggerEmotionInference, 400); // 2.5 times per second
        }

        // ---------------------------------------------------------------------
        // Camera Startup
        // ---------------------------------------------------------------------
        const camera = new Camera(video, {
            onFrame: async () => {
                await hands.send({ image: video });
            },
            width: 1280,
            height: 720
        });

        camera.start().then(() => {
            triggerEmotionInference();
        }).catch(err => {
            alert("Camera Initialization Failed: " + err.message);
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.post("/api/predict_face")
async def predict_face(payload: FaceFramePayload):
    try:
        img_bytes = base64.b64decode(payload.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return JSONResponse({"error": "Invalid frame"}, status_code=400)

        _, results = engine.process_frame(frame_bgr, draw_annotations=False)

        dom_emo = results[0]["emotion"] if results else None
        dom_conf = results[0]["confidence"] if results else 0.0
        probs = results[0]["probabilities"] if results else None

        return {
            "dominant_emotion": dom_emo,
            "dominant_confidence": dom_conf,
            "probabilities": probs
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
