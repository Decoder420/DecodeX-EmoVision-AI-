import cv2
import numpy as np
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import EmotionEngine
from src.hand_tracker import HandTracker
from src.gesture_engine import GestureEngine
from src.hud_renderer import HUDRenderer

def open_camera(preferred_idx=0):
    """
    Tries multiple camera indices and backends (AVFoundation on macOS, V4L2/DSHOW on others)
    to safely open the webcam.
    """
    backends = []
    if sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    candidates = [preferred_idx, 0, 1, 2]
    seen = set()

    for backend in backends:
        for idx in candidates:
            if (backend, idx) in seen:
                continue
            seen.add((backend, idx))
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[INFO] Successfully connected to Camera #{idx} via backend {backend}.")
                    return cap, idx
                cap.release()

    return None, None

def run_desktop_hud(args):
    """
    High-framerate (60 FPS) native desktop Iron Man Holographic HUD.
    Combines face emotion inference, hand landmark tracking, and gesture recognition.
    """
    print("=" * 70)
    print("       EMOVISION HOLO-HUD — GESTURE-CONTROLLED INTERFACE")
    print("=" * 70)
    print("[INFO] Initializing Emotion Engine, Hand Tracker, and Gesture State Machine...")

    engine = EmotionEngine(model_path=args.model_path, model_type=args.model, detector_type=args.detector)
    hand_tracker = HandTracker(max_num_hands=2, smooth_alpha=0.65)
    gesture_engine = GestureEngine()
    hud_renderer = HUDRenderer()

    cap, active_idx = open_camera(args.camera)
    if cap is None:
        print("\n" + "!" * 70)
        print("[ERROR] macOS Camera Access Denied or Camera Device Not Found.")
        print("!" * 70)
        print("💡 Solution:")
        print("  1. Grant Camera Permission to your Terminal / IDE:")
        print("     Go to: System Settings -> Privacy & Security -> Camera -> Enable for your IDE/Terminal.")
        print("  2. OR Run the In-Browser Holo-HUD on Streamlit (where browser handles camera automatically):")
        print("     Open: http://localhost:8501 (Navigate to the '🦾 Iron Man Holo-HUD' tab)")
        print("!" * 70 + "\n")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.ocl.setUseOpenCL(False)

    print("\n[CONTROLS]")
    print("  • Pinch (Thumb + Index) : Select / Click active element")
    print("  • Open Palm             : Reset HUD & floating cards")
    print("  • Horizontal Swipe      : Switch modes (Spectrum / Telemetry / Analytics)")
    print("  • Two-Hand Spread       : Zoom / scale holographic panels")
    print("  • Keyboard 'q'          : Exit application\n")

    frame_count = 0
    cached_face_results = []
    fps_time = time.time()
    fps_counter = 0
    fps_display = "0.0 FPS"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break

        # Flip horizontally for intuitive mirror-like interaction
        frame = cv2.flip(frame, 1)
        frame_count += 1
        fps_counter += 1

        # Calculate FPS every 30 frames
        if time.time() - fps_time >= 0.5:
            fps_display = f"{fps_counter / (time.time() - fps_time):.1f} FPS"
            fps_counter = 0
            fps_time = time.time()

        # Cadence interleaving: Face detection every 2 frames for optimal 50+ FPS throughput
        if frame_count % 2 == 0 or not cached_face_results:
            _, cached_face_results = engine.process_frame(frame, draw_annotations=False)

        # Hand tracking on EVERY frame for immediate touchless responsiveness
        hands_data = hand_tracker.process_frame(frame)

        # Gesture Analysis
        gesture_state = gesture_engine.analyze(hands_data)

        # Holographic HUD Rendering
        rendered_hud = hud_renderer.render(frame, cached_face_results, hands_data, gesture_state)

        # Render FPS & Engine telemetry badge
        cv2.putText(rendered_hud, fps_display, (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 1, cv2.LINE_AA)

        # Display window
        cv2.imshow("EmoVision Holo-HUD Pro", rendered_hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            hud_renderer.active_mode = (hud_renderer.active_mode + 1) % len(hud_renderer.modes)
        elif key == ord('r'):
            hud_renderer.panel_scale = 1.0

    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.close()
    engine.detector.close()

def main():
    parser = argparse.ArgumentParser(description="EmoVision Iron Man Holographic HUD")
    parser.add_argument("--mode", default="desktop", choices=["desktop", "web"], help="Execution mode (desktop or web)")
    parser.add_argument("--model", default="cnn", choices=["cnn", "mobilenet"], help="Emotion classifier model")
    parser.add_argument("--detector", default="haar", choices=["haar", "mediapipe"], help="Face detector backend")
    parser.add_argument("--model_path", default="model.h5", help="Path to trained weights file")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    if args.mode == "desktop":
        run_desktop_hud(args)
    elif args.mode == "web":
        print("[INFO] Launching Web 3D HUD via Streamlit...")
        os.system("streamlit run app.py")

if __name__ == '__main__':
    main()
