import cv2
import numpy as np
import math
import time
from collections import deque

class HUDRenderer:
    """
    Holographic Sci-Fi HUD rendering engine.
    Draws Iron Man-style floating glassmorphic panels, rotating arc reticles,
    hand skeleton tracking overlays, and real-time emotion telemetry.
    """
    def __init__(self):
        self.angle = 0
        self.mood_history = deque(maxlen=40)
        self.start_time = time.time()
        self.active_mode = 0
        self.modes = ["EMOTION SPECTRUM", "TELEMETRY & HUD", "ANALYTICS"]
        self.panel_scale = 1.0

    def render(self, frame_bgr, face_results, hand_results, gesture_state):
        """
        Renders the holographic HUD overlay on top of the video frame.
        """
        h_img, w_img = frame_bgr.shape[:2]
        hud_overlay = frame_bgr.copy()
        glow_layer = np.zeros_like(frame_bgr)

        self.angle = (self.angle + 4) % 360

        # Handle Mode Switch gesture
        gestures = gesture_state.get("gestures", [])
        if "SWIPE_RIGHT" in gestures:
            self.active_mode = (self.active_mode + 1) % len(self.modes)
        elif "SWIPE_LEFT" in gestures:
            self.active_mode = (self.active_mode - 1) % len(self.modes)

        # Handle Two-Hand Zoom gesture
        if gesture_state.get("zoom_ratio") is not None:
            self.panel_scale = np.clip(self.panel_scale * gesture_state["zoom_ratio"], 0.7, 1.4)

        # 1. Draw Top Sci-Fi Header & Status Bar
        self._draw_header(hud_overlay, glow_layer, w_img, gesture_state)

        # 2. Draw Arc-Reactor Face Reticle
        if face_results:
            top_face = face_results[0]
            self._draw_face_arc_reticle(hud_overlay, glow_layer, top_face)
            self.mood_history.append(top_face["emotion"])

        # 3. Draw Floating Hologram Panels
        self._draw_floating_panels(hud_overlay, glow_layer, w_img, h_img, face_results)

        # 4. Draw Hand Skeletons & Pinch Reticle
        if hand_results:
            self._draw_hand_tracking(hud_overlay, glow_layer, hand_results, gesture_state)

        # 5. Blend glow layer for luminous holographic look
        alpha_base = 0.82
        alpha_glow = 0.40
        blended = cv2.addWeighted(hud_overlay, alpha_base, frame_bgr, 1.0 - alpha_base, 0)
        final_hud = cv2.add(blended, (glow_layer * alpha_glow).astype(np.uint8))

        return final_hud

    def _draw_header(self, overlay, glow, w_img, gesture_state):
        # Top banner panel
        cv2.rectangle(overlay, (20, 15), (w_img - 20, 65), (20, 15, 35), -1)
        cv2.rectangle(overlay, (20, 15), (w_img - 20, 65), (0, 242, 254), 1)

        # Title
        cv2.putText(overlay, "EMOVISION HOLO-HUD v2.0", (35, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 242, 254), 2, cv2.LINE_AA)
        cv2.putText(glow, "EMOVISION HOLO-HUD v2.0", (35, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 242, 254), 3, cv2.LINE_AA)

        # Active Mode Tabs
        mode_text = f"MODE: [{self.modes[self.active_mode]}]"
        cv2.putText(overlay, mode_text, (w_img - 340, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Gesture vs Fallback Badge
        if gesture_state.get("fallback_active", False):
            status_text = "MOUSE FALLBACK"
            status_color = (0, 165, 255) # Orange
        else:
            status_text = "GESTURE CONTROL"
            status_color = (0, 255, 136) # Neon Green

        cv2.circle(overlay, (w_img - 365, 44), 5, status_color, -1)
        cv2.putText(overlay, status_text, (w_img - 510, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    def _draw_face_arc_reticle(self, overlay, glow, face_data):
        x, y, w, h = face_data["bbox"]
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2 + 30

        # Primary Rotating Cyan Ring
        cv2.circle(overlay, (cx, cy), radius, (0, 242, 254), 1, cv2.LINE_AA)
        cv2.circle(glow, (cx, cy), radius, (0, 242, 254), 2, cv2.LINE_AA)

        # Rotating dashed segments
        for i in range(4):
            start_ang = self.angle + i * 90
            end_ang = start_ang + 45
            cv2.ellipse(overlay, (cx, cy), (radius + 12, radius + 12), 0, start_ang, end_ang, (254, 8, 255), 2, cv2.LINE_AA)
            cv2.ellipse(glow, (cx, cy), (radius + 12, radius + 12), 0, start_ang, end_ang, (254, 8, 255), 3, cv2.LINE_AA)

        # Corner bracket reticles
        corner_len = 20
        # Top-left
        cv2.line(overlay, (x - 10, y - 10), (x - 10 + corner_len, y - 10), (0, 242, 254), 2)
        cv2.line(overlay, (x - 10, y - 10), (x - 10, y - 10 + corner_len), (0, 242, 254), 2)
        # Top-right
        cv2.line(overlay, (x + w + 10, y - 10), (x + w + 10 - corner_len, y - 10), (0, 242, 254), 2)
        cv2.line(overlay, (x + w + 10, y - 10), (x + w + 10, y - 10 + corner_len), (0, 242, 254), 2)
        # Bottom-left
        cv2.line(overlay, (x - 10, y + h + 10), (x - 10 + corner_len, y + h + 10), (0, 242, 254), 2)
        cv2.line(overlay, (x - 10, y + h + 10), (x - 10, y + h + 10 - corner_len), (0, 242, 254), 2)
        # Bottom-right
        cv2.line(overlay, (x + w + 10, y + h + 10), (x + w + 10 - corner_len, y + h + 10), (0, 242, 254), 2)
        cv2.line(overlay, (x + w + 10, y + h + 10), (x + w + 10, y + h + 10 - corner_len), (0, 242, 254), 2)

        # Floating Emotion HUD Tag
        emotion = face_data["emotion"].upper()
        conf = face_data["confidence"] * 100
        tag_text = f"TARGET // {emotion} ({conf:.1f}%)"

        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x - 10, y - 45), (x + tw + 15, y - 15), (15, 10, 30), -1)
        cv2.rectangle(overlay, (x - 10, y - 45), (x + tw + 15, y - 15), (0, 242, 254), 1)
        cv2.putText(overlay, tag_text, (x - 2, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2, cv2.LINE_AA)

    def _draw_floating_panels(self, overlay, glow, w_img, h_img, face_results):
        panel_w = int(270 * self.panel_scale)
        panel_h = int(240 * self.panel_scale)
        px1 = w_img - panel_w - 25
        py1 = 80
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        # Semi-transparent glass panel background
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (18, 12, 32), -1)
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 242, 254), 1)

        # Header for panel
        cv2.putText(overlay, "SPECTRUM TELEMETRY", (px1 + 15, py1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 1, cv2.LINE_AA)
        cv2.line(overlay, (px1 + 15, py1 + 32), (px2 - 15, py1 + 32), (0, 242, 254), 1)

        if face_results:
            probs = face_results[0]["probabilities"]
            bar_start_y = py1 + 55
            row_gap = 25

            for i, (emo, p) in enumerate(probs.items()):
                curr_y = bar_start_y + i * row_gap
                # Text label
                cv2.putText(overlay, emo[:4].upper(), (px1 + 15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1, cv2.LINE_AA)

                # Progress bar track
                bar_x = px1 + 65
                max_bar_w = panel_w - 120
                cv2.rectangle(overlay, (bar_x, curr_y - 9), (bar_x + max_bar_w, curr_y - 1), (35, 25, 55), -1)

                # Filled progress bar
                fill_w = int(p * max_bar_w)
                if fill_w > 0:
                    bar_col = (0, 242, 254) if p > 0.3 else (180, 100, 240)
                    cv2.rectangle(overlay, (bar_x, curr_y - 9), (bar_x + fill_w, curr_y - 1), bar_col, -1)
                    cv2.rectangle(glow, (bar_x, curr_y - 9), (bar_x + fill_w, curr_y - 1), bar_col, -1)

                # Percent text
                cv2.putText(overlay, f"{p*100:.0f}%", (bar_x + max_bar_w + 8, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Left Gesture Status Panel
        g_px1 = 25
        g_py1 = h_img - 110
        g_px2 = 280
        g_py2 = h_img - 25

        cv2.rectangle(overlay, (g_px1, g_py1), (g_px2, g_py2), (18, 12, 32), -1)
        cv2.rectangle(overlay, (g_px1, g_py1), (g_px2, g_py2), (254, 8, 255), 1)

        cv2.putText(overlay, "GESTURE RECOGNITION", (g_px1 + 15, g_py1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (254, 8, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "• PINCH  : Select / Click", (g_px1 + 15, g_py1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, "• SWIPE  : Mode Switch", (g_px1 + 15, g_py1 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(overlay, "• PALM   : Reset Panels", (g_px1 + 15, g_py1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_hand_tracking(self, overlay, glow, hand_results, gesture_state):
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Base
        ]

        for hand in hand_results:
            lms_px = hand["landmarks_px"]

            # Draw Neon Skeleton Lines
            for (p1, p2) in hand_connections:
                pt1 = (lms_px[p1, 0], lms_px[p1, 1])
                pt2 = (lms_px[p2, 0], lms_px[p2, 1])
                cv2.line(overlay, pt1, pt2, (0, 242, 254), 1, cv2.LINE_AA)
                cv2.line(glow, pt1, pt2, (0, 242, 254), 2, cv2.LINE_AA)

            # Draw Glowing Landmark Nodes
            for idx in range(21):
                pt = (lms_px[idx, 0], lms_px[idx, 1])
                if idx in [4, 8, 12, 16, 20]: # Fingertips
                    cv2.circle(overlay, pt, 4, (254, 8, 255), -1, cv2.LINE_AA)
                    cv2.circle(glow, pt, 7, (254, 8, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, pt, 2, (0, 242, 254), -1, cv2.LINE_AA)

            # Draw Interactive Cursor Reticle on Index Tip #8
            ix, iy = lms_px[8, 0], lms_px[8, 1]
            tx, ty = lms_px[4, 0], lms_px[4, 1]

            if gesture_state.get("pinch_active", False):
                # Pinch Active -> Pulsing Magenta Crosshair
                pinch_cx = (ix + tx) // 2
                pinch_cy = (iy + ty) // 2
                cv2.circle(overlay, (pinch_cx, pinch_cy), 12, (254, 8, 255), 2, cv2.LINE_AA)
                cv2.circle(glow, (pinch_cx, pinch_cy), 15, (254, 8, 255), 3, cv2.LINE_AA)
                cv2.putText(overlay, "PINCH // CLICK", (pinch_cx + 18, pinch_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 8, 255), 2, cv2.LINE_AA)
            else:
                # Hovering Cursor Reticle
                cv2.circle(overlay, (ix, iy), 8, (0, 242, 254), 1, cv2.LINE_AA)
                cv2.line(overlay, (ix - 12, iy), (ix + 12, iy), (0, 242, 254), 1)
                cv2.line(overlay, (ix, iy - 12), (ix, iy + 12), (0, 242, 254), 1)
