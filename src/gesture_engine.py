import time
import numpy as np
from collections import deque

class GestureEngine:
    """
    Rule-based gesture recognition engine supporting Pinch-to-Click,
    Open Palm Reset, Horizontal Swipe Mode Switching, and Two-Hand Zooming.
    Includes state debouncing and automatic mouse fallback handling.
    """
    def __init__(self, swipe_threshold=0.15, pinch_threshold=0.048, fallback_timeout_frames=10):
        self.swipe_threshold = swipe_threshold
        self.pinch_threshold = pinch_threshold
        self.fallback_timeout_frames = fallback_timeout_frames

        # State tracking
        self.is_pinching = False
        self.pinch_start_pos = None
        self.last_swipe_time = 0
        self.swipe_cooldown = 0.6  # seconds
        self.index_x_history = deque(maxlen=8)
        self.no_hand_frames = 0
        self.fallback_active = True
        self.last_two_hand_distance = None

    def analyze(self, hands_data):
        """
        Analyzes processed hand landmarks and returns gesture events.
        
        Args:
            hands_data: List of hand dicts from HandTracker.process_frame()
            
        Returns:
            dict containing:
                "gestures": List of active gesture strings ["PINCH_CLICK", "OPEN_PALM", "SWIPE_LEFT", etc.]
                "cursor_pos": (x, y) normalized cursor position from index fingertip
                "pinch_active": bool
                "pinch_distance": float
                "zoom_ratio": float or None
                "fallback_active": bool (True if mouse fallback should be used)
        """
        current_time = time.time()
        events = []
        cursor_pos = None
        zoom_ratio = 1.0
        pinch_dist = 1.0

        # Check hand presence
        if not hands_data:
            self.no_hand_frames += 1
            if self.no_hand_frames >= self.fallback_timeout_frames:
                self.fallback_active = True
            self.is_pinching = False
            self.index_x_history.clear()
            self.last_two_hand_distance = None
            return {
                "gestures": events,
                "cursor_pos": None,
                "pinch_active": False,
                "pinch_distance": 1.0,
                "zoom_ratio": None,
                "fallback_active": self.fallback_active
            }

        # Hand detected -> Resume gesture mode
        self.no_hand_frames = 0
        self.fallback_active = False

        # Primary hand (first detected hand)
        primary_hand = hands_data[0]
        lms = primary_hand["landmarks"] # (21, 3)
        palm_scale = primary_hand.get("palm_scale", 0.1)

        # 1. Cursor Position: Index Fingertip (#8)
        index_tip = lms[8, :2]
        thumb_tip = lms[4, :2]
        wrist = lms[0, :2]
        cursor_pos = (float(index_tip[0]), float(index_tip[1]))

        # 2. Pinch Detection (Thumb Tip #4 to Index Tip #8 distance)
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        normalized_pinch = pinch_dist / (palm_scale * 1.5)

        if normalized_pinch < self.pinch_threshold or pinch_dist < 0.045:
            if not self.is_pinching:
                self.is_pinching = True
                self.pinch_start_pos = cursor_pos
                events.append("PINCH_CLICK")
            else:
                events.append("PINCH_HOLD")
        else:
            if self.is_pinching:
                self.is_pinching = False
                events.append("PINCH_RELEASE")

        # 3. Open Palm Detection (All fingertips extended and spread)
        is_palm = self._check_open_palm(lms, wrist, palm_scale)
        if is_palm and not self.is_pinching:
            events.append("OPEN_PALM")

        # 4. Horizontal Swipe Detection (Rate of change of index x)
        self.index_x_history.append((current_time, index_tip[0]))
        if len(self.index_x_history) >= 5 and (current_time - self.last_swipe_time) > self.swipe_cooldown:
            t_old, x_old = self.index_x_history[0]
            t_new, x_new = self.index_x_history[-1]
            dt = t_new - t_old
            dx = x_new - x_old

            if dt > 0.05:
                velocity = dx / dt
                if velocity > self.swipe_threshold * 8.0:
                    events.append("SWIPE_RIGHT")
                    self.last_swipe_time = current_time
                    self.index_x_history.clear()
                elif velocity < -self.swipe_threshold * 8.0:
                    events.append("SWIPE_LEFT")
                    self.last_swipe_time = current_time
                    self.index_x_history.clear()

        # 5. Two-Hand Distance Zoom
        if len(hands_data) >= 2:
            hand1_center = hands_data[0]["landmarks"][9, :2]
            hand2_center = hands_data[1]["landmarks"][9, :2]
            curr_dist = float(np.linalg.norm(hand1_center - hand2_center))

            if self.last_two_hand_distance is not None and self.last_two_hand_distance > 0.05:
                zoom_ratio = curr_dist / self.last_two_hand_distance
                if abs(zoom_ratio - 1.0) > 0.08:
                    events.append("TWO_HAND_ZOOM")
            self.last_two_hand_distance = curr_dist
        else:
            self.last_two_hand_distance = None

        return {
            "gestures": events,
            "cursor_pos": cursor_pos,
            "pinch_active": self.is_pinching,
            "pinch_distance": pinch_dist,
            "zoom_ratio": zoom_ratio if len(hands_data) >= 2 else None,
            "fallback_active": self.fallback_active
        }

    def _check_open_palm(self, lms, wrist, palm_scale):
        """
        Determines if all fingers are extended outwards away from palm.
        """
        # Fingertips: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # PIP joints: 6 (index), 10 (middle), 14 (ring), 18 (pinky)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        extended_count = 0
        for tip, pip in zip(tips, pips):
            dist_tip_wrist = np.linalg.norm(lms[tip, :2] - wrist)
            dist_pip_wrist = np.linalg.norm(lms[pip, :2] - wrist)
            if dist_tip_wrist > dist_pip_wrist * 1.15:
                extended_count += 1

        # Check finger spread (Index tip to Pinky tip distance)
        span = np.linalg.norm(lms[8, :2] - lms[20, :2])
        return extended_count >= 4 and span > (palm_scale * 0.8)
