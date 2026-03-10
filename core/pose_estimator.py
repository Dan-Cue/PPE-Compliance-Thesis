"""
Pose Estimator - POSE ONLY (Optimized)
Only loads pose landmarks model for maximum speed
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight result wrapper — exposes only the locked person's landmarks
# ─────────────────────────────────────────────────────────────────────────────

class FilteredPoseResult:
    """
    Wraps a MediaPipe PoseLandmarker result but exposes only ONE person's
    landmarks (the currently locked subject).  All verifier functions that
    do  `pose_landmarks.pose_landmarks[0]`  continue to work without change.
    """
    def __init__(self, landmarks_list):
        # landmarks_list: a plain Python list containing at most one
        # NormalizedLandmarkList (the locked person), or an empty list.
        self.pose_landmarks = landmarks_list


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pose_landmarks(image, landmarks):
    """Draw pose skeleton on image."""
    h, w = image.shape[:2]

    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
        (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)
    ]

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(image, points[a], points[b], (0, 255, 0), 2)

    for p in points:
        cv2.circle(image, p, 4, (0, 0, 255), -1)


# ─────────────────────────────────────────────────────────────────────────────
# PoseEstimator
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimator:
    """Lightweight pose estimator — POSE ONLY, with person-locking."""

    # Normalized-coordinate distance at which we consider the tracked person
    # to have been replaced by a different one (triggers a re-lock).
    _ANCHOR_DRIFT_LIMIT = 0.15

    def __init__(self, model_dir):
        """
        Initialize pose estimator.
        Only loads pose model

        Args:
            model_dir: Directory containing pose_landmarker_lite.task
        """
        pose_path = os.path.join(model_dir, "pose_landmarker_lite.task")

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing pose model: {pose_path}")

        print(f"Loading pose model from: {pose_path}")

        # ── CHANGE: confidence raised from 0.5 → 0.8 for both detection and
        #    tracking, reducing jitter and false landmark assignments. ──────
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.8,   # was 0.5
                min_tracking_confidence=0.8,          # was 0.5
            )
        )

        # ── Person-lock state ────────────────────────────────────────────
        # Stored as normalised (x, y) mid-shoulder of the tracked subject.
        # None means no lock yet — will lock on first detection.
        self._locked_anchor = None

        print("✓ Pose detector initialized (face/hand models NOT loaded)")
        print("  Detection/tracking confidence: 0.8")
        print("  Person-locking: ENABLED")

    # ── Person-locking helpers ────────────────────────────────────────────────

    def _mid_shoulder(self, landmarks):
        """
        Return normalised (x, y) midpoint between the two shoulders,
        or None if shoulder visibility is too low.
        """
        if len(landmarks) <= 12:
            return None
        ls = landmarks[11]   # left  shoulder (MediaPipe index)
        rs = landmarks[12]   # right shoulder
        avg_vis = (getattr(ls, 'visibility', 1.0) +
                   getattr(rs, 'visibility', 1.0)) / 2.0
        if avg_vis < 0.3:
            return None
        return ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)

    def _select_and_lock(self, pose_result):
        """
        Choose which person in the frame to track and update the anchor.

        Logic:
          - Single person  → always use them, update anchor.
          - Multiple people → find the one closest to the locked anchor.
            If all candidates are beyond _ANCHOR_DRIFT_LIMIT the lock resets
            and the closest person becomes the new subject.

        Returns the integer index into pose_result.pose_landmarks, or None
        if no valid pose is present.
        """
        if not pose_result or not pose_result.pose_landmarks:
            return None

        candidates = pose_result.pose_landmarks

        if len(candidates) == 1:
            mid = self._mid_shoulder(candidates[0])
            if mid:
                self._locked_anchor = mid
            return 0

        # ── Multiple people in frame ─────────────────────────────────────
        if self._locked_anchor is None:
            # First detection ever — lock to index 0 (highest-confidence pose)
            mid = self._mid_shoulder(candidates[0])
            if mid:
                self._locked_anchor = mid
            return 0

        # Find candidate closest to the current anchor
        best_idx = 0
        best_dist = float('inf')
        for i, lm in enumerate(candidates):
            mid = self._mid_shoulder(lm)
            if mid is None:
                continue
            dist = ((mid[0] - self._locked_anchor[0]) ** 2 +
                    (mid[1] - self._locked_anchor[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Update anchor to track person's natural movement
        mid = self._mid_shoulder(candidates[best_idx])
        if mid:
            if best_dist > self._ANCHOR_DRIFT_LIMIT:
                # Large jump — log and re-lock (original person may have left)
                print(f"[PoseLock] Anchor reset — drift={best_dist:.3f} "
                      f"(>{self._ANCHOR_DRIFT_LIMIT:.2f})")
            self._locked_anchor = mid

        return best_idx

    # ── Main API ──────────────────────────────────────────────────────────────

    def process(self, frame):
        """
        Process frame, lock to one person, and return a FilteredPoseResult
        containing only that person's landmarks.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        raw_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)

        # Select and lock to one person
        idx = self._select_and_lock(raw_result)

        if idx is not None and raw_result.pose_landmarks:
            filtered = FilteredPoseResult([raw_result.pose_landmarks[idx]])
        else:
            filtered = FilteredPoseResult([])

        return filtered, None, None

    def draw(self, frame, pose_result, hand_result=None, face_result=None):
        """Draw pose landmarks on frame."""
        if pose_result and pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks:
                draw_pose_landmarks(frame, lm)

    def reset_lock(self):
        """
        Manually clear the person lock.
        Call this on system reset so the next person can be tracked fresh.
        """
        self._locked_anchor = None
        print("[PoseLock] Person lock cleared")

    def close(self):
        """Close pose detector."""
        self.pose_detector.close()
