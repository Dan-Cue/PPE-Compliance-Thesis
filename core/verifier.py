"""
verifier.py — PPE Geometric Verification via MediaPipe Pose Keypoints
======================================================================

OVERVIEW
--------
Provides per-item PPE verification functions that decide whether a YOLO
bounding box contains a correctly worn piece of equipment. All decisions
are made by checking whether specific MediaPipe Pose keypoints fall inside
(or sufficiently close to) the detection bounding box. No face mesh or
hand landmark models are used, making the module reliable at distances
of 2 metres and beyond where those models degrade.

Each function returns one of three string constants:
  "DETECTED_CORRECT"    — item is present and geometrically confirmed worn.
  "DETECTED_INCORRECT"  — item is detected but keypoints suggest wrong position.
  "NOT_DETECTED"        — insufficient pose data or bbox is None.


KEYPOINT INDEX MAP  (POSE_KEYPOINTS)
--------------------------------------
A dict mapping human-readable landmark names to their MediaPipe Pose indices
(0–32). All indices have been swapped for horizontal mirroring so that
"left_wrist" refers to the wearer's left wrist on a horizontally flipped
camera feed.


SHARED GEOMETRY HELPERS
------------------------
  _get_pose(pose_landmarks)
    Extracts the first-person landmark list from a MediaPipe result object.
    Returns None if the result is empty or malformed.

  _lm_pixel(lm, img_h, img_w, min_visibility)
    Converts a single landmark to (px, py) pixel coordinates.
    Returns None if the landmark's visibility score is below the threshold,
    preventing low-confidence keypoints from generating false positives.

  _point_in_center_region(bbox, point, center_ratio)
    Returns True if a point falls within the inner center_ratio fraction of
    a bounding box. Used to reject PPE items that are adjacent to the body
    rather than worn on it (e.g. a glove resting on top of a hand).


VERIFICATION FUNCTIONS
----------------------

  verify_mask(mask_bbox, pose_landmarks, img_w, img_h, margin)
    Checks that the mask bbox covers the nose AND both mouth corners.
    All three landmarks must be visible (returns NOT_DETECTED if any is
    missing — refusal to guess prevents a chin-worn mask from passing).
    A fast early-reject fires if the bbox top edge starts below the nose
    bridge. Nose coverage uses a strict 0 px margin; mouth corners use the
    caller-supplied margin (default 10 px).

  verify_haircap(haircap_bbox, pose_landmarks, img_w, img_h, margin)
    Checks coverage of the head region using nose, both eyes, and both
    ears (5 landmarks). Requires at least 3 visible points to make a
    decision. Returns DETECTED_CORRECT if ≥ 60% of visible points are
    covered, DETECTED_INCORRECT if ≥ 30%, NOT_DETECTED otherwise.

  verify_gloves(gloves_detections, pose_landmarks, img_w, img_h, margin)
    Checks both hands independently. Primary requirement: the wrist
    landmark must be inside a glove bbox. Secondary bonus: if finger
    landmarks (pinky, index, thumb) are visible, at least half must also
    be covered. If fingers are not visible at distance, wrist coverage
    alone is sufficient to pass. Returns a dict:
      {"left": <status>, "right": <status>}
    Both sides must be DETECTED_CORRECT for the stage to advance.

  verify_boots(boots_detections, pose_landmarks, img_w, img_h, margin)
    Checks both feet independently using ankle, heel, and foot_index
    landmarks. Requires at least 2 visible points per foot. Returns a dict:
      {"left": <status>, "right": <status>}
    Both sides must be DETECTED_CORRECT for the stage to advance.

  verify_apron(apron_bbox, pose_landmarks, img_w, img_h, margin)
    Uses 4 check points to distinguish a worn apron from a held one:
      Point 1 — Left hip  (landmark)
      Point 2 — Right hip (landmark)
      Point 3 — Belly button: virtual midpoint between both hips
      Point 4 — Chest centre: virtual midpoint between both shoulders
    The chest centre is above the top edge of a handheld apron, so
    holding the apron at waist level fails this check. Passes if at
    least 3 of the 4 available points fall inside the bbox.

  verify_long_sleeves(long_sleeves_bbox, pose_landmarks, img_w, img_h, margin)
    Checks coverage of 6 arm landmarks: both shoulders, both elbows,
    both wrists. Requires at least 4 visible points. Returns
    DETECTED_CORRECT if ≥ 80% are covered, DETECTED_INCORRECT if ≥ 50%.
    Note: an arm-crossing detection variant was prototyped in v2.2 but
    reverted because it produced false passes; the simpler coverage ratio
    approach is more reliable in practice.

DEBUG VISUALISATION  (draw_verification_points)
------------------------------------------------
Draws colour-coded keypoint markers and single-letter labels onto a live
camera frame for use with the debug overlay (D key in main.py / test.py).

  Red     (M)  — mask points:    nose, mouth left, mouth right
  Magenta (H)  — haircap points: eyes, ears
  Cyan    (G)  — glove points:   wrists
  Purple  (B)  — boot points:    ankles, heels, foot indices
  Yellow  (A)  — apron points:   hips, virtual belly button (Ab),
                                  virtual chest centre (Ac)
  Orange  (S)  — sleeve points:  elbows


DEPENDENCIES
------------
  core/geometry.py   — bbox_contains_point, landmark_to_pixel
  cv2 (opencv-python) — circle and text drawing in draw_verification_points
"""

from core.geometry import bbox_contains_point, landmark_to_pixel
import cv2

# ============================================
# KEYPOINT INDICES (MediaPipe Pose - 33 points)
# ============================================

POSE_KEYPOINTS = {
    # Head keypoints (Swapped for Mirroring)
    "nose": 0,
    "left_eye_inner": 4,
    "right_eye_inner": 1,
    "left_eye": 5,
    "right_eye": 2,
    "left_eye_outer": 6,
    "right_eye_outer": 3,
    "left_ear": 8,
    "right_ear": 7,
    "mouth_left": 10,
    "mouth_right": 9,

    # Upper body keypoints (Swapped for Mirroring)
    "left_shoulder": 12,
    "right_shoulder": 11,
    "left_elbow": 14,
    "right_elbow": 13,
    "left_wrist": 16,
    "right_wrist": 15,

    # Hand keypoints (Swapped for Mirroring)
    "left_pinky": 18,
    "right_pinky": 17,
    "left_index": 20,
    "right_index": 19,
    "left_thumb": 22,
    "right_thumb": 21,

    # Lower body keypoints (Swapped for Mirroring)
    "left_hip": 24,
    "right_hip": 23,
    "left_knee": 26,
    "right_knee": 25,
    "left_ankle": 28,
    "right_ankle": 27,
    "left_heel": 30,
    "right_heel": 29,
    "left_foot_index": 32,
    "right_foot_index": 31,
}


# ============================================
# SHARED GEOMETRY HELPERS
# ============================================

def _get_pose(pose_landmarks):
    """
    Extract the single-person landmark list from a pose result object.
    Returns None if pose data is unavailable.
    """
    if not pose_landmarks:
        return None
    if not hasattr(pose_landmarks, 'pose_landmarks') or not pose_landmarks.pose_landmarks:
        return None
    return pose_landmarks.pose_landmarks[0]


def _lm_pixel(lm, img_h, img_w, min_visibility=0.5):
    """
    Convert a single landmark to pixel coords.
    Returns (px, py) or None if visibility is below threshold.
    """
    if hasattr(lm, 'visibility') and lm.visibility < min_visibility:
        return None
    return landmark_to_pixel(lm, (img_h, img_w))


def _point_in_center_region(bbox, point, center_ratio=0.70):
    """
    Return True if *point* falls within the inner *center_ratio* region of
    *bbox*.  Used to distinguish a PPE item worn on the body (keypoint near
    bbox center) from an item merely placed adjacent to the body (keypoint
    at bbox edge).

    Args:
        bbox:         (x1, y1, x2, y2)
        point:        (px, py)
        center_ratio: fraction of each dimension considered "central"
                      e.g. 0.70 means the middle 70% in both axes.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    pad_x = bw * (1.0 - center_ratio) / 2.0
    pad_y = bh * (1.0 - center_ratio) / 2.0
    px, py = point
    return (x1 + pad_x <= px <= x2 - pad_x and
            y1 + pad_y <= py <= y2 - pad_y)


# ============================================
# PPE VERIFICATION FUNCTIONS - POSE ONLY
# ============================================

def verify_mask(mask_bbox, pose_landmarks, img_w, img_h, margin=10):
    
    if mask_bbox is None:
        return "NOT_DETECTED"

    pose = _get_pose(pose_landmarks)
    if pose is None:
        return "NOT_DETECTED"

    nose_idx        = POSE_KEYPOINTS["nose"]
    mouth_left_idx  = POSE_KEYPOINTS["mouth_left"]
    mouth_right_idx = POSE_KEYPOINTS["mouth_right"]

    # ── Collect all three landmarks — all must be visible ────────────────
    points = {}
    for name, idx in [("nose",         nose_idx),
                      ("mouth_left",   mouth_left_idx),
                      ("mouth_right",  mouth_right_idx)]:
        if idx >= len(pose):
            return "NOT_DETECTED"
        pt = _lm_pixel(pose[idx], img_h, img_w, min_visibility=0.5)
        if pt is None:
            # Landmark not visible — cannot confirm coverage, refuse to pass
            return "NOT_DETECTED"
        points[name] = pt

    # ── Fast early-reject: bbox top must reach at or above the nose ──────
    x1, y1, x2, y2 = mask_bbox
    nose_py = points["nose"][1]
    if y1 > nose_py + 8:       # 8 px jitter tolerance
        return "DETECTED_INCORRECT"

    # ── Strict coverage: ALL three points must be inside the bbox ────────
    # Nose uses 0 px margin — it must be unambiguously inside.
    # Mouth corners use the caller's margin (default 10 px) for normal tolerance.
    nose_ok        = bbox_contains_point(mask_bbox, points["nose"],        margin=0)
    mouth_left_ok  = bbox_contains_point(mask_bbox, points["mouth_left"],  margin)
    mouth_right_ok = bbox_contains_point(mask_bbox, points["mouth_right"], margin)

    if nose_ok and mouth_left_ok and mouth_right_ok:
        return "DETECTED_CORRECT"

    # At least one point missed — tell the system the mask is wrong position
    return "DETECTED_INCORRECT"


def verify_haircap(haircap_bbox, pose_landmarks, img_w, img_h, margin=15):
    
    if haircap_bbox is None:
        return "NOT_DETECTED"
    
    # Check if pose_landmarks is a MediaPipe result object
    if not pose_landmarks:
        return "NOT_DETECTED"
    
    # Access pose landmarks from MediaPipe result
    if not hasattr(pose_landmarks, 'pose_landmarks') or not pose_landmarks.pose_landmarks:
        return "NOT_DETECTED"
    
    pose = pose_landmarks.pose_landmarks[0]  # Get first person's landmarks
    
    # Critical points for haircap verification
    # Eyes and ears define the upper head region
    # Nose is used as reference point
    required_points = [
        POSE_KEYPOINTS["nose"],  # Top of exposed face
        POSE_KEYPOINTS["left_eye"],
        POSE_KEYPOINTS["right_eye"],
        POSE_KEYPOINTS["left_ear"],
        POSE_KEYPOINTS["right_ear"],
    ]
    
    covered_count = 0
    valid_points = 0
    
    for idx in required_points:
        if idx >= len(pose):
            continue
        
        lm = pose[idx]
        
        # Skip if visibility is too low
        if hasattr(lm, 'visibility') and lm.visibility < 0.5:
            continue
        
        valid_points += 1
        px, py = landmark_to_pixel(lm, (img_h, img_w))
        
        if bbox_contains_point(haircap_bbox, (px, py), margin):
            covered_count += 1
    
    # Need at least 3 valid points to make a decision
    if valid_points < 3:
        return "NOT_DETECTED"
    
    # At least 60% of visible points should be covered
    coverage_ratio = covered_count / valid_points
    
    if coverage_ratio >= 0.6:
        return "DETECTED_CORRECT"
    elif coverage_ratio >= 0.3:
        return "DETECTED_INCORRECT"
    else:
        return "NOT_DETECTED"


def verify_gloves(gloves_detections, pose_landmarks, img_w, img_h, margin=20):

    result = {
        "left": "NOT_DETECTED",
        "right": "NOT_DETECTED"
    }
    
    if not gloves_detections or len(gloves_detections) == 0:
        return result
    
    # Check if pose_landmarks is a MediaPipe result object
    if not pose_landmarks:
        return result
    
    # Access pose landmarks from MediaPipe result
    if not hasattr(pose_landmarks, 'pose_landmarks') or not pose_landmarks.pose_landmarks:
        return result
    
    pose = pose_landmarks.pose_landmarks[0]  # Get first person's landmarks
    
    # Define wrist and hand reference points for each side
    # We use wrist as primary + optional finger indices if visible
    hands_keypoints = {
        "left": {
            "primary": [POSE_KEYPOINTS["left_wrist"]],
            "secondary": [POSE_KEYPOINTS["left_pinky"], POSE_KEYPOINTS["left_index"], POSE_KEYPOINTS["left_thumb"]]
        },
        "right": {
            "primary": [POSE_KEYPOINTS["right_wrist"]],
            "secondary": [POSE_KEYPOINTS["right_pinky"], POSE_KEYPOINTS["right_index"], POSE_KEYPOINTS["right_thumb"]]
        }
    }
    
    for side, keypoints_dict in hands_keypoints.items():
        best_match = "NOT_DETECTED"
        
        for glove_bbox in gloves_detections:
            # Check primary keypoints (wrist - most important)
            primary_covered = 0
            primary_valid = 0
            
            for idx in keypoints_dict["primary"]:
                if idx >= len(pose):
                    continue
                
                lm = pose[idx]
                
                # Check visibility
                if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                    continue
                
                primary_valid += 1
                px, py = landmark_to_pixel(lm, (img_h, img_w))
                
                if bbox_contains_point(glove_bbox, (px, py), margin):
                    primary_covered += 1
            
            # Check secondary keypoints (fingers - bonus points)
            secondary_covered = 0
            secondary_valid = 0
            
            for idx in keypoints_dict["secondary"]:
                if idx >= len(pose):
                    continue
                
                lm = pose[idx]
                
                # Check visibility
                if hasattr(lm, 'visibility') and lm.visibility < 0.4:
                    continue
                
                secondary_valid += 1
                px, py = landmark_to_pixel(lm, (img_h, img_w))
                
                if bbox_contains_point(glove_bbox, (px, py), margin):
                    secondary_covered += 1
            
            # Evaluation logic:
            # CRITICAL: Wrist must be covered (primary)
            # BONUS: Finger points add confidence (secondary)
            
            if primary_valid > 0 and primary_covered > 0:
                # Wrist is covered - this is the main requirement
                if secondary_valid > 0:
                    # We can see fingers too - check if they're covered
                    secondary_ratio = secondary_covered / secondary_valid
                    if secondary_ratio >= 0.5:  # At least half of visible fingers covered
                        best_match = "DETECTED_CORRECT"
                        break
                    else:
                        # Wrist covered but fingers not - might be incorrect position
                        if best_match == "NOT_DETECTED":
                            best_match = "DETECTED_INCORRECT"
                else:
                    # Can't see fingers clearly (distance/occlusion) - wrist coverage is enough
                    best_match = "DETECTED_CORRECT"
                    break
            elif secondary_valid > 0 and secondary_covered >= 2:
                # Wrist not visible but multiple fingers are covered
                if best_match == "NOT_DETECTED":
                    best_match = "DETECTED_INCORRECT"
        
        result[side] = best_match
    
    return result


def verify_boots(boots_detections, pose_landmarks, img_w, img_h, margin=25):

    result = {"left": "NOT_DETECTED", "right": "NOT_DETECTED"}

    if not boots_detections:
        return result

    pose = _get_pose(pose_landmarks)
    if pose is None:
        return result

    feet_keypoints = {
        "left":  [POSE_KEYPOINTS["left_ankle"],
                  POSE_KEYPOINTS["left_heel"],
                  POSE_KEYPOINTS["left_foot_index"]],
        "right": [POSE_KEYPOINTS["right_ankle"],
                  POSE_KEYPOINTS["right_heel"],
                  POSE_KEYPOINTS["right_foot_index"]],
    }

    for side, indices in feet_keypoints.items():
        best_match = "NOT_DETECTED"

        for boot_bbox in boots_detections:
            covered = 0
            valid   = 0

            for idx in indices:
                if idx >= len(pose):
                    continue
                pt = _lm_pixel(pose[idx], img_h, img_w, min_visibility=0.5)
                if pt is None:
                    continue
                valid += 1
                if bbox_contains_point(boot_bbox, pt, margin):
                    covered += 1

            if valid < 2:
                continue

            if covered >= 2:
                best_match = "DETECTED_CORRECT"
                break
            elif covered >= 1:
                best_match = "DETECTED_INCORRECT"

        result[side] = best_match

    return result


def verify_apron(apron_bbox, pose_landmarks, img_w, img_h, margin=30):

    if apron_bbox is None:
        return "NOT_DETECTED"

    pose = _get_pose(pose_landmarks)
    if pose is None:
        return "NOT_DETECTED"

    # ── Collect raw landmark positions ───────────────────────────────────
    l_hip_idx = POSE_KEYPOINTS["left_hip"]
    r_hip_idx = POSE_KEYPOINTS["right_hip"]
    l_sho_idx = POSE_KEYPOINTS["left_shoulder"]
    r_sho_idx = POSE_KEYPOINTS["right_shoulder"]

    if max(l_hip_idx, r_hip_idx, l_sho_idx, r_sho_idx) >= len(pose):
        return "NOT_DETECTED"

    l_hip_pt = _lm_pixel(pose[l_hip_idx], img_h, img_w, min_visibility=0.5)
    r_hip_pt = _lm_pixel(pose[r_hip_idx], img_h, img_w, min_visibility=0.5)
    l_sho_pt = _lm_pixel(pose[l_sho_idx], img_h, img_w, min_visibility=0.4)
    r_sho_pt = _lm_pixel(pose[r_sho_idx], img_h, img_w, min_visibility=0.4)

    # Need at least both hips to proceed
    if l_hip_pt is None and r_hip_pt is None:
        return "NOT_DETECTED"

    # ── Build the 4 check points ──────────────────────────────────────────
    check_points = []

    # Points 1 & 2: actual hip landmarks
    if l_hip_pt is not None:
        check_points.append(("left_hip",   l_hip_pt))
    if r_hip_pt is not None:
        check_points.append(("right_hip",  r_hip_pt))

    # Point 3: belly button — midpoint of the two hips
    if l_hip_pt is not None and r_hip_pt is not None:
        belly_x = (l_hip_pt[0] + r_hip_pt[0]) // 2
        belly_y = (l_hip_pt[1] + r_hip_pt[1]) // 2
        check_points.append(("belly_button", (belly_x, belly_y)))

    # Point 4: chest centre — midpoint of the two shoulders
    # Use whichever shoulders are visible; fall back gracefully if neither is.
    sho_pts = [p for p in (l_sho_pt, r_sho_pt) if p is not None]
    if sho_pts:
        chest_x = sum(p[0] for p in sho_pts) // len(sho_pts)
        chest_y = sum(p[1] for p in sho_pts) // len(sho_pts)
        check_points.append(("chest_centre", (chest_x, chest_y)))

    if len(check_points) < 2:
        return "NOT_DETECTED"

    # ── Count how many points fall inside the bbox ───────────────────────
    covered = sum(
        1 for _, pt in check_points
        if bbox_contains_point(apron_bbox, pt, margin)
    )
    total = len(check_points)

    # Require at least 3 out of 4 (or all if fewer than 4 are available)
    required = min(3, total)

    if covered >= required:
        return "DETECTED_CORRECT"
    if covered >= 1:
        return "DETECTED_INCORRECT"
    return "NOT_DETECTED"


def verify_long_sleeves(long_sleeves_bbox, pose_landmarks, img_w, img_h, margin=20):

    if long_sleeves_bbox is None:
        return "NOT_DETECTED"

    if not pose_landmarks:
        return "NOT_DETECTED"

    if not hasattr(pose_landmarks, 'pose_landmarks') or not pose_landmarks.pose_landmarks:
        return "NOT_DETECTED"

    pose = pose_landmarks.pose_landmarks[0]

    required_points = [
        POSE_KEYPOINTS["left_shoulder"],
        POSE_KEYPOINTS["right_shoulder"],
        POSE_KEYPOINTS["left_elbow"],
        POSE_KEYPOINTS["right_elbow"],
        POSE_KEYPOINTS["left_wrist"],
        POSE_KEYPOINTS["right_wrist"],
    ]

    covered_count = 0
    valid_count   = 0

    for idx in required_points:
        if idx >= len(pose):
            continue

        lm = pose[idx]

        if hasattr(lm, 'visibility') and lm.visibility < 0.5:
            continue

        valid_count += 1
        px, py = landmark_to_pixel(lm, (img_h, img_w))

        if bbox_contains_point(long_sleeves_bbox, (px, py), margin):
            covered_count += 1

    if valid_count < 4:
        return "NOT_DETECTED"

    coverage_ratio = covered_count / valid_count

    if coverage_ratio >= 0.8:
        return "DETECTED_CORRECT"
    elif coverage_ratio >= 0.5:
        return "DETECTED_INCORRECT"
    else:
        return "NOT_DETECTED"


# ============================================
# DEBUGGING VISUALIZATION
# ============================================

def draw_verification_points(frame, pose_landmarks, img_w, img_h):
    """
    Draw all POSE keypoints used for PPE verification.
    """
    if not pose_landmarks:
        return

    if hasattr(pose_landmarks, 'pose_landmarks') and pose_landmarks.pose_landmarks:
        pose = pose_landmarks.pose_landmarks[0]
    elif isinstance(pose_landmarks, list) and len(pose_landmarks) > 0:
        pose = pose_landmarks[0]
    else:
        return

    colors = {
        "mask":    (0,   0,   255),   # Red
        "haircap": (255, 0,   255),   # Magenta
        "gloves":  (255, 255, 0),     # Cyan
        "boots":   (128, 0,   128),   # Purple
        "apron":   (0,   255, 255),   # Yellow
        "sleeves": (0,   165, 255),   # Orange
    }

    def _draw(indices, color, label):
        for idx in indices:
            if idx >= len(pose):
                continue
            lm = pose[idx]
            if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                continue
            px, py = landmark_to_pixel(lm, (img_h, img_w))
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.putText(frame, label, (px + 5, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    _draw([POSE_KEYPOINTS["nose"],
           POSE_KEYPOINTS["mouth_left"],
           POSE_KEYPOINTS["mouth_right"]], colors["mask"], "M")

    _draw([POSE_KEYPOINTS["left_eye"],
           POSE_KEYPOINTS["right_eye"],
           POSE_KEYPOINTS["left_ear"],
           POSE_KEYPOINTS["right_ear"]], colors["haircap"], "H")

    _draw([POSE_KEYPOINTS["left_wrist"],
           POSE_KEYPOINTS["right_wrist"]], colors["gloves"], "G")

    _draw([POSE_KEYPOINTS["left_ankle"],
           POSE_KEYPOINTS["right_ankle"],
           POSE_KEYPOINTS["left_heel"],
           POSE_KEYPOINTS["right_heel"],
           POSE_KEYPOINTS["left_foot_index"],
           POSE_KEYPOINTS["right_foot_index"]], colors["boots"], "B")

    # Apron: hips + virtual belly button + virtual chest centre (v2.3)
    _draw([POSE_KEYPOINTS["left_hip"],
           POSE_KEYPOINTS["right_hip"]], colors["apron"], "A")

    # Draw virtual belly button and chest centre if landmarks are available
    _l_hip = pose[POSE_KEYPOINTS["left_hip"]]  if POSE_KEYPOINTS["left_hip"]  < len(pose) else None
    _r_hip = pose[POSE_KEYPOINTS["right_hip"]] if POSE_KEYPOINTS["right_hip"] < len(pose) else None
    _l_sho = pose[POSE_KEYPOINTS["left_shoulder"]]  if POSE_KEYPOINTS["left_shoulder"]  < len(pose) else None
    _r_sho = pose[POSE_KEYPOINTS["right_shoulder"]] if POSE_KEYPOINTS["right_shoulder"] < len(pose) else None

    def _vis_ok(lm, t=0.4):
        return lm is not None and not (hasattr(lm, 'visibility') and lm.visibility < t)

    if _vis_ok(_l_hip) and _vis_ok(_r_hip):
        _lhp = landmark_to_pixel(_l_hip, (img_h, img_w))
        _rhp = landmark_to_pixel(_r_hip, (img_h, img_w))
        _bx = (_lhp[0] + _rhp[0]) // 2
        _by = (_lhp[1] + _rhp[1]) // 2
        cv2.circle(frame, (_bx, _by), 5, colors["apron"], -1)
        cv2.putText(frame, "Ab", (_bx + 5, _by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors["apron"], 1)

    _sho_pts = [landmark_to_pixel(lm, (img_h, img_w))
                for lm in (_l_sho, _r_sho) if _vis_ok(lm)]
    if _sho_pts:
        _cx = sum(p[0] for p in _sho_pts) // len(_sho_pts)
        _cy = sum(p[1] for p in _sho_pts) // len(_sho_pts)
        cv2.circle(frame, (_cx, _cy), 5, colors["apron"], -1)
        cv2.putText(frame, "Ac", (_cx + 5, _cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors["apron"], 1)

    _draw([POSE_KEYPOINTS["left_elbow"],
           POSE_KEYPOINTS["right_elbow"]], colors["sleeves"], "S")
