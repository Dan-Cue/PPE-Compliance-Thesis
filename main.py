"""
main.py — PPE Sequential Verification Entry Point
==================================================

OVERVIEW
--------
The top-level application file for the PPE (Personal Protective Equipment)
sequential verification system. It wires together every subsystem — camera,
YOLO detector, MediaPipe pose estimator, optional classifier, audio feedback,
GPIO hardware controller, and the configuration panel — then runs the main
loop that drives real-time PPE checking on a Raspberry Pi 5.


STARTUP SEQUENCE
----------------
A loading screen (ui/loading.py) is displayed while each subsystem
initialises in order. If any critical step fails, the loading screen shows
the error and halts before the main loop begins.

  1. YOLO Detector      — loads the YOLO model for PPE object detection.
  2. MediaPipe Pose     — loads the pose estimator for keypoint extraction.
  3. Classifier         — optional second-pass verifier; skipped if
                          config.USE_CLASSIFIER is False.
  4. Audio & Hardware   — initialises AudioFeedback and HardwareController.
  5. PPE Checker        — creates the SequentialPPEChecker state machine.
  6. Camera             — opens the V4L2 capture device and sets resolution.


MAIN LOOP
---------
Each iteration reads one camera frame, applies the processing pipeline,
and renders a composite output frame.

Frame processing (runs every config.FRAME_SKIP frames):

  Interleaved pipeline
    To keep the loop fast on the Pi, YOLO detection and MediaPipe pose
    estimation alternate on successive processed frames rather than both
    running every frame:
      Odd processed frame  → YOLO detection → updates `grouped` detections
      Even processed frame → Pose estimation → updates `pose_result`
    Both results are cached so the renderer and verifier always have
    the most recent data from each model.

  Verification
    verify_current_stage_only() checks only the PPE item for the current
    stage of the SequentialPPEChecker state machine. It runs the
    appropriate geometric verifier (mask, haircap, gloves, boots, apron,
    or long sleeves) against the cached pose keypoints.

  Classifier (optional)
    If config.USE_CLASSIFIER is enabled and YOLO + keypoint checks both
    pass, each active detection is sent through the classifier model for
    a final confidence check before the stage advances.

  Stage advancement
    checker.process_verification() is called with the yolo, keypoint,
    and classifier results. On advancement, audio is interrupted and
    the stage tracker resets the announcement timers.

  Completion
    When all required PPE stages are verified, the relay is activated
    (door unlocked), a "compliant" audio announcement plays, and a
    countdown timer begins. After config.COMPLETION_AUTO_RESET seconds
    the system resets automatically for the next person.

  No-detection timeout
    If no PPE or person is detected for more than NO_DETECTION_TIMEOUT
    (5 seconds), the system resets: relay locked, audio interrupted,
    checker and pose state cleared.


STATE VARIABLES
---------------
  show_debug    — mirrors config.ENABLE_DEBUG_VIZ; toggled live with D key.
  paused        — freezes all processing while keeping the display alive.
  grouped       — cached dict of the latest YOLO detections by PPE class.
  pose_result   — cached MediaPipe pose result from the latest pose frame.
  run_yolo_next — interleaving flag; alternates YOLO and pose each frame.
  completion_time          — timestamp when the COMPLETE stage was reached.
  last_announced_stage     — prevents duplicate audio announcements.
  last_missing_announce_time — rate-limits "missing PPE" announcements to
                               once per MISSING_ANNOUNCE_INTERVAL (10 s).
  last_detection_time      — updated whenever PPE or a person is seen;
                             drives the no-detection reset timeout.
  _last_config_close       — timestamp of the last config panel close;
                             used for a 700 ms re-open debounce.


KEYBOARD CONTROLS
-----------------
  Q      — quit the application.
  R      — manual reset (locks door, clears all state).
  A      — toggle audio on / off.
  D      — toggle debug visualisation overlay.
  G      — open the configuration panel.
  SPACE  — pause / resume processing.


TOUCH / MOUSE CONTROL
---------------------
Tapping the gear icon drawn by UIRenderer opens the configuration panel.
A 700 ms debounce after the panel closes prevents accidental re-opens
caused by the release event of the closing tap.


CONFIGURATION PANEL INTEGRATION
--------------------------------
When the config panel is saved:
  1. config module is reloaded via importlib so all attribute reads in
     the loop immediately reflect the new values.
  2. show_debug is re-synced from config.ENABLE_DEBUG_VIZ.
  3. audio.enabled is re-synced from config.ENABLE_AUDIO.
  4. audio.sync_volume() applies the new AUDIO_VOLUME immediately.
  5. If USE_CLASSIFIER was just enabled and no classifier instance
     exists, one is loaded on demand.
  6. SequentialPPEChecker is rebuilt so changes to PPE requirements or
     stability frame counts take effect without a restart.
  After either save or cancel, the mouse callback is restored, detection
  state is cleared, and verification resumes from a clean slate.


HELPER FUNCTIONS
----------------
  group_detections_by_class(detections)
    Reorganises the flat YOLO detection list into a dict keyed by PPE
    class. For single-instance items (mask, haircap, apron, long_sleeves)
    only the highest-confidence detection is kept. For paired items
    (gloves, boots) all detections are collected in a list.

  verify_current_stage_only(checker, grouped, pose_result, img_w, img_h)
    Runs the geometric verifier for the checker's current stage only.
    Returns (yolo_ok, kp_ok, ppe_type, active_detections).
    Paired items (gloves, boots) require two detections and both must
    pass their respective left/right keypoint checks.


DEPENDENCIES
------------
  cv2 (opencv-python)          — camera capture, display, keyboard input
  mediapipe                    — pose keypoint estimation
  config                       — runtime parameters (reloaded on save)
  ui / ui.py                   — UIConfig, UIRenderer (composite frame builder)
  ui/loading.py                — LoadingScreen
  hardware.py                  — HardwareController (GPIO relay)
  config_panel.py              — ConfigPanel (touch settings UI)
  core/detector.py             — PPEDetector (YOLO wrapper)
  core/pose_estimator.py       — PoseEstimator (MediaPipe wrapper)
  core/classifier.py           — PPEClassifier (optional second-pass model)
  core/sequential_checker.py   — SequentialPPEChecker, PPECheckStage
  core/verifier.py             — per-item geometric verifiers
  core/audio_feedback.py       — AudioFeedback
"""

import cv2
import time
import importlib
import config

# ── Only lightweight imports at module level so loading screen appears fast ───
from ui      import UIConfig, UIRenderer
from ui.loading import LoadingScreen
from hardware   import HardwareController
from config_panel import ConfigPanel


# ==============================================================================
#  Detection helpers
# ==============================================================================

def group_detections_by_class(detections):
    grouped = {"apron": None, "boots": [], "gloves": [],
               "haircap": None, "long_sleeves": None, "mask": None}
    for det in detections:
        cls = det["class"]
        if cls in ("apron", "haircap", "long_sleeves", "mask"):
            if grouped[cls] is None or det["confidence"] > grouped[cls]["confidence"]:
                grouped[cls] = det
        elif cls in ("boots", "gloves"):
            grouped[cls].append(det)
    return grouped


def verify_current_stage_only(checker, grouped, pose_result, img_w, img_h):
    from core.verifier import (verify_mask, verify_haircap, verify_gloves,
                               verify_boots, verify_apron, verify_long_sleeves)
    from core.sequential_checker import PPECheckStage

    stage = checker.current_stage
    if stage == PPECheckStage.COMPLETE:
        return True, True, None, []

    ppe_type = checker.get_ppe_type_for_stage(stage)
    yolo_ok = kp_ok = False
    active_detections = []

    if stage in (PPECheckStage.HAIRCAP, PPECheckStage.MASK,
                 PPECheckStage.LONG_SLEEVES, PPECheckStage.APRON):
        det = grouped.get(ppe_type)
        yolo_ok = det is not None
        if yolo_ok:
            VERIFIERS = {
                PPECheckStage.HAIRCAP:      (verify_haircap,     config.MARGIN_HAIRCAP),
                PPECheckStage.MASK:         (verify_mask,         config.MARGIN_MASK),
                PPECheckStage.LONG_SLEEVES: (verify_long_sleeves, config.MARGIN_LONG_SLEEVES),
                PPECheckStage.APRON:        (verify_apron,        config.MARGIN_APRON),
            }
            fn, margin = VERIFIERS[stage]
            kp_ok = (fn(det["bbox"], pose_result, img_w, img_h, margin=margin)
                     == "DETECTED_CORRECT")
            active_detections = [det]

    elif stage == PPECheckStage.GLOVES:
        gloves_list = grouped.get("gloves", [])
        yolo_ok = len(gloves_list) >= 2
        if yolo_ok:
            res = verify_gloves(
                [g["bbox"] for g in gloves_list], pose_result, img_w, img_h,
                margin=config.MARGIN_GLOVES)
            kp_ok = (res.get("left") == "DETECTED_CORRECT" and
                     res.get("right") == "DETECTED_CORRECT")
            active_detections = gloves_list[:2]  # Return both detections

    elif stage == PPECheckStage.BOOTS:
        boots_list = grouped.get("boots", [])
        yolo_ok = len(boots_list) >= 2
        if yolo_ok:
            res = verify_boots(
                [b["bbox"] for b in boots_list], pose_result, img_w, img_h,
                margin=config.MARGIN_BOOTS)
            kp_ok = (res.get("left") == "DETECTED_CORRECT" and
                     res.get("right") == "DETECTED_CORRECT")
            active_detections = boots_list[:2]  # Return both detections

    return yolo_ok, kp_ok, ppe_type, active_detections


# ==============================================================================
#  Main
# ==============================================================================

def main():
    print("=" * 70)
    print("       PPE SEQUENTIAL VERIFICATION")
    print("=" * 70)

    ui_cfg = UIConfig()
    loader = LoadingScreen(ui_cfg)

    try:
        loader.set_step("YOLO Detector")
        from core.detector import PPEDetector
        detector = PPEDetector(config.YOLO_MODEL_PATH)
        print("v YOLO detector loaded")
        loader.mark_ok()

        loader.set_step("MediaPipe Pose")
        from core.pose_estimator import PoseEstimator
        pose = PoseEstimator(config.MEDIAPIPE_DIR)
        print("v MediaPipe Pose loaded")
        loader.mark_ok()

        loader.set_step("Classifier")
        classifier = None
        if config.USE_CLASSIFIER:
            from core.classifier import PPEClassifier
            classifier = PPEClassifier(config.CLASSIFIER_MODEL_PATH)
            print("v Classifier loaded")
            loader.mark_ok()
        else:
            print("! Classifier disabled")
            loader.mark_skip("disabled")

        loader.set_step("Audio & Hardware")
        from core.audio_feedback import AudioFeedback
        audio = AudioFeedback(config)
        hw    = HardwareController()
        loader.mark_ok()

        loader.set_step("PPE Checker")
        from core.sequential_checker import SequentialPPEChecker, PPECheckStage
        checker = SequentialPPEChecker(config)
        loader.mark_ok()

        loader.set_step("Camera")
        cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            raise RuntimeError("Camera failed to open")
        print("v Camera ready")
        loader.mark_ok()

    except Exception as e:
        print(f"x Startup failed: {e}")
        loader.mark_fail(str(e))
        loader.wait()
        return

    loader.finish()

    from core.verifier import draw_verification_points

    win_name = "PPE Verification System"
    renderer = UIRenderer(ui_cfg)

    # Create window first (required before setWindowProperty / setMouseCallback)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if ui_cfg.FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win_name, ui_cfg.OUTPUT_W, ui_cfg.OUTPUT_H)

    print("\nControls:  Q quit | R reset | A audio | D debug | G config | SPACE pause")
    print("=" * 70 + "\n")

    # ── Touch / mouse state ────────────────────────────────────────────────────
    _open_config = [False]   # use list so lambda can mutate it

    def _mouse_cb(event, x, y, flags, param):
        """Detect tap on gear button; everything else is ignored here."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Debounce: ignore taps for 700 ms after config panel closed
            if time.time() - _last_config_close < 0.70:
                return
            gx1, gy1, gx2, gy2 = UIRenderer.GEAR_RECT
            if gx1 <= x <= gx2 and gy1 <= y <= gy2:
                _open_config[0] = True

    cv2.setMouseCallback(win_name, _mouse_cb)

    # Config panel instance (created once, reused)
    _cfg_panel = ConfigPanel("config.py")
    _last_config_close = 0.0     # time.time() when panel last closed (debounce)

    # ── State ─────────────────────────────────────────────────────────────────
    show_debug  = config.ENABLE_DEBUG_VIZ
    paused      = False
    frame_count = fps = 0
    fps_time    = time.time()

    NO_DETECTION_TIMEOUT      = 5.0
    MISSING_ANNOUNCE_INTERVAL = 10.0
    # NOTE: COMPLETION_AUTO_RESET is read from config.COMPLETION_AUTO_RESET each rame so it updates immediately after a config save without needing a restart.


    last_detection_time        = time.time()
    completion_time            = None
    last_announced_stage       = None
    last_missing_announce_time = 0

    yolo_ok = kp_ok = False
    active_detections = []
    pose_result = None
    
    # Cache for interleaving
    grouped = {"apron": None, "boots": [], "gloves": [], "haircap": None, "long_sleeves": None, "mask": None}
    run_yolo_next = True 

    from core.sequential_checker import PPECheckStage

    # ── Loop ──────────────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = UIRenderer.rotate_frame(frame, ui_cfg.CAMERA_ROTATION)

        frame_count  += 1
        current_time  = time.time()

        if current_time - fps_time >= 1.0:
            fps         = frame_count / (current_time - fps_time)
            frame_count = 0
            fps_time    = current_time

        if frame_count % config.FRAME_SKIP != 0:
            composite = renderer.build_frame(
                frame, checker, fps, audio.enabled,
                None, None, yolo_ok, kp_ok, active_detections,
                show_debug, pose_result, paused)
            cv2.imshow(win_name, composite)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        det_countdown = comp_countdown = None

        if not paused:
            img_h, img_w = frame.shape[:2]

            # ── HACK: INTERLEAVING PIPELINE ──────────────────────────────────
            # Alternates heavy processing: Frame 1 = YOLO, Frame 2 = MediaPipe
            if run_yolo_next:
                detections = detector.detect(frame, conf_threshold=config.YOLO_CONFIDENCE)
                grouped    = group_detections_by_class(detections)

                has_ppe = any([grouped["mask"], grouped["haircap"],
                               grouped["long_sleeves"], grouped["apron"],
                               grouped["gloves"], grouped["boots"]])
                if has_ppe:
                    last_detection_time = current_time
                
                run_yolo_next = False
            else:
                pose_result, _, _ = pose.process(frame)
                
                # Keep system awake if it sees a person even without PPE
                if pose_result and hasattr(pose_result, 'pose_landmarks') and pose_result.pose_landmarks:
                    last_detection_time = current_time

                run_yolo_next = True
            # ─────────────────────────────────────────────────────────────────

            time_since_det = current_time - last_detection_time
            if time_since_det > NO_DETECTION_TIMEOUT:
                if not checker.is_complete() or completion_time is None:
                    hw.set_low(); audio.interrupt(); checker.reset(); pose.reset_lock()
                    completion_time = last_announced_stage = None
                    last_missing_announce_time = 0
                    yolo_ok = kp_ok = False
                    active_detections = []
                    pose_result = None
                    grouped = {"apron": None, "boots": [], "gloves": [], "haircap": None, "long_sleeves": None, "mask": None}
            elif time_since_det > 2.0:
                det_countdown = NO_DETECTION_TIMEOUT - time_since_det

            if not checker.is_complete():
                completion_time = None

                yolo_ok, kp_ok, ppe_type, active_detections = verify_current_stage_only(
                    checker, grouped, pose_result, img_w, img_h)

                classifier_ok = None
                if (config.USE_CLASSIFIER and classifier and yolo_ok and kp_ok and
                        checker.should_run_classifier and active_detections):
                    try:
                        all_correct = True
                        for det in active_detections:
                            r = classifier.verify_detection(
                                frame, det["bbox"], ppe_type,
                                confidence_threshold=config.CLASSIFIER_CONFIDENCE)
                            if not r["is_correct"]:
                                all_correct = False
                        classifier_ok = all_correct
                    except Exception as e:
                        print(f"Classifier error: {e}")
                        classifier_ok = True

                advanced  = checker.process_verification(yolo_ok, kp_ok, classifier_ok)
                cur_stage = checker.current_stage

                human_in_frame = (pose_result is not None and
                                  hasattr(pose_result, 'pose_landmarks') and 
                                  pose_result.pose_landmarks and
                                  len(pose_result.pose_landmarks) > 0)

                if advanced:
                    audio.interrupt()
                    last_announced_stage = cur_stage
                    last_missing_announce_time = 0
                elif cur_stage != last_announced_stage:
                    last_announced_stage = cur_stage
                    last_missing_announce_time = 0

                if (human_in_frame and not yolo_ok and not checker.is_complete() and
                        current_time - last_missing_announce_time >= MISSING_ANNOUNCE_INTERVAL):
                    last_missing_announce_time = current_time
                    if ppe_type:
                        audio.announce(f"missing_{ppe_type}")

            else:
                if completion_time is None:
                    completion_time = current_time
                    hw.set_high()
                    if audio.enabled:
                        audio.announce("compliant", force=True)

                elapsed_since = current_time - completion_time
                comp_countdown = max(0.0, config.COMPLETION_AUTO_RESET - elapsed_since)

                if elapsed_since > config.COMPLETION_AUTO_RESET:
                    hw.set_low(); audio.interrupt(); checker.reset(); pose.reset_lock()
                    completion_time = last_announced_stage = None
                    last_detection_time = current_time
                    last_missing_announce_time = 0
                    yolo_ok = kp_ok = False
                    active_detections = []
                    pose_result = None
                    grouped = {"apron": None, "boots": [], "gloves": [], "haircap": None, "long_sleeves": None, "mask": None}

        composite = renderer.build_frame(
            frame, checker, fps, audio.enabled,
            det_countdown, comp_countdown,
            yolo_ok, kp_ok, active_detections,
            show_debug, pose_result, paused)
        cv2.imshow(win_name, composite)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('a'):
            audio.enabled = not audio.enabled
            print(f"Audio: {'ON' if audio.enabled else 'OFF'}")
        elif key == ord('r'):
            hw.set_low(); audio.interrupt(); checker.reset(); pose.reset_lock()
            completion_time = last_announced_stage = None
            last_detection_time = current_time
            last_missing_announce_time = 0
            yolo_ok = kp_ok = False
            active_detections = []
            pose_result = None
            grouped = {"apron": None, "boots": [], "gloves": [], "haircap": None, "long_sleeves": None, "mask": None}
            print("Manual reset")
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('g') or _open_config[0]:
            # ── Open config panel ──────────────────────────────────────────
            _open_config[0] = False
            print("\n⚙  Opening configuration panel…")

            # Show a brief "pausing" overlay so user gets feedback fast
            import numpy as _np
            _freeze = _np.full((ui_cfg.OUTPUT_H, ui_cfg.OUTPUT_W, 3),
                               (14, 16, 22), dtype=_np.uint8)
            cv2.putText(_freeze, "Opening config...",
                        (ui_cfg.OUTPUT_W // 2 - 120, ui_cfg.OUTPUT_H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 200, 110), 2, cv2.LINE_AA)
            cv2.imshow(win_name, _freeze)
            cv2.waitKey(1)

            # Suspend audio and GPIO during config
            audio.interrupt()
            hw.set_low()

            saved = _cfg_panel.open(win_name, ui_cfg.OUTPUT_W, ui_cfg.OUTPUT_H, config)
            _last_config_close = time.time()   # debounce starts now

            if saved:
                # Reload config module so all new values are live
                importlib.reload(config)

                # ── Sync everything that copied config values at startup ──
                # 1. show_debug was set once from config.ENABLE_DEBUG_VIZ
                show_debug = config.ENABLE_DEBUG_VIZ

                # 2. audio.enabled copied config.ENABLE_AUDIO at construction
                audio.enabled = config.ENABLE_AUDIO

                # 3. Sync audio volume immediately
                audio.sync_volume()

                # 4. classifier may have been None if USE_CLASSIFIER was off
                #    at startup — load it now if the user just enabled it
                if config.USE_CLASSIFIER and classifier is None:
                    try:
                        from core.classifier import PPEClassifier
                        classifier = PPEClassifier(config.CLASSIFIER_MODEL_PATH)
                        print("✓  Classifier loaded on demand")
                    except Exception as e:
                        print(f"⚠  Classifier load failed: {e}")
                        config.USE_CLASSIFIER = False

                # 5. Rebuild checker in case PPE requirements / stability changed
                from core.sequential_checker import SequentialPPEChecker
                checker = SequentialPPEChecker(config)
                print("✓  Config applied — checker rebuilt")
            else:
                print("  Config unchanged")

            # Restore mouse callback (config panel unhooked it)
            cv2.setMouseCallback(win_name, _mouse_cb)

            # Reset detection state so we start fresh
            completion_time = last_announced_stage = None
            last_detection_time = time.time()
            last_missing_announce_time = 0
            yolo_ok = kp_ok = False
            active_detections = []
            pose_result = None
            grouped = {"apron": None, "boots": [], "gloves": [],
                       "haircap": None, "long_sleeves": None, "mask": None}
            run_yolo_next = True
            print("⚙  Resuming verification…\n")

    cap.release()
    pose.close()
    audio.stop()
    hw.cleanup()
    cv2.destroyAllWindows()
    print("\nSystem shutdown")


if __name__ == "__main__":
    main()
