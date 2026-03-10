"""
test.py — PPE Verification System: Comprehensive Testing Module
===============================================================

OVERVIEW
--------
A full evaluation harness for the PPE sequential verification system.
Run this instead of main.py when benchmarking or collecting thesis data.
It executes the same detection pipeline as the live system but wraps
every verification step in timing, confidence recording, hardware
telemetry, and visual evidence capture. Results are written to
structured folders and CSV files for offline analysis.


USAGE
-----
  python test.py

The module auto-detects whether config.USE_CLASSIFIER is enabled and
organises all output under a matching subfolder:
  test_captures/With_Classifier/   — when classifier is on
  test_captures/YOLO_Only/         — when classifier is off

Set MAX_TEST_BATCHES at the top of the file to limit how many
successful verification runs are collected before the module exits
automatically. Set to 0 to run indefinitely until Q is pressed.


BATCH LIFECYCLE
---------------
One "batch" = one complete sequential verification of all required
PPE items from start to finish.

  1. A new MetricsRecorder is created for the batch.
  2. The SequentialPPEChecker state machine steps through each PPE
     stage in order (haircap → mask → long sleeves → apron →
     gloves → boots).
  3. When a stage is verified, timing and confidence data are recorded
     and four evidence images are saved.
  4. When all stages are complete, the batch is finalised, the
     overall summary files are updated, and a 5-second hold is shown
     before the next batch begins.
  5. On reaching MAX_TEST_BATCHES the module prints a summary and exits.


EVIDENCE CAPTURE  (save_capture_quad)
--------------------------------------
Four images are saved per verified PPE item into a numbered stage
subfolder (e.g. Test_001/02_mask/):

  01_yolo_N.jpg       — cropped detection with YOLO bounding box and
                        confidence label overlaid.
  02_keypoints_N.jpg  — same crop with MediaPipe pose skeleton drawn.
  03_classifier_N.jpg — crop with a classifier confidence bar chart
                        panel, PASS/FAIL banner, and per-class scores.
                        Shows "CLASSIFIER OFF" if disabled.
  04_combined.jpg     — full frame with pose skeleton, all YOLO boxes,
                        and a classifier result chip in the corner.
                        Stamped with batch number and stage name.

For paired items (gloves, boots) one set of images is saved per
detection (N = 1 and N = 2).


METRICS RECORDING  (MetricsRecorder)
--------------------------------------
Captures the following per PPE stage per batch:

  duration_seconds    — wall-clock time from stage start to verified.
  frames_to_verify    — number of processed frame loops taken.
  yolo_conf_1/2       — YOLO confidence for each active detection.
  cls_conf_1/2        — classifier confidence for each detection
                        (empty if classifier is disabled).
  cpu_percent         — CPU utilisation at the moment of verification.
  ram_percent         — RAM utilisation at the moment of verification.
  temp_c              — Raspberry Pi SoC temperature (°C) via vcgencmd
                        or psutil fallback.

A BATCH_TOTAL row is appended at the end of each per-batch CSV.

Output files:
  test_captures/<Mode>/Test_NNN/metrics.csv   — per-batch detail
  test_captures/Overall_Results/overall_summary_<mode>.txt  — human-readable aggregate
  test_captures/Overall_Results/overall_summary_<mode>.csv  — machine-readable aggregate


SUMMARY STATISTICS
------------------
After every batch the overall summary is regenerated from all
metrics.csv files collected so far, giving running totals without
needing to wait until the end. Per PPE item the summary reports:

  Avg / Min / Max / StdDev of verification time
  Average frames-to-verify
  Average YOLO confidence
  Average classifier confidence (if applicable)
  Sample count (N)

Hardware telemetry is averaged and peak values are noted.
Batch totals are listed individually with overall average, best,
and worst times.


INTERLEAVING NOTE
-----------------
Unlike the test loop in main.py, this module runs YOLO and MediaPipe
pose estimation on every processed frame (no interleaving). This
provides more accurate per-frame latency figures for thesis evaluation
at the cost of lower throughput.


HUD OVERLAY  (_draw_hud)
-------------------------
A semi-transparent panel drawn over the live camera feed shows:

  Current batch number and progress (stages verified / total)
  Stability counter vs required threshold and elapsed time
  YOLO detection status for the current stage
  MediaPipe keypoint verification status
  Classifier result with predicted class and confidence
  Stage strip along the bottom (green = verified, amber = current)
  Keyboard shortcut hint bar


KEYBOARD CONTROLS
-----------------
  Q      — quit and print final summary paths.
  R      — manual reset: saves partial metrics, starts a new batch.
  D      — toggle debug bounding-box / skeleton overlay.
  A      — toggle audio on / off.
  SPACE  — pause / resume processing.


HELPER FUNCTIONS
----------------
  _group(detections)
    Same grouping logic as main.py: organises the flat YOLO list into
    a dict by PPE class, keeping only the highest-confidence singleton
    for single items and collecting all detections for paired items.

  _verify_stage(checker, grouped, pose_result, img_w, img_h)
    Runs the geometric verifier for the current checker stage and
    returns (yolo_ok, kp_ok, ppe_type, active_detections).

  get_hardware_stats()
    Reads CPU and RAM via psutil, and SoC temperature via vcgencmd
    (Pi-native) with a psutil.sensors_temperatures fallback. Returns
    (0, 0, 0) if psutil is not installed.

  _draw_yolo_box / _draw_pose_keypoints / _draw_classifier_overlay / _crop
    Low-level drawing primitives used by save_capture_quad to build
    the four evidence images for each verified stage.


DEPENDENCIES
------------
  cv2 (opencv-python)          — camera capture, display, drawing
  numpy                        — image array manipulation
  psutil                       — CPU / RAM / temperature telemetry
  csv, statistics, datetime    — metrics recording and summary generation
  subprocess                   — vcgencmd temperature reads on Raspberry Pi
  config                       — runtime parameters
  ui / ui.py                   — UIConfig, UIRenderer (rotation helper)
  core/detector.py             — PPEDetector (YOLO wrapper)
  core/pose_estimator.py       — PoseEstimator (MediaPipe wrapper)
  core/classifier.py           — PPEClassifier (optional, loaded on demand)
  core/sequential_checker.py   — SequentialPPEChecker, PPECheckStage
  core/verifier.py             — per-item geometric verifiers
  core/audio_feedback.py       — AudioFeedback (optional)
"""

import os
import csv
import time
import datetime
import subprocess
import statistics

import cv2
import numpy as np

# Hardware telemetry
try:
    import psutil
    psutil.cpu_percent(interval=None) # Prime the CPU monitor
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠  psutil not installed. Hardware metrics will be 0. (pip install psutil)")

import config
from ui                      import UIConfig, UIRenderer
from core.detector           import PPEDetector
from core.pose_estimator     import PoseEstimator
from core.sequential_checker import SequentialPPEChecker, PPECheckStage
from core.verifier           import (verify_mask, verify_haircap, verify_gloves,
                                     verify_boots, verify_apron, verify_long_sleeves)

# =============================================================================
#  TESTING CONFIGURATION
# =============================================================================

# Limit how many successful batches to run before auto-terminating. 
# Set to 0 to run infinitely until 'q' is pressed.
MAX_TEST_BATCHES = 30 

# ─────────────────────────────────────────────────────────────────────────────
# Mode Detection & Classifier Loading
# ─────────────────────────────────────────────────────────────────────────────
MODE_NAME = "With_Classifier" if config.USE_CLASSIFIER else "YOLO_Only"

classifier        = None
CLASSIFIER_ACTIVE = False
if config.USE_CLASSIFIER:
    try:
        from core.classifier import PPEClassifier
        classifier        = PPEClassifier(config.CLASSIFIER_MODEL_PATH)
        CLASSIFIER_ACTIVE = True
        print("✓  Classifier loaded")
    except Exception as _ce:
        print(f"⚠  Classifier failed to load: {_ce}")
        CLASSIFIER_ACTIVE = False

audio = None
try:
    from core.audio_feedback import AudioFeedback
    audio = AudioFeedback(config)
except Exception:
    pass

# =============================================================================
#  CONSTANTS & PATHS
# =============================================================================

CAPTURES_ROOT   = os.path.join("test_captures", MODE_NAME)
OVERALL_DIR     = os.path.join("test_captures", "Overall_Results")
OVERALL_FILE    = os.path.join(OVERALL_DIR, f"overall_summary_{MODE_NAME.lower()}.txt")
OVERALL_CSV_FILE = os.path.join(OVERALL_DIR, f"overall_summary_{MODE_NAME.lower()}.csv")
METRICS_CSV     = "metrics.csv"
CROP_PADDING    = 30
MISSING_ANNOUNCE_INTERVAL = 10.0  # Seconds between "missing" voice lines

STAGE_LABEL_MAP = {
    PPECheckStage.HAIRCAP:      ("01_haircap",     "haircap"),
    PPECheckStage.MASK:         ("02_mask",         "mask"),
    PPECheckStage.LONG_SLEEVES: ("03_long_sleeves", "long_sleeves"),
    PPECheckStage.APRON:        ("04_apron",        "apron"),
    PPECheckStage.GLOVES:       ("05_gloves",       "gloves"),
    PPECheckStage.BOOTS:        ("06_boots",        "boots"),
}

_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTM = cv2.FONT_HERSHEY_PLAIN

# =============================================================================
#  HARDWARE TELEMETRY
# =============================================================================

def get_hardware_stats():
    """Returns CPU (%), RAM (%), and Temperature (°C)."""
    if not PSUTIL_AVAILABLE:
        return 0.0, 0.0, 0.0
        
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    temp = 0.0
    
    try:
        res = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
        temp = float(res.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        try:
            temps = psutil.sensors_temperatures()
            if 'cpu_thermal' in temps:
                temp = temps['cpu_thermal'][0].current
            elif 'coretemp' in temps:
                temp = temps['coretemp'][0].current
        except Exception:
            pass
            
    return cpu, ram, temp

# =============================================================================
#  DIRECTORY HELPERS
# =============================================================================

def _next_batch_number() -> int:
    os.makedirs(CAPTURES_ROOT, exist_ok=True)
    nums = []
    for d in os.listdir(CAPTURES_ROOT):
        if d.startswith("Test_") and os.path.isdir(os.path.join(CAPTURES_ROOT, d)):
            try:
                nums.append(int(d.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(nums, default=0) + 1

def _batch_dir(n: int) -> str:
    return os.path.join(CAPTURES_ROOT, f"Test_{n:03d}")

def _stage_dir(n: int, stage: PPECheckStage) -> str:
    folder = STAGE_LABEL_MAP.get(stage, (f"unknown_{stage.name}", ""))[0]
    return os.path.join(_batch_dir(n), folder)

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)

# =============================================================================
#  DRAWING PRIMITIVES
# =============================================================================

def _draw_yolo_box(img, bbox, label, conf):
    out = img.copy()
    x1, y1, x2, y2 = bbox
    col = (0, 200, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
    tag = f"YOLO: {label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(tag, _FONT, 0.65, 2)
    cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), col, -1)
    cv2.putText(out, tag, (x1 + 3, y1 - 5), _FONT, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out

def _draw_pose_keypoints(img, pose_result, col_node=(0, 255, 100), col_edge=(100, 255, 0)):
    out = img.copy()
    if pose_result is None or not pose_result.pose_landmarks:
        return out
    h, w = out.shape[:2]
    for landmarks in pose_result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in _POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], col_edge, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 5, col_node, -1, cv2.LINE_AA)
            cv2.circle(out, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _crop(img, bbox, pad=CROP_PADDING):
    x1, y1, x2, y2 = bbox
    ih, iw = img.shape[:2]
    return img[max(0, y1 - pad) : min(ih, y2 + pad), max(0, x1 - pad) : min(iw, x2 + pad)].copy()

def _draw_classifier_overlay(base_img, cls_result, expected_class, bbox):
    crop = _crop(base_img, bbox)
    if crop.size == 0:
        return base_img.copy()

    h, w = crop.shape[:2]

    if cls_result is None or not CLASSIFIER_ACTIVE:
        out = crop.copy()
        ov  = out.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
        cv2.putText(out, "CLASSIFIER",    (w // 2 - 75, h // 2 - 14), _FONT, 0.65, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(out, "OFF / N/A", (w // 2 - 60, h // 2 + 18), _FONT, 0.65, (150, 150, 150), 2, cv2.LINE_AA)
        return out

    predicted  = cls_result.get("predicted_class", "?")
    conf       = cls_result.get("confidence", 0.0)
    is_correct = cls_result.get("is_correct", False)
    all_confs  = cls_result.get("all_confidences", {})

    sorted_classes = sorted(all_confs.items(), key=lambda kv: (kv[0] != predicted, -kv[1]))

    BAR_H, BAR_GAP, LABEL_W, HEADER_H = 18, 5, 115, 28
    panel_h   = HEADER_H + len(sorted_classes) * (BAR_H + BAR_GAP) + 12
    canvas_h  = h + panel_h
    canvas    = np.zeros((canvas_h, w, 3), dtype=np.uint8)
    canvas[:h] = crop

    cv2.rectangle(canvas, (0, h), (w, canvas_h), (18, 18, 18), -1)
    header = f"Classifier  ->  {predicted}  ({conf * 100:.1f}%)"
    cv2.putText(canvas, header, (6, h + HEADER_H - 6), _FONT, 0.50, (210, 210, 210), 1, cv2.LINE_AA)

    bar_max_w = w - LABEL_W - 14
    for i, (cls_name, cls_conf) in enumerate(sorted_classes):
        by = h + HEADER_H + i * (BAR_H + BAR_GAP)
        is_pred, is_exp = (cls_name == predicted), (cls_name == expected_class)

        lbl_col = ((0, 255, 120) if is_pred and is_exp else
                   (0, 140, 255) if is_pred else
                   (80, 80, 255) if is_exp else (130, 130, 130))

        cv2.putText(canvas, cls_name, (4, by + BAR_H - 4), _FONT, 0.40, lbl_col, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (LABEL_W, by), (LABEL_W + bar_max_w, by + BAR_H), (45, 45, 45), -1)

        filled = max(1, int(bar_max_w * cls_conf)) if cls_conf > 0 else 0
        if filled > 0:
            bar_col = (0, 200, 60) if is_pred else (50, 100, 180)
            cv2.rectangle(canvas, (LABEL_W, by), (LABEL_W + filled, by + BAR_H), bar_col, -1)

        cv2.putText(canvas, f"{cls_conf * 100:.1f}%", (LABEL_W + filled + 4, by + BAR_H - 4), _FONT, 0.36, (190, 190, 190), 1, cv2.LINE_AA)

    banner_col = (0, 165, 0) if is_correct else (0, 30, 200)
    banner_txt = f"PASS  {predicted}" if is_correct else f"FAIL  {predicted}  (exp: {expected_class})"
    cv2.rectangle(canvas, (0, 0), (w, 30), banner_col, -1)
    cv2.putText(canvas, banner_txt, (6, 22), _FONT, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

# =============================================================================
#  IMAGE CAPTURE SAVE
# =============================================================================

def save_capture_quad(batch_num, stage, frame_early, pose_early,
                      frame_cls, pose_cls, detections, cls_results):
    out_dir  = _stage_dir(batch_num, stage)
    _ensure(out_dir)

    img4 = _draw_pose_keypoints(frame_cls, pose_cls)
    all_correct = True

    for i, det in enumerate(detections):
        bbox, label, conf = det["bbox"], det["class"], det["confidence"]
        expected = STAGE_LABEL_MAP.get(stage, ("", label))[1]

        img1 = _crop(_draw_yolo_box(frame_early, bbox, label, conf), bbox)
        if img1.size > 0: cv2.imwrite(os.path.join(out_dir, f"01_yolo_{i+1}.jpg"), img1)

        img2 = _crop(_draw_pose_keypoints(frame_early, pose_early), bbox)
        if img2.size > 0: cv2.imwrite(os.path.join(out_dir, f"02_keypoints_{i+1}.jpg"), img2)

        if CLASSIFIER_ACTIVE and cls_results is not None and i < len(cls_results):
            res = cls_results[i]
            if not res.get("is_correct", False): all_correct = False
            img3 = _draw_classifier_overlay(frame_cls, res, expected, bbox)
        else:
            img3 = _draw_classifier_overlay(frame_cls, None, expected, bbox)
            
        if img3.size > 0: cv2.imwrite(os.path.join(out_dir, f"03_classifier_{i+1}.jpg"), img3)

        img4 = _draw_yolo_box(img4, bbox, label, conf)

    if CLASSIFIER_ACTIVE and cls_results is not None and len(cls_results) > 0:
        pred     = cls_results[0].get("predicted_class", "?")
        avg_conf = sum(r.get("confidence", 0.0) for r in cls_results) / len(cls_results)
        chip_col = (0, 180, 0) if all_correct else (0, 40, 210)
        chip_txt = (f"CLF  PASS  {pred}  {avg_conf * 100:.1f}%" if all_correct
                    else f"CLF  FAIL  {pred}  {avg_conf * 100:.1f}%")
        iw = img4.shape[1]
        (tw, th), _ = cv2.getTextSize(chip_txt, _FONT, 0.60, 2)
        cv2.rectangle(img4, (iw - tw - 16, 6), (iw - 4, th + 16), chip_col, -1)
        cv2.putText(img4, chip_txt, (iw - tw - 12, th + 10), _FONT, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img4, "CLASSIFIER: OFF", (img4.shape[1] - 180, 28), _FONT, 0.65, (130, 130, 130), 2, cv2.LINE_AA)

    stamp = f"Test_{batch_num:03d}  |  {stage.name}  |  VERIFIED"
    cv2.putText(img4, stamp, (10, img4.shape[0] - 10), _FONT, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "04_combined.jpg"), img4)
    print(f"  📷  Saved → {out_dir}/")


# =============================================================================
#  METRICS RECORDER (TIMING + FRAMES + CONFIDENCE + HARDWARE)
# =============================================================================

class MetricsRecorder:
    def __init__(self, batch_num: int):
        self.batch_num   = batch_num
        self.batch_start = time.time()
        self._starts: dict = {}
        self._frame_counts: dict = {}
        self.stage_data: dict = {}
        self.batch_total = None

    def mark_stage_start(self, stage: PPECheckStage):
        if stage not in self._starts:
            self._starts[stage] = time.time()
            self._frame_counts[stage] = 0

    def increment_frame(self, stage: PPECheckStage):
        """Record another frame loop spent attempting to verify this stage."""
        if stage in self._frame_counts:
            self._frame_counts[stage] += 1

    def record_stage_metrics(self, stage: PPECheckStage, detections: list, cls_results: list):
        start = self._starts.get(stage)
        if start is None: return
        
        # Calculate EXACT time latency taken to verify this specific item
        elapsed = round(time.time() - start, 3)
        frames = self._frame_counts.get(stage, 0)
        cpu, ram, temp = get_hardware_stats()
        
        yolo_1 = round(detections[0]["confidence"], 4) if len(detections) > 0 else ""
        yolo_2 = round(detections[1]["confidence"], 4) if len(detections) > 1 else ""

        cls_1, cls_2 = "", ""
        if CLASSIFIER_ACTIVE and cls_results is not None:
            cls_1 = round(cls_results[0].get("confidence", 0.0), 4) if len(cls_results) > 0 else ""
            cls_2 = round(cls_results[1].get("confidence", 0.0), 4) if len(cls_results) > 1 else ""

        self.stage_data[stage.name] = {
            "duration": elapsed,
            "frames_to_verify": frames,
            "yolo_conf_1": yolo_1, "yolo_conf_2": yolo_2,
            "cls_conf_1": cls_1, "cls_conf_2": cls_2,
            "cpu_util": cpu, "ram_util": ram, "temp_c": temp
        }
        
        print(f"  ⏱  {stage.name}: {elapsed:.2f}s  ({frames} loops)  (CPU: {cpu}% | Temp: {temp}°C)")

    def mark_batch_done(self):
        self.batch_total = round(time.time() - self.batch_start, 3)
        print(f"\n  🏁  Batch Test_{self.batch_num:03d} complete in {self.batch_total:.2f}s\n")
        self._write_csv()
        self._update_summary()

    def _write_csv(self):
        path = os.path.join(_batch_dir(self.batch_num), METRICS_CSV)
        _ensure(_batch_dir(self.batch_num))
        
        rows = []
        for stage_name, data in self.stage_data.items():
            rows.append({
                "ppe_item": stage_name,
                "duration_seconds": data["duration"],
                "frames_to_verify": data["frames_to_verify"],
                "yolo_conf_1": data["yolo_conf_1"], "yolo_conf_2": data["yolo_conf_2"],
                "cls_conf_1": data["cls_conf_1"], "cls_conf_2": data["cls_conf_2"],
                "cpu_percent": data["cpu_util"], "ram_percent": data["ram_util"], "temp_c": data["temp_c"],
                "batch_num": self.batch_num
            })
            
        rows.append({
            "ppe_item": "BATCH_TOTAL",
            "duration_seconds": self.batch_total,
            "frames_to_verify": "",
            "yolo_conf_1": "", "yolo_conf_2": "", "cls_conf_1": "", "cls_conf_2": "",
            "cpu_percent": "", "ram_percent": "", "temp_c": "",
            "batch_num": self.batch_num
        })
        
        fieldnames = ["ppe_item", "duration_seconds", "frames_to_verify", "yolo_conf_1", "yolo_conf_2", "cls_conf_1", "cls_conf_2", "cpu_percent", "ram_percent", "temp_c", "batch_num"]
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)
        print(f"  💾  Metrics saved → {path}")

    def _update_summary(self):
        _ensure(OVERALL_DIR)
        all_rows = []
        
        for entry in sorted(os.listdir(CAPTURES_ROOT)):
            if not entry.startswith("Test_"): continue
            csv_path = os.path.join(CAPTURES_ROOT, entry, METRICS_CSV)
            if os.path.isfile(csv_path):
                with open(csv_path, newline="") as f:
                    all_rows.extend(csv.DictReader(f))
                    
        if not all_rows: return

        per_item     = {}
        batch_totals = {}
        hw_metrics   = {"cpu": [], "ram": [], "temp": []}
        
        for row in all_rows:
            item = row["ppe_item"]
            dur  = float(row["duration_seconds"])
            blbl = f"Test_{int(row['batch_num']):03d}"
            
            if item == "BATCH_TOTAL":
                batch_totals[blbl] = dur
            else:
                if item not in per_item:
                    per_item[item] = {"dur": [], "frames": [], "yolo": [], "cls": []}
                    
                per_item[item]["dur"].append(dur)
                if row.get("frames_to_verify") and row["frames_to_verify"] != "": 
                    per_item[item]["frames"].append(int(row["frames_to_verify"]))
                if row.get("yolo_conf_1") and row["yolo_conf_1"] != "": 
                    per_item[item]["yolo"].append(float(row["yolo_conf_1"]))
                if row.get("yolo_conf_2") and row["yolo_conf_2"] != "": 
                    per_item[item]["yolo"].append(float(row["yolo_conf_2"]))
                if row.get("cls_conf_1") and row["cls_conf_1"] != "":  
                    per_item[item]["cls"].append(float(row["cls_conf_1"]))
                if row.get("cls_conf_2") and row["cls_conf_2"] != "":  
                    per_item[item]["cls"].append(float(row["cls_conf_2"]))
                
                if row.get("cpu_percent") and row["cpu_percent"] != "": hw_metrics["cpu"].append(float(row["cpu_percent"]))
                if row.get("ram_percent") and row["ram_percent"] != "": hw_metrics["ram"].append(float(row["ram_percent"]))
                if row.get("temp_c") and row["temp_c"] != "":      hw_metrics["temp"].append(float(row["temp_c"]))

        # ---------------------------------------------------------
        # GENERATE TXT SUMMARY
        # ---------------------------------------------------------
        lines = [
            "=" * 98,
            f"       PPE VERIFICATION — {MODE_NAME.replace('_', ' ').upper()} RESULTS",
            f"       Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "=" * 98, "",
            "─── AI PERFORMANCE, LOOP LATENCY & ACCURACY PER PPE ITEM ───",
            f"  {'PPE Item':<15} {'Avg Time':>9} {'Min Time':>9} {'Max Time':>9} {'StdDev':>7} {'Avg Frames':>10} {'YOLO Acc':>9} {'Cls Acc':>9} {'N':>4}",
            "  " + "─" * 90,
        ]
        
        csv_overall_rows = []
        
        for item in sorted(per_item):
            durs, frames, yolos, clss = per_item[item]["dur"], per_item[item]["frames"], per_item[item]["yolo"], per_item[item]["cls"]
            
            n_count  = len(durs)
            avg_dur  = sum(durs) / n_count if n_count else 0
            min_dur  = min(durs) if n_count else 0
            max_dur  = max(durs) if n_count else 0
            std_dur  = statistics.stdev(durs) if n_count > 1 else 0
            
            avg_frm  = sum(frames) / len(frames) if frames else 0
            avg_yolo = (sum(yolos) / len(yolos) * 100) if yolos else 0
            avg_cls  = (sum(clss) / len(clss) * 100) if clss else 0
            cls_str  = f"{avg_cls:>8.1f}%" if clss else f"{'N/A':>9}"
            
            lines.append(f"  {item:<15} {avg_dur:>8.2f}s {min_dur:>8.2f}s {max_dur:>8.2f}s {std_dur:>6.2f}s {avg_frm:>10.1f} {avg_yolo:>8.1f}% {cls_str} {n_count:>4}")
            
            # Prepare data for overall CSV
            csv_overall_rows.append({
                "ppe_item": item,
                "avg_duration_sec": round(avg_dur, 3),
                "min_duration_sec": round(min_dur, 3),
                "max_duration_sec": round(max_dur, 3),
                "stddev_duration_sec": round(std_dur, 3),
                "avg_frames_to_verify": round(avg_frm, 1),
                "avg_yolo_conf": round(avg_yolo, 2),
                "avg_cls_conf": round(avg_cls, 2) if clss else "",
                "samples_n": n_count
            })
            
        # Hardware Metrics Section
        lines += ["", "─── RASPBERRY PI 5 HARDWARE TELEMETRY ───"]
        if hw_metrics["cpu"]:
            avg_cpu, max_cpu = sum(hw_metrics["cpu"])/len(hw_metrics["cpu"]), max(hw_metrics["cpu"])
            avg_ram, max_ram = sum(hw_metrics["ram"])/len(hw_metrics["ram"]), max(hw_metrics["ram"])
            avg_tmp, max_tmp = sum(hw_metrics["temp"])/len(hw_metrics["temp"]), max(hw_metrics["temp"])
            lines += [
                f"  CPU Utilization : Avg {avg_cpu:.1f}%   (Peak {max_cpu:.1f}%)",
                f"  RAM Utilization : Avg {avg_ram:.1f}%   (Peak {max_ram:.1f}%)",
                f"  SoC Temperature : Avg {avg_tmp:.1f}°C   (Peak {max_tmp:.1f}°C)"
            ]
        else:
            lines.append("  (Hardware data unavailable)")

        lines += ["", "─── TOTAL TIME PER BATCH ───",
                  f"  {'Batch':<14}  {'Duration':>10}", "  " + "─" * 28]
        for b in sorted(batch_totals):
            lines.append(f"  {b:<14}  {batch_totals[b]:>9.2f}s")
        if batch_totals:
            v = list(batch_totals.values())
            lines += [f"  {'AVERAGE':<14}  {sum(v)/len(v):>9.2f}s",
                      f"  {'BEST':<14}  {min(v):>9.2f}s",
                      f"  {'WORST':<14}  {max(v):>9.2f}s"]
        lines += ["", "=" * 98, f"  Total completed batches ({MODE_NAME}): {len(batch_totals)}", "=" * 98]
                  
        with open(OVERALL_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")
            
        # ---------------------------------------------------------
        # GENERATE CSV SUMMARY
        # ---------------------------------------------------------
        csv_fieldnames = ["ppe_item", "avg_duration_sec", "min_duration_sec", "max_duration_sec", 
                          "stddev_duration_sec", "avg_frames_to_verify", "avg_yolo_conf", "avg_cls_conf", "samples_n"]
        
        with open(OVERALL_CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_overall_rows)
            
        print(f"  📊  Summary updated → {OVERALL_FILE}")
        print(f"  📊  CSV Summary generated → {OVERALL_CSV_FILE}")


# =============================================================================
#  DETECTION GROUPING 
# =============================================================================

def _group(detections: list) -> dict:
    g = {"apron": None, "boots": [], "gloves": [],
         "haircap": None, "long_sleeves": None, "mask": None}
    for d in detections:
        c = d["class"]
        if c in ("apron", "haircap", "long_sleeves", "mask"):
            if g[c] is None or d["confidence"] > g[c]["confidence"]:
                g[c] = d
        elif c in ("boots", "gloves"):
            g[c].append(d)
    return g

# =============================================================================
#  STAGE VERIFICATION
# =============================================================================

def _verify_stage(checker, grouped, pose_result, img_w, img_h):
    stage = checker.current_stage
    if stage == PPECheckStage.COMPLETE:
        return True, True, None, []

    ppe_type  = checker.get_ppe_type_for_stage(stage)
    yolo_ok   = kp_ok = False
    active_detections = []

    if stage in (PPECheckStage.HAIRCAP, PPECheckStage.MASK,
                 PPECheckStage.LONG_SLEEVES, PPECheckStage.APRON):
        det = grouped.get(ppe_type)
        yolo_ok = det is not None
        if yolo_ok:
            FN = {
                PPECheckStage.HAIRCAP:      (verify_haircap,     config.MARGIN_HAIRCAP),
                PPECheckStage.MASK:         (verify_mask,         config.MARGIN_MASK),
                PPECheckStage.LONG_SLEEVES: (verify_long_sleeves, config.MARGIN_LONG_SLEEVES),
                PPECheckStage.APRON:        (verify_apron,        config.MARGIN_APRON),
            }
            fn, margin = FN[stage]
            kp_ok = (fn(det["bbox"], pose_result, img_w, img_h, margin=margin) == "DETECTED_CORRECT")
            active_detections = [det]

    elif stage == PPECheckStage.GLOVES:
        gloves_list = grouped.get("gloves", [])
        yolo_ok = len(gloves_list) >= 2
        if yolo_ok:
            res = verify_gloves([g["bbox"] for g in gloves_list], pose_result, img_w, img_h, margin=config.MARGIN_GLOVES)
            kp_ok = (res.get("left") == "DETECTED_CORRECT" and res.get("right") == "DETECTED_CORRECT")
            active_detections = gloves_list[:2]

    elif stage == PPECheckStage.BOOTS:
        boots_list = grouped.get("boots", [])
        yolo_ok = len(boots_list) >= 2
        if yolo_ok:
            res = verify_boots([b["bbox"] for b in boots_list], pose_result, img_w, img_h, margin=config.MARGIN_BOOTS)
            kp_ok = (res.get("left") == "DETECTED_CORRECT" and res.get("right") == "DETECTED_CORRECT")
            active_detections = boots_list[:2]

    return yolo_ok, kp_ok, ppe_type, active_detections

# =============================================================================
#  HUD
# =============================================================================

def _draw_hud(frame, batch_num, checker, fps, yolo_ok, kp_ok, cls_result, paused, ui_cfg, batches_completed):
    out = frame.copy()
    PANEL_W, PANEL_H = out.shape[1], 230               
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.58, out, 0.42, 0, out)

    status = checker.get_status_summary()
    PAD, y, dy = 10, 22, 22

    def put(text, col=(215, 215, 215), scale=0.55, thick=1):
        nonlocal y
        cv2.putText(out, text, (PAD, y), _FONT, scale, col, thick, cv2.LINE_AA)
        y += dy

    put("[ TEST MODE ]", (0, 220, 255), 0.68, 2)

    fps_str = f"FPS: {fps:.1f}"
    (fsw, _), _ = cv2.getTextSize(fps_str, _FONT, 0.50, 1)
    cv2.putText(out, fps_str, (PANEL_W - fsw - PAD, y - dy + 14), _FONT, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
    
    batch_str = f"Batch: Test_{batch_num:03d}"
    if MAX_TEST_BATCHES > 0:
        batch_str += f" ({batches_completed}/{MAX_TEST_BATCHES})"
        
    put(f"{batch_str}   Stage {status['verified']}/{status['total']}: {checker.get_current_stage_name()}", (255, 200, 70), 0.50)
    put(f"Stability: {checker.stability_counter} / {checker.required_stability}   Elapsed: {checker.get_elapsed_time():.1f}s", (155, 200, 255), 0.48)

    yc = (0, 225, 100) if yolo_ok else (55, 55, 225)
    kc = (0, 225, 100) if kp_ok   else (55, 55, 225)
    put(f"[1] YOLO      : {'DETECTED' if yolo_ok else 'not found'}", yc, 0.50)
    put(f"[2] MediaPipe : {'correct position' if kp_ok else 'wrong / missing'}", kc, 0.50)

    if not CLASSIFIER_ACTIVE:
        put("[3] Classifier: OFF (config.USE_CLASSIFIER = False)", (110, 110, 110), 0.50)
    elif cls_result is None or len(cls_result) == 0:
        put("[3] Classifier: waiting for stability …", (150, 150, 150), 0.48)
    else:
        if isinstance(cls_result, list):
            all_pass = all(r.get("is_correct", False) for r in cls_result)
            cc = (0, 225, 100) if all_pass else (55, 55, 225)
            avg_conf = sum(r.get("confidence", 0.0) for r in cls_result) / len(cls_result)
            put(f"[3] Classifier: {'PASS (All)' if all_pass else 'FAIL'}  {cls_result[0].get('predicted_class', '?')}  {avg_conf * 100:.1f}%", cc, 0.50)
        else:
            pred, conf_p, correct = cls_result.get("predicted_class", "?"), cls_result.get("confidence", 0.0), cls_result.get("is_correct", False)
            cc = (0, 225, 100) if correct else (55, 55, 225)
            put(f"[3] Classifier: {'PASS' if correct else 'FAIL'}  {pred}  {conf_p * 100:.1f}%", cc, 0.50)

    if paused: put("*** PAUSED  ***", (0, 80, 255), 0.65, 2)

    stages = checker.check_order[:-1]
    n = len(stages)
    tile_w = PANEL_W // n if n else PANEL_W
    strip_y = out.shape[0] - 38

    for i, stage in enumerate(stages):
        name = checker._get_stage_name(stage)[:6]
        done, cur = checker.verified_items.get(stage, False), (stage == checker.current_stage)
        col = (0, 215, 0) if done else ((255, 200, 50) if cur else (50, 50, 50))
        tx = i * tile_w + tile_w // 2

        if done or cur:
            bg = (15, 34, 24) if done else (25, 32, 44)
            cv2.rectangle(out, (i * tile_w, strip_y - 4), (i * tile_w + tile_w - 2, out.shape[0] - 20), bg, -1)

        (lw, lh), _ = cv2.getTextSize(name, _FONTM, 0.9, 1)
        cv2.putText(out, name, (tx - lw // 2, strip_y + lh), _FONTM, 0.9, col, 1, cv2.LINE_AA)

    hint = "Q=quit  R=reset  D=debug  A=audio  SPC=pause"
    (hw, _), _ = cv2.getTextSize(hint, _FONTM, 0.75, 1)
    cv2.putText(out, hint, (PANEL_W // 2 - hw // 2, out.shape[0] - 6), _FONTM, 0.75, (80, 80, 80), 1, cv2.LINE_AA)

    return out

# =============================================================================
#  MAIN TEST LOOP
# =============================================================================

def main():
    ui_cfg = UIConfig()
    OUT_W, OUT_H, ROT = ui_cfg.OUTPUT_W, ui_cfg.OUTPUT_H, ui_cfg.CAMERA_ROTATION   

    print("=" * 70)
    print("       PPE VERIFICATION — COMPREHENSIVE TEST MODE")
    print("=" * 70)
    print(f"  Resolution      : {OUT_W} × {OUT_H}  ({ui_cfg.ORIENTATION})")
    print(f"  Camera rotation : {ROT}°")
    print(f"  Mode            : {MODE_NAME.upper()}")
    print(f"  Hardware Mon.   : {'ACTIVE (psutil + vcgencmd)' if PSUTIL_AVAILABLE else 'DISABLED'}")
    print(f"  Batch Limit     : {MAX_TEST_BATCHES if MAX_TEST_BATCHES > 0 else 'Infinite'}")
    print(f"  Overall Results : {os.path.abspath(OVERALL_FILE)}")
    print(f"  CSV Summary     : {os.path.abspath(OVERALL_CSV_FILE)}")
    print("=" * 70 + "\n")

    detector = PPEDetector(config.YOLO_MODEL_PATH)
    pose_est = PoseEstimator(config.MEDIAPIPE_DIR)

    cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened(): raise RuntimeError("Camera failed to open")

    WIN = "PPE Test Mode"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if ui_cfg.FULLSCREEN: cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else: cv2.resizeWindow(WIN, OUT_W, OUT_H)

    batch_num = _next_batch_number()
    checker   = SequentialPPEChecker(config)
    recorder  = MetricsRecorder(batch_num)

    show_debug = config.ENABLE_DEBUG_VIZ
    paused     = False
    frame_count = fps = 0
    fps_time    = time.time()
    batches_completed = 0

    early_frame, early_pose, early_detection = None, None, None
    cls_frame, cls_pose, cls_detection, cls_result = None, None, None, None

    prev_stage   = checker.current_stage
    yolo_ok      = kp_ok = False
    active_detections = []
    pose_result = None
    detections   = []

    last_det_time, completion_time = time.time(), None
    last_missing_announce_time = 0

    print(f"\n  ▶  Starting Batch  →  Test_{batch_num:03d}")
    recorder.mark_stage_start(checker.current_stage)

    while cap.isOpened():
        ret, raw = cap.read()
        if not ret: break

        frame = UIRenderer.rotate_frame(cv2.flip(raw, 1), ROT)
        frame_count += 1
        now = time.time()

        if now - fps_time >= 1.0:
            fps, frame_count, fps_time = frame_count / (now - fps_time), 0, now

        if frame_count % config.FRAME_SKIP != 0:
            cv2.imshow(WIN, _draw_hud(frame, batch_num, checker, fps, yolo_ok, kp_ok, cls_result, paused, ui_cfg, batches_completed))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        if not paused:
            img_h, img_w = frame.shape[:2]      

            detections = detector.detect(frame, conf_threshold=config.YOLO_CONFIDENCE)
            grouped    = _group(detections)

            if any([grouped["mask"], grouped["haircap"], grouped["long_sleeves"], grouped["apron"], grouped["gloves"], grouped["boots"]]):
                last_det_time = now

            if now - last_det_time > 5.0:
                if not checker.is_complete() or completion_time is None:
                    if audio: audio.interrupt()
                    checker.reset()
                    recorder = MetricsRecorder(batch_num)
                    completion_time, yolo_ok, kp_ok, active_detections, pose_result = None, False, False, [], None
                    early_frame, early_pose, early_detection, cls_frame, cls_pose, cls_detection, cls_result = None, None, None, None, None, None, None
                    last_missing_announce_time = 0
                    recorder.mark_stage_start(checker.current_stage)

            elif not checker.is_complete():
                # Tally up 1 frame loop for tracking algorithmic latency
                recorder.increment_frame(checker.current_stage)
                
                completion_time = None
                pose_result, _, _ = pose_est.process(frame)

                if checker.current_stage != prev_stage:
                    recorder.mark_stage_start(checker.current_stage)
                    prev_stage = checker.current_stage
                    cls_result, early_frame, early_pose, early_detection, cls_frame, cls_pose, cls_detection = None, None, None, None, None, None, None

                yolo_ok, kp_ok, ppe_type, active_detections = _verify_stage(checker, grouped, pose_result, img_w, img_h)

                if yolo_ok and kp_ok and active_detections:
                    if early_frame is None:
                        early_frame, early_pose, early_detection = frame.copy(), pose_result, active_detections

                classifier_ok = None
                if CLASSIFIER_ACTIVE and classifier and yolo_ok and kp_ok and active_detections and checker.stability_counter >= checker.required_stability and not checker.classifier_checked:
                    try:
                        cls_results_list = [classifier.verify_detection(frame, det["bbox"], ppe_type, confidence_threshold=config.CLASSIFIER_CONFIDENCE) for det in active_detections]
                        classifier_ok = all(r["is_correct"] for r in cls_results_list)
                        cls_result, cls_frame, cls_pose, cls_detection = cls_results_list, frame.copy(), pose_result, active_detections
                    except Exception:
                        classifier_ok, cls_result = True, None

                if not CLASSIFIER_ACTIVE: classifier_ok = True

                stage_before = checker.current_stage
                advanced = checker.process_verification(yolo_ok, kp_ok, classifier_ok)
                
                # Check for audio announcement (person is visible but PPE not verified)
                human_in_frame = (pose_result is not None and 
                                  hasattr(pose_result, 'pose_landmarks') and 
                                  pose_result.pose_landmarks and 
                                  len(pose_result.pose_landmarks) > 0)
                                  
                if advanced:
                    if audio: audio.interrupt()
                    last_missing_announce_time = 0
                elif checker.current_stage != prev_stage:
                    last_missing_announce_time = 0
                    
                if (human_in_frame and not yolo_ok and not checker.is_complete() and 
                    now - last_missing_announce_time >= MISSING_ANNOUNCE_INTERVAL):
                    last_missing_announce_time = now
                    if ppe_type and audio:
                        audio.announce(f"missing_{ppe_type}")

                if advanced and stage_before != PPECheckStage.COMPLETE:
                    d12 = early_detection if early_detection else active_detections
                    recorder.record_stage_metrics(stage_before, d12, cls_result)

                    f12, p12 = early_frame if early_frame is not None else frame, early_pose if early_pose is not None else pose_result
                    f34, p34 = cls_frame if cls_frame is not None else f12, cls_pose if cls_pose is not None else p12

                    if d12: save_capture_quad(batch_num, stage_before, f12, p12, f34, p34, d12, cls_result)

                    early_frame, early_pose, early_detection, cls_frame, cls_pose, cls_detection, cls_result = None, None, None, None, None, None, None
                    if audio: audio.interrupt()
                    if checker.current_stage != PPECheckStage.COMPLETE: recorder.mark_stage_start(checker.current_stage)
                    prev_stage = checker.current_stage

            else:
                if completion_time is None:
                    completion_time = now
                    recorder.mark_batch_done()
                    if audio: audio.announce("compliant", force=True)

                if now - completion_time > 5.0:
                    batches_completed += 1
                    
                    if MAX_TEST_BATCHES > 0 and batches_completed >= MAX_TEST_BATCHES:
                        print(f"\n  🎯 Reached batch limit of {MAX_TEST_BATCHES}. Terminating testing module.")
                        break
                        
                    if audio: audio.interrupt()
                    batch_num += 1
                    checker, recorder = SequentialPPEChecker(config), MetricsRecorder(batch_num)
                    completion_time, last_det_time, yolo_ok, kp_ok, active_detections, pose_result = None, now, False, False, [], None
                    early_frame, early_pose, early_detection, cls_frame, cls_pose, cls_detection, cls_result = None, None, None, None, None, None, None
                    prev_stage = checker.current_stage
                    last_missing_announce_time = 0
                    recorder.mark_stage_start(checker.current_stage)

        display = frame.copy()
        if show_debug:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(display, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 6), _FONT, 0.50, (0, 165, 255), 1)
            if pose_result and pose_result.pose_landmarks:
                from core.pose_estimator import draw_pose_landmarks
                for lm_set in pose_result.pose_landmarks: draw_pose_landmarks(display, lm_set)

        cv2.imshow(WIN, _draw_hud(display, batch_num, checker, fps, yolo_ok, kp_ok, cls_result, paused, ui_cfg, batches_completed))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            if audio: audio.interrupt()
            if recorder.stage_data:
                recorder.batch_total = round(now - recorder.batch_start, 3)
                recorder._write_csv()
                recorder._update_summary()
            batch_num += 1
            checker, recorder = SequentialPPEChecker(config), MetricsRecorder(batch_num)
            completion_time, last_det_time, yolo_ok, kp_ok, active_detections, pose_result = None, now, False, False, [], None
            early_frame, early_pose, early_detection, cls_frame, cls_pose, cls_detection, cls_result = None, None, None, None, None, None, None
            prev_stage = checker.current_stage
            last_missing_announce_time = 0
            recorder.mark_stage_start(checker.current_stage)
        elif key == ord('d'): show_debug = not show_debug
        elif key == ord('a'):
            if audio: audio.enabled = not audio.enabled
        elif key == ord(' '): paused = not paused

    cap.release()
    pose_est.close()
    if audio: audio.stop()
    cv2.destroyAllWindows()
    print(f"\n  Text Summary : {os.path.abspath(OVERALL_FILE)}")
    print(f"  CSV Summary  : {os.path.abspath(OVERALL_CSV_FILE)}")

if __name__ == "__main__":
    main()
