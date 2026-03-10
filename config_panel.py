"""
config_panel.py — Touch-Friendly Configuration Panel
=====================================================

OVERVIEW
--------
A full-screen, touch-optimised settings panel built on OpenCV. It is
designed to run on a Raspberry Pi 5 touchscreen as part of a PPE
(Personal Protective Equipment) detection system. The panel lets operators
adjust runtime parameters and writes all changes back to config.py on disk
so settings survive restarts.


ACCESS MODES
------------
The panel operates in two distinct access tiers:

  USER mode  — the default. Exposes only the controls an everyday operator
               needs: YOLO confidence threshold, PPE item requirements,
               audio settings (enable / volume / cooldown), and the
               auto-reset cooldown timer.

  ADMIN mode — unlocks the full control set, adding:
                 • Classifier confidence threshold
                 • Performance tuning (frame skip, stability frames,
                   TFLite thread count)
                 • Feature flags (classifier on/off, debug visualisation)
                 • Audio event toggles (announce on status change /
                   on completion)
                 • Detection margin fine-tuning for every PPE item

Switching modes
  User → Admin : press and HOLD the mode button in the header for 2 seconds.
                 A progress bar fills during the hold; releasing early cancels.
  Admin → User : single tap on the same mode button.
  The mode change (ADMIN_MODE flag) is written to config.py immediately —
  it does NOT wait for the SAVE button.


WORKFLOW
--------
1. ConfigPanel("config.py") creates the panel, bound to a config file path.
2. panel.open(win_name, W, H, config_module) blocks until the user acts:
     SAVE   — writes all changed values to config.py AND updates the live
              module object in memory. Returns the close timestamp.
     CANCEL / ESC / X button — discards all changes, returns close timestamp.
3. The returned timestamp is used by the caller to enforce a re-open
   debounce (prevents the gear button from immediately re-triggering).


CONTROLS
--------
Sliders   — used for numeric parameters (float and int).
            Drag the handle left/right, or tap anywhere on the track to
            jump. Values snap to the defined step size. The current value
            is displayed in the top-right of each row.

Toggles   — used for boolean parameters. Tap anywhere in the row to flip
            the switch. The row background turns green when enabled.

PPE Grid  — a 2-column checkbox grid listing all six PPE items:
            Mask, Hair Cap, Gloves, Boots, Apron, Long Sleeves.
            Tap a cell to require / un-require that item for compliance.

Reset     — hold for 1 second to revert all working values to what they
            were when the panel was opened (i.e. the last-saved state).
            A progress bar shows how far through the hold you are.

Save      — commits all current working values to config.py and to the
            live module, then closes the panel.

Cancel/X  — discards all unsaved changes and closes the panel.


LAYOUT SYSTEM
-------------
The panel uses a flat, scrollable list of layout items built by
_rebuild_layout(). Each item is a dict describing its type, y-position,
and height. Supported types:

  "section"  — a labelled divider row. Admin-only sections show an
               "ADMIN" badge when in admin mode.
  "float"    — a slider row for floating-point values.
  "int"      — a slider row for integer values.
  "bool"     — a toggle-switch row for boolean values.
  "ppe_grid" — the PPE requirements checkbox grid (always one instance).

The layout is rebuilt whenever the access mode changes so that admin-only
rows appear or disappear instantly.

Scrolling is driven by touch-drag on the content area. A thin scrollbar
on the right edge indicates position. Hit-test rectangles for sliders,
toggles, and PPE cells are repopulated on every render pass and stored in
dictionaries keyed by parameter attribute name.


PARAMETER DEFINITIONS  (_PARAMS)
---------------------------------
Each entry in _PARAMS is a tuple:
  (section, attr, display_label, type, min, max, step, user_visible)

  section       — groups the param under a section header.
  attr          — the exact attribute name in config.py.
  display_label — human-readable label shown on screen.
  type          — "float", "int", or "bool".
  min / max     — numeric bounds (None for bool).
  step          — snap granularity for sliders (None for bool).
  user_visible  — True: shown in both modes. False: admin-only.


CONFIG FILE I/O
---------------
Values are read from the live config module object on open().
On save, _write_config_file() uses regex substitution to update each
parameter line in config.py in-place, preserving all comments and
surrounding formatting. PPE requirement booleans inside the
PPE_REQUIREMENTS dict are updated with a separate regex pass.


COLOUR SCHEME
-------------
All colours are defined as BGR tuples (OpenCV convention) near the top of
the file. The palette uses a dark-navy base (_BG, _PANEL) with a green
accent (_ACCENT) for active/enabled states and amber (_AMBER / _MODE_ADMIN_FG)
for admin-mode highlights.


DEPENDENCIES
------------
  opencv-python (cv2)   — rendering, window management, mouse callbacks
  numpy                 — canvas array manipulation
  re                    — regex-based config.py rewriting
  math                  — gear icon geometry
  copy                  — deep-copying param values for reset support
  time                  — hold-timer logic (reset & mode-unlock)


HARDWARE TARGET
---------------
Designed for Raspberry Pi 5 with a 720×1280 (or similar) touchscreen.
TFLite thread count is capped at 4 to match the Pi 5's core count.
Touch input is received via OpenCV's mouse callback (works with most
Linux touch drivers that map touches to mouse events).
"""

import cv2
import numpy as np
import re
import math
import copy
import time

# ─── Colours ──────────────────────────────────────────────────────────────────
_BG       = (18,  20,  26)
_PANEL    = (22,  28,  38)
_HDR      = (14,  16,  22)
_SEC      = (28,  34,  46)
_DIV      = (35,  42,  54)
_ACCENT   = (0,  200, 110)
_RED      = (60,  60, 220)
_AMBER    = (0,  170, 240)       # amber in BGR, used for admin tinting
_TXT1     = (220, 228, 238)
_TXT2     = (135, 146, 163)
_DIM      = (58,  66,  82)
_TOOLTIP  = (90, 100, 118)
_SLD_BG   = (34,  40,  52)
_SLD_FILL = (0,  160,  80)
_SLD_HNDL = (0,  220, 120)
_TOG_ON   = (0,  190,  90)
_TOG_OFF  = (44,  52,  68)
_TOG_KNOB = (200, 210, 220)
_BTN_SAV  = (0,  120,  50)
_BTN_CAN  = (50,  50,  70)
_BTN_RST  = (50,  28,  28)
_WHITE    = (235, 240, 245)

# Mode button colours
_MODE_USER_BG   = (30,  34,  48)   # dim — locked / user
_MODE_USER_FG   = (100, 110, 130)
_MODE_ADMIN_BG  = (22,  40,  62)   # amber tint — admin
_MODE_ADMIN_FG  = (0,  180, 240)

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONTB = cv2.FONT_HERSHEY_DUPLEX

# ─── Layout constants ─────────────────────────────────────────────────────────
_HDR_H   = 64    # header height
_FTR_H   = 84    # footer height
_PAD     = 24    # horizontal padding
_SEC_H   = 38    # section header row
_BOOL_H  = 82    # bool toggle row (tall enough for tooltip)
_SLD_H   = 98    # slider row (tall enough for tooltip)
_PPE_H   = 210   # PPE requirement grid
_BTN_H   = 52    # footer button height
_SLD_T   = 10    # slider track thickness
_TOG_W   = 64    # toggle switch width
_TOG_H   = 32    # toggle switch height
_KNOB_R  = 14    # toggle knob radius

_RESET_HOLD_SEC  = 1.0   # hold duration to fire RESET
_UNLOCK_HOLD_SEC = 2.0   # hold duration to enter ADMIN mode

# ─── Tooltips ─────────────────────────────────────────────────────────────────
_TIPS = {
    "YOLO_CONFIDENCE":
        "Minimum YOLO score to report a detection. Lower = more detections, more false positives.",
    "CLASSIFIER_CONFIDENCE":
        "Minimum classifier score to accept a detected item. Raise if wrong items pass verification.",
    "FRAME_SKIP":
        "Process every Nth frame. 1 = every frame (slowest). 2-3 recommended for RPi 5.",
    "STABILITY_FRAMES":
        "Consecutive correct frames required before advancing to the next PPE item.",
    "TFLITE_NUM_THREADS":
        "CPU threads for TFLite inference. RPi 5 has 4 cores; 3-4 is optimal.",
    "USE_CLASSIFIER":
        "Run a second classifier model to confirm YOLO detections. More accurate, slightly slower.",
    "ENABLE_DEBUG_VIZ":
        "Draw pose skeleton and keypoint markers on the live camera feed.",
    "ENABLE_AUDIO":
        "Play prerecorded audio announcements for missing or verified PPE items.",
    "AUDIO_VOLUME":
        "Playback volume for all audio announcements. 0 = silent, 1 = full volume.",
    "AUDIO_COOLDOWN":
        "Minimum seconds between repeating the same audio announcement.",
    "AUDIO_ON_COMPLIANCE_CHANGE":
        "Announce when a PPE item transitions between missing and detected.",
    "AUDIO_ON_COMPLETION":
        "Announce when all required PPE items have been successfully verified.",
    "COMPLETION_AUTO_RESET":
        "Seconds to show the ALL CLEAR screen before auto-resetting for the next person.",
    "MARGIN_MASK":
        "Pixel tolerance when checking if the mask bbox overlaps face keypoints.",
    "MARGIN_HAIRCAP":
        "Pixel tolerance for hair cap vs head keypoints. Increase if cap is rejected too easily.",
    "MARGIN_GLOVES":
        "Pixel tolerance for glove bbox vs wrist keypoints.",
    "MARGIN_BOOTS":
        "Pixel tolerance for boot bbox vs ankle keypoints. Increase for tall boots.",
    "MARGIN_APRON":
        "Pixel tolerance for apron bbox vs torso keypoints.",
    "MARGIN_LONG_SLEEVES":
        "Pixel tolerance for sleeve bbox vs elbow/wrist keypoints.",
}

_PPE_TIPS = {
    "mask":         "Face mask covering nose and mouth",
    "haircap":      "Hair net or cap covering all hair",
    "gloves":       "Both left and right protective gloves",
    "boots":        "Both left and right safety boots",
    "apron":        "Full-length protective apron",
    "long_sleeves": "Long-sleeved garment on both arms",
}

# ─── Parameter master list ────────────────────────────────────────────────────
# Tuple: (section, attr, display_label, type, min, max, step, user_visible)
#
#   user_visible = True  → shown in both USER and ADMIN mode
#   user_visible = False → shown in ADMIN mode only
#
# PPE Requirements grid is always shown; it is inserted by _build_layout().
_PARAMS = [
    # ── Detection ──────────────────────────────────────────────────────────────
    ("DETECTION",  "YOLO_CONFIDENCE",           "YOLO Confidence",       "float", 0.10, 1.00, 0.05, True ),
    ("DETECTION",  "CLASSIFIER_CONFIDENCE",      "Classifier Confidence", "float", 0.10, 1.00, 0.05, False),
    # ── Performance (admin only) ───────────────────────────────────────────────
    ("PERFORMANCE","FRAME_SKIP",                 "Frame Skip",            "int",   1,    10,   1,    False),
    ("PERFORMANCE","STABILITY_FRAMES",           "Stability Frames",      "int",   1,    60,   1,    False),
    ("PERFORMANCE","TFLITE_NUM_THREADS",         "TFLite Threads",        "int",   1,    4,    1,    False),
    # ── Features (admin only) ─────────────────────────────────────────────────
    ("FEATURES",   "USE_CLASSIFIER",             "Use Classifier",        "bool",  None, None, None, False),
    ("FEATURES",   "ENABLE_DEBUG_VIZ",           "Debug Visualisation",   "bool",  None, None, None, False),
    # ── Audio ─────────────────────────────────────────────────────────────────
    ("AUDIO",      "ENABLE_AUDIO",               "Enable Audio",          "bool",  None, None, None, True ),
    ("AUDIO",      "AUDIO_VOLUME",               "Volume",                "float", 0.00, 1.00, 0.05, True ),
    ("AUDIO",      "AUDIO_COOLDOWN",             "Announcement Cooldown", "int",   1,    120,  1,    True ),
    ("AUDIO",      "AUDIO_ON_COMPLIANCE_CHANGE", "On Status Change",      "bool",  None, None, None, False),
    ("AUDIO",      "AUDIO_ON_COMPLETION",        "On Completion",         "bool",  None, None, None, False),
    # ── System ────────────────────────────────────────────────────────────────
    ("SYSTEM",     "COMPLETION_AUTO_RESET",      "Reset Cooldown (sec)",  "float", 2.0,  30.0, 0.5,  True ),
    # ── Margins (admin only) ──────────────────────────────────────────────────
    ("MARGINS",    "MARGIN_MASK",                "Mask",                  "int",   0,    80,   1,    False),
    ("MARGINS",    "MARGIN_HAIRCAP",             "Hair Cap",              "int",   0,    80,   1,    False),
    ("MARGINS",    "MARGIN_GLOVES",              "Gloves",                "int",   0,    80,   1,    False),
    ("MARGINS",    "MARGIN_BOOTS",               "Boots",                 "int",   0,    80,   1,    False),
    ("MARGINS",    "MARGIN_APRON",               "Apron",                 "int",   0,    80,   1,    False),
    ("MARGINS",    "MARGIN_LONG_SLEEVES",        "Long Sleeves",          "int",   0,    80,   1,    False),
]

# Sections that are admin-only (shown with ADMIN badge in section header)
_ADMIN_ONLY_SECTIONS = {"PERFORMANCE", "FEATURES", "MARGINS"}

_PPE_ITEMS  = ["mask", "haircap", "gloves", "boots", "apron", "long_sleeves"]
_PPE_LABELS = {
    "mask":         "Mask",
    "haircap":      "Hair Cap",
    "gloves":       "Gloves",
    "boots":        "Boots",
    "apron":        "Apron",
    "long_sleeves": "Long Sleeves",
}


# ==============================================================================
#   Gear icon helper  (also inlined in ui/ui.py as UIRenderer._draw_gear_icon)
# ==============================================================================

def draw_gear_icon(img, cx, cy, r=15, n_teeth=8,
                   fg=(190, 200, 210), bg=(18, 20, 26)):
    tooth_len  = int(r * 0.44)
    tooth_half = int(r * 0.28)
    cv2.circle(img, (cx, cy), r, fg, -1)
    step = 2 * math.pi / n_teeth
    for i in range(n_teeth):
        a        = i * step
        c, s     = math.cos(a), math.sin(a)
        px2, py2 = -s, c
        ix, iy   = cx + r * c,               cy + r * s
        ox, oy   = cx + (r + tooth_len) * c, cy + (r + tooth_len) * s
        pts = np.array([
            [int(ix - tooth_half * px2), int(iy - tooth_half * py2)],
            [int(ix + tooth_half * px2), int(iy + tooth_half * py2)],
            [int(ox + tooth_half * px2), int(oy + tooth_half * py2)],
            [int(ox - tooth_half * px2), int(oy - tooth_half * py2)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fg)
    cv2.circle(img, (cx, cy), max(1, r // 2), bg, -1)


# ==============================================================================
#   ConfigPanel
# ==============================================================================

class ConfigPanel:
    """
    Full-screen touch configuration panel with USER / ADMIN access modes.

    Usage:
        panel     = ConfigPanel("config.py")
        closed_at = panel.open(win_name, W, H, config_module)
        # closed_at is time.time() at close — pass to caller for debounce
    """

    # Mode button dimensions (used in header layout)
    _MODE_BTN_W = 110
    _MODE_BTN_H = 38

    def __init__(self, config_path: str = "config.py"):
        self.config_path = config_path

        # Working values and originals
        self._vals: dict = {}
        self._orig: dict = {}

        # Pending action set by mouse callback
        self._action: str | None = None

        # Current access mode (loaded from config.ADMIN_MODE on each open())
        self._is_admin: bool = False

        # Scroll state
        self._scroll_y   = 0
        self._max_scroll = 0

        # Slider drag state
        self._drag_key: str | None = None
        self._drag_x0:  int = 0
        self._drag_w:   int = 0

        # Scroll drag state
        self._scroll_drag    = False
        self._scroll_drag_y0 = 0
        self._scroll_start   = 0

        # Hold-to-reset state
        self._reset_press_time: float | None = None
        self._reset_armed: bool = False

        # Hold-to-unlock (user → admin) state
        self._lock_press_time: float | None = None
        self._lock_armed: bool = False

        # Layout cache
        self._layout:    list[dict] = []
        self._content_h: int = 0

        # Canvas size (set per open() call)
        self._W = 720
        self._H = 1280

        # Hit-test rects — rebuilt every render pass
        self._slider_rects: dict[str, tuple] = {}
        self._toggle_rects: dict[str, tuple] = {}
        self._ppe_rects:    dict[str, tuple] = {}
        self._btn_reset  = None
        self._btn_cancel = None
        self._btn_save   = None
        self._btn_close  = None
        self._btn_mode   = None   # USER ↔ ADMIN toggle

    # ──────────────────────────────────────────────────────────────────────────
    #   Public API
    # ──────────────────────────────────────────────────────────────────────────

    def open(self, win_name: str, width: int, height: int, config_module) -> float:
        """
        Display the panel and block until the user saves or cancels.

        Returns time.time() at the moment the panel closed.
        The caller should use this value to enforce a debounce before
        allowing the gear button to re-open the panel.
        """
        self._W = width
        self._H = height

        # Sync mode from live config
        self._is_admin = bool(getattr(config_module, 'ADMIN_MODE', False))

        self._load_from_module(config_module)
        self._orig = copy.deepcopy(self._vals)
        self._rebuild_layout()

        self._action           = None
        self._reset_press_time = None
        self._reset_armed      = False
        self._lock_press_time  = None
        self._lock_armed       = False

        cv2.setMouseCallback(win_name, self._on_mouse)

        result: bool | None = None

        while result is None:
            canvas = self._render()
            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:    # ESC → cancel
                result = False
            elif key == 13:  # ENTER → save
                result = True

            # Poll hold-to-reset
            if self._reset_armed and self._reset_press_time is not None:
                if time.time() - self._reset_press_time >= _RESET_HOLD_SEC:
                    self._load_defaults()
                    self._reset_press_time = None
                    self._reset_armed      = False

            # Poll hold-to-unlock (user → admin)
            if self._lock_armed and self._lock_press_time is not None:
                if time.time() - self._lock_press_time >= _UNLOCK_HOLD_SEC:
                    self._set_mode(config_module, admin=True)
                    self._lock_press_time = None
                    self._lock_armed      = False

            if self._action == "save":
                result = True
            elif self._action == "cancel":
                result = False
            elif self._action == "mode_tap":
                # Admin → User on single tap
                if self._is_admin:
                    self._set_mode(config_module, admin=False)
            self._action = None

        # ── Flush pending events so the caller's callback isn't re-triggered ─
        cv2.setMouseCallback(win_name, lambda *_: None)
        for _ in range(6):    # ~100 ms flush
            cv2.waitKey(16)

        closed_at = time.time()

        if result:
            self._write_config_file(config_module)
            self._apply_to_module(config_module)
            print("✓  Configuration saved to config.py")
        else:
            print("  Configuration changes discarded")

        return closed_at

    # ──────────────────────────────────────────────────────────────────────────
    #   Mode management
    # ──────────────────────────────────────────────────────────────────────────

    def _set_mode(self, config_module, admin: bool):
        """Switch access mode and write ADMIN_MODE to config.py immediately."""
        self._is_admin = admin
        setattr(config_module, 'ADMIN_MODE', admin)

        # Write just the ADMIN_MODE line — independent of SAVE
        try:
            with open(self.config_path, "r") as f:
                src = f.read()
            new_val = "True" if admin else "False"
            src = re.sub(
                r"^(ADMIN_MODE\s*=\s*)([^\n#]+)",
                rf"\g<1>{new_val}",
                src, flags=re.MULTILINE
            )
            with open(self.config_path, "w") as f:
                f.write(src)
        except Exception as e:
            print(f"⚠  Failed to write ADMIN_MODE: {e}")

        self._rebuild_layout()
        print(f"⚙  Mode switched to {'ADMIN' if admin else 'USER'}")

    # ──────────────────────────────────────────────────────────────────────────
    #   Value I/O
    # ──────────────────────────────────────────────────────────────────────────

    def _load_from_module(self, cfg):
        """Copy all tracked param values from the live config module."""
        for _, attr, _, typ, mn, _mx, _step, _ in _PARAMS:
            raw = getattr(cfg, attr, None)
            if raw is None:
                raw = mn if mn is not None else False
            if typ == "float":
                self._vals[attr] = float(raw)
            elif typ == "int":
                self._vals[attr] = int(raw)
            else:
                self._vals[attr] = bool(raw)
        req = getattr(cfg, "PPE_REQUIREMENTS", {})
        for k in _PPE_ITEMS:
            self._vals[f"ppe_{k}"] = bool(req.get(k, False))

    def _load_defaults(self):
        """Reset working values to what they were when the panel was opened."""
        self._vals = copy.deepcopy(self._orig)

    def _apply_to_module(self, cfg):
        """Push working values back into the live config module object."""
        for _, attr, _, _, _, _, _, _ in _PARAMS:
            if attr in self._vals:
                setattr(cfg, attr, self._vals[attr])
        req = getattr(cfg, "PPE_REQUIREMENTS", {})
        for k in _PPE_ITEMS:
            req[k] = self._vals.get(f"ppe_{k}", False)

    def _write_config_file(self, cfg):
        """Rewrite config.py on disk, preserving comments and formatting."""
        try:
            with open(self.config_path, "r") as f:
                src = f.read()

            for _, attr, _, typ, _, _, _, _ in _PARAMS:
                val = self._vals.get(attr)
                if val is None:
                    continue
                if typ == "bool":
                    new_str = "True" if val else "False"
                elif typ == "float":
                    new_str = f"{val:.2f}"
                else:
                    new_str = str(int(val))
                src = re.sub(
                    rf"^({re.escape(attr)}\s*=\s*)([^\n#]+)",
                    rf"\g<1>{new_str}",
                    src, flags=re.MULTILINE
                )

            for k in _PPE_ITEMS:
                val_str = "True" if self._vals.get(f"ppe_{k}", False) else "False"
                src = re.sub(
                    rf'("{re.escape(k)}"\s*:\s*)(True|False)',
                    rf"\g<1>{val_str}", src
                )

            with open(self.config_path, "w") as f:
                f.write(src)

        except Exception as e:
            print(f"⚠  Failed to write config.py: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    #   Layout
    # ──────────────────────────────────────────────────────────────────────────

    def _active_params(self):
        """Return the subset of _PARAMS visible in the current access mode."""
        if self._is_admin:
            return _PARAMS
        return [p for p in _PARAMS if p[7]]   # user_visible == True

    def _rebuild_layout(self):
        """
        Build (or rebuild) the flat list of layout items.
        PPE Requirements grid is always inserted before the AUDIO section.
        """
        params = self._active_params()
        items  = []
        y      = 0
        seen_sections = set()
        ppe_inserted  = False

        for section, attr, label, typ, mn, mx, step, _ in params:
            # Possibly insert PPE grid before AUDIO
            if section == "AUDIO" and not ppe_inserted:
                ppe_inserted = True
                items.append({"type": "section",  "y": y, "h": _SEC_H,
                              "label": "PPE REQUIREMENTS"})
                y += _SEC_H
                items.append({"type": "ppe_grid", "y": y, "h": _PPE_H})
                y += _PPE_H

            # Section header (once per section)
            if section not in seen_sections:
                seen_sections.add(section)
                items.append({"type": "section", "y": y, "h": _SEC_H,
                              "label": section, "admin_only": section in _ADMIN_ONLY_SECTIONS})
                y += _SEC_H

            row_h = _BOOL_H if typ == "bool" else _SLD_H
            items.append({
                "type":  typ,
                "y":     y,
                "h":     row_h,
                "attr":  attr,
                "label": label,
                "min":   mn,
                "max":   mx,
                "step":  step,
            })
            y += row_h

        # PPE grid at the very end if AUDIO section was never encountered
        if not ppe_inserted:
            items.append({"type": "section",  "y": y, "h": _SEC_H,
                          "label": "PPE REQUIREMENTS"})
            y += _SEC_H
            items.append({"type": "ppe_grid", "y": y, "h": _PPE_H})
            y += _PPE_H

        y += 12   # bottom padding
        self._layout    = items
        self._content_h = y
        self._max_scroll = max(0, self._content_h - (self._H - _HDR_H - _FTR_H))
        self._scroll_y   = min(self._scroll_y, self._max_scroll)

    # ──────────────────────────────────────────────────────────────────────────
    #   Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        canvas = np.full((self._H, self._W, 3), _BG, dtype=np.uint8)

        # ── Header ─────────────────────────────────────────────────────────
        self._fill(canvas, 0, 0, self._W, _HDR_H, _HDR)

        # Title (left side)
        cv2.putText(canvas, "CONFIGURATION",
                    (_PAD, 36), _FONTB, 0.72, _TXT1, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Tap SAVE to apply  |  ESC or X to cancel",
                    (_PAD, 54), _FONT, 0.29, _TXT2, 1, cv2.LINE_AA)

        # X close button (far right)
        xbx  = self._W - 52
        xby1 = 12
        xby2 = _HDR_H - 12
        self._fill(canvas, xbx, xby1, xbx + 38, xby2, (50, 28, 28))
        cv2.putText(canvas, "X", (xbx + 10, xby2 - 6),
                    _FONTB, 0.80, _RED, 2, cv2.LINE_AA)
        self._btn_close = (xbx, xby1, xbx + 38, xby2)

        # Mode button (left of X)
        mbw  = self._MODE_BTN_W
        mbh  = self._MODE_BTN_H
        mbx1 = xbx - mbw - 8
        mby1 = (_HDR_H - mbh) // 2
        mbx2 = mbx1 + mbw
        mby2 = mby1 + mbh
        self._draw_mode_button(canvas, mbx1, mby1, mbx2, mby2)
        self._btn_mode = (mbx1, mby1, mbx2, mby2)

        cv2.line(canvas, (0, _HDR_H), (self._W, _HDR_H), _DIV, 1)

        # ── Scrollable content ─────────────────────────────────────────────
        vis_top = _HDR_H
        vis_bot = self._H - _FTR_H

        # Clear hit-test rects before re-populating
        self._slider_rects = {}
        self._toggle_rects = {}
        self._ppe_rects    = {}

        for item in self._layout:
            it = vis_top + item["y"] - self._scroll_y
            ib = it + item["h"]
            if ib < vis_top or it > vis_bot:
                continue
            c1 = max(it, vis_top)
            c2 = min(ib, vis_bot)
            t  = item["type"]
            if t == "section":
                self._draw_section_header(
                    canvas, item["label"], it,
                    admin_badge=item.get("admin_only", False)
                )
            elif t in ("float", "int"):
                self._draw_slider_row(canvas, item, it, c1, c2)
            elif t == "bool":
                self._draw_bool_row(canvas, item, it, c1, c2)
            elif t == "ppe_grid":
                self._draw_ppe_grid(canvas, it, c1, c2)

        # Scrollbar
        if self._max_scroll > 0:
            sb_x  = self._W - 5
            sb_rng = vis_bot - vis_top
            th    = max(40, int(sb_rng * sb_rng / max(1, self._content_h)))
            tt    = int(vis_top + (sb_rng - th) * self._scroll_y
                        / max(1, self._max_scroll))
            cv2.line(canvas, (sb_x, vis_top), (sb_x, vis_bot), _DIV, 2)
            cv2.line(canvas, (sb_x, tt), (sb_x, tt + th), _ACCENT, 4)

        # ── Footer ─────────────────────────────────────────────────────────
        fy = self._H - _FTR_H
        cv2.line(canvas, (0, fy), (self._W, fy), _DIV, 1)
        self._fill(canvas, 0, fy, self._W, self._H, _HDR)

        btn_y = fy + 16
        mg    = 14
        bw    = (self._W - 4 * mg) // 3

        # RESET — hold to confirm
        rx1 = mg
        rx2 = rx1 + bw
        self._fill(canvas, rx1, btn_y, rx2, btn_y + _BTN_H, _BTN_RST)
        cv2.rectangle(canvas, (rx1, btn_y), (rx2, btn_y + _BTN_H), (80, 40, 40), 1)
        if self._reset_armed and self._reset_press_time is not None:
            frac  = min(1.0, (time.time() - self._reset_press_time) / _RESET_HOLD_SEC)
            pw    = int((rx2 - rx1 - 4) * frac)
            if pw > 0:
                self._fill(canvas, rx1 + 2, btn_y + _BTN_H - 6,
                           rx1 + 2 + pw, btn_y + _BTN_H - 2, _RED)
            self._put_btn_text(canvas, f"HOLD {int((1 - frac) * 100)}%",
                               rx1, btn_y, bw, _BTN_H, _RED)
        else:
            self._put_btn_text(canvas, "RESET",
                               rx1, btn_y, bw, _BTN_H, (140, 100, 100))
            cv2.putText(canvas, "hold 1s",
                        (rx1 + 6, btn_y + _BTN_H - 6),
                        _FONT, 0.26, (100, 70, 70), 1, cv2.LINE_AA)
        self._btn_reset = (rx1, btn_y, rx2, btn_y + _BTN_H)

        # CANCEL
        cx1 = rx1 + bw + mg
        cx2 = cx1 + bw
        self._fill(canvas, cx1, btn_y, cx2, btn_y + _BTN_H, _BTN_CAN)
        cv2.rectangle(canvas, (cx1, btn_y), (cx2, btn_y + _BTN_H), (80, 80, 100), 1)
        self._put_btn_text(canvas, "CANCEL", cx1, btn_y, bw, _BTN_H, _TXT2)
        self._btn_cancel = (cx1, btn_y, cx2, btn_y + _BTN_H)

        # SAVE
        sx1 = cx1 + bw + mg
        sx2 = sx1 + bw
        self._fill(canvas, sx1, btn_y, sx2, btn_y + _BTN_H, _BTN_SAV)
        cv2.rectangle(canvas, (sx1, btn_y), (sx2, btn_y + _BTN_H), (0, 180, 80), 1)
        self._put_btn_text(canvas, "SAVE", sx1, btn_y, bw, _BTN_H, _ACCENT)
        self._btn_save = (sx1, btn_y, sx2, btn_y + _BTN_H)

        return canvas

    # ── Mode button ───────────────────────────────────────────────────────────

    def _draw_mode_button(self, canvas, x1, y1, x2, y2):
        """Draw the USER ↔ ADMIN mode toggle button in the header."""
        w = x2 - x1
        h = y2 - y1

        if self._is_admin:
            bg, fg = _MODE_ADMIN_BG, _MODE_ADMIN_FG
            label  = "ADMIN MODE"
            hint   = "tap to exit"
        else:
            bg, fg = _MODE_USER_BG, _MODE_USER_FG
            label  = "ADMIN MODE"
            if self._lock_armed and self._lock_press_time is not None:
                frac = min(1.0, (time.time() - self._lock_press_time) / _UNLOCK_HOLD_SEC)
                hint = f"hold {int((1 - frac) * 100)}%"
            else:
                hint = "hold 2s"

        self._fill(canvas, x1, y1, x2, y2, bg)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fg, 1)

        # Progress bar (bottom strip) while holding to unlock
        if not self._is_admin and self._lock_armed and self._lock_press_time is not None:
            frac = min(1.0, (time.time() - self._lock_press_time) / _UNLOCK_HOLD_SEC)
            pw   = int((w - 4) * frac)
            if pw > 0:
                self._fill(canvas, x1 + 2, y2 - 5,
                           x1 + 2 + pw, y2 - 1, _MODE_ADMIN_FG)

        # Lock icon (simple unicode-ish drawn with lines)
        ix, iy = x1 + 7, y1 + h // 2
        if self._is_admin:
            # Open padlock — two horizontal lines (shackle open)
            cv2.rectangle(canvas, (ix, iy - 5), (ix + 8, iy + 5), fg, 1)
            cv2.line(canvas, (ix + 2, iy - 5), (ix + 2, iy - 9), fg, 2)
            cv2.line(canvas, (ix + 2, iy - 9), (ix + 8, iy - 9), fg, 2)
        else:
            # Closed padlock
            cv2.rectangle(canvas, (ix, iy - 4), (ix + 8, iy + 5), fg, 1)
            cv2.ellipse(canvas, (ix + 4, iy - 4), (4, 5), 0, 180, 360, fg, 1)

        # Text labels
        (lw, _), _ = cv2.getTextSize(label, _FONT, 0.30, 1)
        cv2.putText(canvas, label,
                    (x1 + (w - lw) // 2, y1 + 18),
                    _FONT, 0.30, fg, 1, cv2.LINE_AA)
        (hw, _), _ = cv2.getTextSize(hint, _FONT, 0.26, 1)
        cv2.putText(canvas, hint,
                    (x1 + (w - hw) // 2, y2 - 4),
                    _FONT, 0.26, fg, 1, cv2.LINE_AA)

    # ── Row drawers ───────────────────────────────────────────────────────────

    def _draw_section_header(self, canvas, label, y, admin_badge=False):
        self._fill(canvas, 0, y, self._W, y + _SEC_H, _SEC)
        cv2.line(canvas, (0, y), (self._W, y), _DIV, 1)
        cv2.putText(canvas, label,
                    (_PAD, y + 24), _FONTB, 0.42, _TXT2, 1, cv2.LINE_AA)

        # ADMIN badge on admin-only sections
        if admin_badge and self._is_admin:
            tag = "ADMIN"
            (tw, _), _ = cv2.getTextSize(tag, _FONT, 0.27, 1)
            tx = self._W - _PAD - tw - 6
            self._fill(canvas, tx - 4, y + 8, tx + tw + 4, y + 28, _MODE_ADMIN_BG)
            cv2.putText(canvas, tag, (tx, y + 24),
                        _FONT, 0.27, _MODE_ADMIN_FG, 1, cv2.LINE_AA)

        cv2.line(canvas, (0, y + _SEC_H - 1),
                 (self._W, y + _SEC_H - 1), _DIV, 1)

    def _draw_slider_row(self, canvas, item, y, clip_y1, clip_y2):
        attr  = item["attr"]
        label = item["label"]
        mn, mx, step = item["min"], item["max"], item["step"]
        val   = self._vals.get(attr, mn)
        typ   = item["type"]
        tip   = _TIPS.get(attr, "")

        self._fill(canvas, 0, max(y, clip_y1),
                   self._W, min(y + _SLD_H, clip_y2), _PANEL)
        cv2.line(canvas, (0, y + _SLD_H - 1),
                 (self._W, y + _SLD_H - 1), _DIV, 1)

        # Label
        cv2.putText(canvas, label,
                    (_PAD, y + 22), _FONT, 0.46, _TXT1, 1, cv2.LINE_AA)
        # Tooltip (up to 2 lines)
        for li, line in enumerate(_wrap(tip, 66)):
            cv2.putText(canvas, line,
                        (_PAD, y + 37 + li * 14),
                        _FONT, 0.27, _TOOLTIP, 1, cv2.LINE_AA)

        # Value readout (top-right)
        val_str = f"{val:.2f}" if typ == "float" else str(int(val))
        (vw, _), _ = cv2.getTextSize(val_str, _FONTB, 0.58, 2)
        cv2.putText(canvas, val_str,
                    (self._W - _PAD - vw, y + 22),
                    _FONTB, 0.58, _ACCENT, 2, cv2.LINE_AA)

        # Track
        tx1 = _PAD
        tx2 = self._W - _PAD - vw - 16
        ty  = y + 68
        tw  = max(1, tx2 - tx1)

        cv2.rectangle(canvas,
                      (tx1, ty - _SLD_T // 2), (tx2, ty + _SLD_T // 2),
                      _SLD_BG, -1)
        cv2.rectangle(canvas,
                      (tx1, ty - _SLD_T // 2), (tx2, ty + _SLD_T // 2),
                      _DIV, 1)
        frac   = max(0.0, min(1.0, (val - mn) / (mx - mn))) if mx > mn else 0.0
        fill_w = int(tw * frac)
        if fill_w > 0:
            cv2.rectangle(canvas,
                          (tx1, ty - _SLD_T // 2),
                          (tx1 + fill_w, ty + _SLD_T // 2),
                          _SLD_FILL, -1)
        hx = tx1 + fill_w
        cv2.circle(canvas, (hx, ty), 14, _SLD_HNDL, -1)
        cv2.circle(canvas, (hx, ty), 14, (0, 155, 85), 2)

        # Min / max labels under track
        mn_s = f"{mn:.1f}" if typ == "float" else str(int(mn))
        mx_s = f"{mx:.1f}" if typ == "float" else str(int(mx))
        cv2.putText(canvas, mn_s,
                    (tx1, y + _SLD_H - 5), _FONT, 0.26, _DIM, 1, cv2.LINE_AA)
        (mxw, _), _ = cv2.getTextSize(mx_s, _FONT, 0.26, 1)
        cv2.putText(canvas, mx_s,
                    (tx2 - mxw, y + _SLD_H - 5), _FONT, 0.26, _DIM, 1, cv2.LINE_AA)

        # Store for hit-testing (screen-space y values)
        self._slider_rects[attr] = (tx1, tx2, ty, y + _SLD_H)

    def _draw_bool_row(self, canvas, item, y, clip_y1, clip_y2):
        attr  = item["attr"]
        label = item["label"]
        val   = bool(self._vals.get(attr, False))
        tip   = _TIPS.get(attr, "")

        row_bg = (22, 38, 28) if val else _PANEL
        self._fill(canvas, 0, max(y, clip_y1),
                   self._W, min(y + _BOOL_H, clip_y2), row_bg)
        cv2.line(canvas, (0, y + _BOOL_H - 1),
                 (self._W, y + _BOOL_H - 1), _DIV, 1)

        # Label
        cv2.putText(canvas, label,
                    (_PAD, y + 26), _FONT, 0.48,
                    _TXT1 if val else _TXT2, 1, cv2.LINE_AA)
        # Tooltip
        for li, line in enumerate(_wrap(tip, 60)):
            cv2.putText(canvas, line,
                        (_PAD, y + 42 + li * 14),
                        _FONT, 0.27, _TOOLTIP, 1, cv2.LINE_AA)

        # Toggle switch
        tx  = self._W - _PAD - _TOG_W
        tgy = y + (_BOOL_H - _TOG_H) // 2 - 6
        tbg = _TOG_ON if val else _TOG_OFF
        mdy = tgy + _TOG_H // 2
        # Track (pill shape)
        cv2.rectangle(canvas,
                      (tx + _TOG_H // 2, tgy),
                      (tx + _TOG_W - _TOG_H // 2, tgy + _TOG_H),
                      tbg, -1)
        cv2.circle(canvas, (tx + _TOG_H // 2,          mdy), _TOG_H // 2, tbg, -1)
        cv2.circle(canvas, (tx + _TOG_W - _TOG_H // 2, mdy), _TOG_H // 2, tbg, -1)
        # Knob
        kx = (tx + _TOG_W - _KNOB_R - 2) if val else (tx + _KNOB_R + 2)
        cv2.circle(canvas, (kx, mdy), _KNOB_R, _TOG_KNOB, -1)
        cv2.circle(canvas, (kx, mdy), _KNOB_R, (155, 165, 175), 1)
        # ON / OFF label
        cv2.putText(canvas, "ON" if val else "OFF",
                    (tx - 42, y + 30), _FONT, 0.40,
                    _ACCENT if val else _DIM, 1, cv2.LINE_AA)

        # Whole row is tappable
        self._toggle_rects[attr] = (0, y, self._W, y + _BOOL_H)

    def _draw_ppe_grid(self, canvas, y, clip_y1, clip_y2):
        """2-column × 3-row grid of PPE requirement checkboxes."""
        self._fill(canvas, 0, max(y, clip_y1),
                   self._W, min(y + _PPE_H, clip_y2), _PANEL)

        cols   = 2
        cell_w = (self._W - 2 * _PAD) // cols
        cell_h = (_PPE_H - 8) // (len(_PPE_ITEMS) // cols)

        for idx, key in enumerate(_PPE_ITEMS):
            col = idx % cols
            row = idx // cols
            cx  = _PAD + col * cell_w
            cy  = y + row * cell_h + 4
            val = bool(self._vals.get(f"ppe_{key}", False))

            # Cell background
            cell_bg = (16, 32, 22) if val else (22, 26, 36)
            self._fill(canvas, cx, max(cy, clip_y1),
                       cx + cell_w - 4, min(cy + cell_h - 4, clip_y2), cell_bg)
            cv2.rectangle(canvas, (cx, cy),
                          (cx + cell_w - 4, cy + cell_h - 4),
                          (_ACCENT if val else _DIV), 1)

            # Checkbox
            chk = 28
            cx_ = cx + 10
            cy_ = cy + 10
            cv2.rectangle(canvas, (cx_, cy_),
                          (cx_ + chk, cy_ + chk),
                          (_TOG_ON if val else _TOG_OFF), -1)
            cv2.rectangle(canvas, (cx_, cy_),
                          (cx_ + chk, cy_ + chk), _DIV, 1)
            if val:
                cv2.line(canvas, (cx_ + 5,  cy_ + 14),
                         (cx_ + 11, cy_ + 22), _WHITE, 2)
                cv2.line(canvas, (cx_ + 11, cy_ + 22),
                         (cx_ + 23, cy_ + 6),  _WHITE, 2)

            # Label and tooltip
            lx = cx_ + chk + 8
            cv2.putText(canvas, _PPE_LABELS.get(key, key),
                        (lx, cy + 28), _FONT, 0.42,
                        _TXT1 if val else _TXT2, 1, cv2.LINE_AA)
            tip = _PPE_TIPS.get(key, "")
            if tip:
                cv2.putText(canvas, tip, (lx, cy + 44),
                            _FONT, 0.26, _TOOLTIP, 1, cv2.LINE_AA)

            self._ppe_rects[key] = (cx, cy, cx + cell_w - 4, cy + cell_h - 4)

    # ── Drawing primitives ────────────────────────────────────────────────────

    @staticmethod
    def _fill(img, x1, y1, x2, y2, color):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    @staticmethod
    def _put_btn_text(img, text, bx, by, bw, bh, color):
        (tw, th), _ = cv2.getTextSize(text, _FONTB, 0.52, 2)
        cv2.putText(img, text,
                    (bx + (bw - tw) // 2, by + (bh + th) // 2),
                    _FONTB, 0.52, color, 2, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    #   Mouse / touch callback
    # ──────────────────────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._on_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            self._on_drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._on_up(x, y)

    def _on_down(self, x, y):
        # ── Header / footer buttons ───────────────────────────────────────
        if self._hit(x, y, self._btn_close):
            self._action = "cancel";  return
        if self._hit(x, y, self._btn_save):
            self._action = "save";    return
        if self._hit(x, y, self._btn_cancel):
            self._action = "cancel";  return
        if self._hit(x, y, self._btn_reset):
            self._reset_press_time = time.time()
            self._reset_armed      = True
            return
        if self._hit(x, y, self._btn_mode):
            if self._is_admin:
                # Tap to exit admin — fires on down for responsiveness
                self._action = "mode_tap"
            else:
                # Begin hold-to-unlock
                self._lock_press_time = time.time()
                self._lock_armed      = True
            return

        # ── Sliders ───────────────────────────────────────────────────────
        for attr, (lx, rx, track_y, _) in self._slider_rects.items():
            if lx - 10 <= x <= rx + 10 and track_y - 22 <= y <= track_y + 22:
                self._drag_key = attr
                self._drag_x0  = lx
                self._drag_w   = rx - lx
                self._set_slider_from_x(attr, x)
                return

        # ── Bool toggles ──────────────────────────────────────────────────
        for attr, (lx, ty, rx, by) in self._toggle_rects.items():
            if lx <= x <= rx and ty <= y <= by:
                self._vals[attr] = not bool(self._vals.get(attr, False))
                return

        # ── PPE checkboxes ────────────────────────────────────────────────
        for key, (lx, ty, rx, by) in self._ppe_rects.items():
            if lx <= x <= rx and ty <= y <= by:
                k = f"ppe_{key}"
                self._vals[k] = not bool(self._vals.get(k, False))
                return

        # ── Begin scroll drag ─────────────────────────────────────────────
        if _HDR_H <= y <= self._H - _FTR_H:
            self._scroll_drag    = True
            self._scroll_drag_y0 = y
            self._scroll_start   = self._scroll_y

    def _on_drag(self, x, y):
        if self._drag_key is not None:
            self._set_slider_from_x(self._drag_key, x)
            return
        if self._scroll_drag:
            delta = self._scroll_drag_y0 - y
            self._scroll_y = max(0, min(self._max_scroll,
                                        self._scroll_start + delta))

    def _on_up(self, x, y):
        self._drag_key    = None
        self._scroll_drag = False
        # Release before hold completes → cancel both holds
        if self._reset_armed:
            self._reset_armed      = False
            self._reset_press_time = None
        if self._lock_armed:
            self._lock_armed      = False
            self._lock_press_time = None

    # ── Value helpers ─────────────────────────────────────────────────────────

    def _set_slider_from_x(self, attr, x):
        """Map a pixel x position to a param value and store it."""
        for p in _PARAMS:
            if p[1] == attr:
                _, _, _, typ, mn, mx, step, _ = p
                frac = max(0.0, min(1.0, (x - self._drag_x0)
                                   / max(1, self._drag_w)))
                raw  = mn + frac * (mx - mn)
                if typ == "int":
                    val = max(int(mn), min(int(mx),
                                          int(round(raw / step) * step)))
                else:
                    val = max(mn, min(mx, round(round(raw / step) * step, 4)))
                self._vals[attr] = val
                return

    @staticmethod
    def _hit(x, y, rect) -> bool:
        if rect is None:
            return False
        lx, ty, rx, by = rect
        return lx <= x <= rx and ty <= y <= by


# ─── Word-wrap helper ─────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    """Break `text` into lines of at most `width` characters."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]
