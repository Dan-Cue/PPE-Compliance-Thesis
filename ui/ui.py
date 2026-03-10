"""
ui/ui.py — UI Configuration & Renderer
=======================================

OVERVIEW
--------
All display logic for the PPE verification system lives here. The file
is split into two classes: UIConfig holds every visual setting as plain
attributes that can be edited without touching renderer code, and
UIRenderer consumes UIConfig to produce a fully composited display frame
on every loop iteration. main.py imports only these two classes and never
calls cv2 directly for display purposes.


CLASS: UIConfig
---------------
A plain data class — no methods, no logic. Edit its attributes to
customise the entire UI without reading any renderer code.

  ORIENTATION
    "portrait"   — info panel on top, camera feed below.
                   Default for a vertically mounted or rotated display.
    "landscape"  — info panel on the right, camera feed on the left.

  OUTPUT_W / OUTPUT_H
    Canvas resolution in pixels. Should match the display's native
    resolution when FULLSCREEN is True to avoid scaling artefacts.
    Portrait default: 720 × 1280.

  PANEL_HEIGHT_PORTRAIT / PANEL_WIDTH_LANDSCAPE
    Thickness of the info panel in pixels for each orientation.

  CAMERA_ROTATION
    Rotates the raw camera frame clockwise before display AND before
    YOLO / pose inference, so bounding boxes automatically stay aligned
    with the on-screen image. Supported values: 0, 90, 180, 270 (fast
    path via cv2.rotate), or any other integer/float for an arbitrary
    angle via cv2.warpAffine (note: arbitrary angles may clip corners).

  FULLSCREEN
    True  — borderless fullscreen window; OS taskbar is hidden.
    False — normal resizable window, useful for development.

  C_* colour constants
    All colours are BGR tuples following the OpenCV convention.
    Grouped by purpose:
      Backgrounds       C_PANEL_BG, C_CAM_BG
      Accent hierarchy  C_ACCENT (green), C_ACCENT_WARN (amber),
                        C_ACCENT_BAD (red)
      Text hierarchy    C_TEXT_PRIMARY, C_TEXT_SECONDARY, C_TEXT_DIM
      Checklist states  C_ITEM_VERIFIED, C_ITEM_ACTIVE, C_ITEM_PENDING
      Panel chrome      C_DIVIDER, C_ROW_ACTIVE, C_ROW_DONE, C_CARD_BG
      Bounding boxes    C_BBOX_OK (keypoint verified), C_BBOX_CHECKING
      Completion        C_COMPLETE_OVERLAY, C_COMPLETE_TEXT

  FONT_TITLE / FONT_LABEL / FONT_SMALL
    Any cv2.FONT_HERSHEY_* constant. Append | cv2.FONT_ITALIC to
    italicise. FS_* attributes control the scale for each role;
    FT_* attributes control the stroke thickness.

  Spacing constants
    PAD            — left/right padding inside the panel (pixels).
    ITEM_H         — checklist row height (pixels).
    BAR_H_MAIN     — overall progress bar height.
    BAR_H_STAB     — stability bar height.
    BORDER_ACCENT_W — left-border accent strip width on status cards.
    BBOX_THICKNESS — detection bounding box line thickness.

  LABEL_* text strings
    Every string displayed in the UI is defined here. Change any label
    without modifying renderer code.

  ICON_VERIFIED / ICON_ACTIVE / ICON_PENDING
    Single-character status icons used in the checklist.
    Default to ASCII ("v", ">", "o") for compatibility with Pi fonts.

  PULSE_PERIOD
    Period in seconds for the warning-pulse animation.

  COMPLETE_ALPHA
    Transparency (0.0–1.0) of the green overlay on the completion flash.


CLASS: UIRenderer
-----------------
Produces one composite display frame per main loop iteration.

  GEAR_RECT  (class variable)
    Screen-space bounding box (x1, y1, x2, y2) of the gear/settings
    button, updated every time a portrait frame is built. main.py reads
    this to detect touch or click events on the button.

  __init__(cfg)
    Accepts a UIConfig instance. Stores a _cam_transform tuple
    (scale, offset_x, offset_y) that is updated by _letterbox_camera()
    and consumed by _draw_bboxes() for coordinate remapping.

  rotate_frame(frame, angle)  [static]
    Rotates a camera frame clockwise by the given angle. 0/90/180/270
    use the fast cv2.rotate() path. Any other value uses cv2.warpAffine.
    Call this in main.py right after cv2.flip() and before passing the
    frame to detector.detect() or pose.process().

  build_frame(cam_frame, checker, fps, audio_enabled, detection_lost_cd,
              completion_cd, yolo_ok, keypoint_ok, active_detections,
              show_debug, pose_result, paused)
    Main entry point. Dispatches to _build_portrait() or
    _build_landscape() based on cfg.ORIENTATION, then returns the
    finished composite numpy array ready for cv2.imshow().


PORTRAIT LAYOUT  (_build_portrait / _make_portrait_panel)
----------------------------------------------------------
The canvas is split vertically:
  Top PANEL_HEIGHT_PORTRAIT pixels  — info panel
  Remaining pixels                  — letterboxed camera feed

Info panel contents (top to bottom):
  Header bar    — title, FPS readout with colour-coded performance
                  indicator (green ≥ 20, amber ≥ 10, red < 10), audio
                  dot, and the gear settings button (far right).
  Progress bar  — overall PPE completion fraction with stage count.
  Checklist tiles — one tile per PPE stage, laid out horizontally.
                  Each tile shows a top-colour accent strip, a status
                  icon, and an abbreviated stage name. Background tints
                  green when verified, blue when active.
  Status card   — fills remaining height. Shows either:
                    Verified state: "ALL PPE VERIFIED" banner with
                    entry-permitted label and auto-reset countdown.
                    Active state: current stage name in large text,
                    DETECTED / NOT FOUND and POS OK / POS -- pills,
                    stability progress bar, and a pulsing no-person
                    warning when detection is about to reset.


LANDSCAPE LAYOUT  (_build_landscape / _make_landscape_panel)
-------------------------------------------------------------
The canvas is split horizontally:
  Left (OUTPUT_W − PANEL_WIDTH_LANDSCAPE) pixels — letterboxed camera feed
  Right PANEL_WIDTH_LANDSCAPE pixels              — info panel

Info panel contents (top to bottom):
  Header        — title, subtitle.
  Progress bar  — same as portrait.
  Status card   — current stage or completion banner.
  Stability bar — stage elapsed time.
  Checklist     — vertical list with icon, name, and PASS pill.
  Warning strip — no-person countdown, auto-reset timer, PAUSED label.
  Footer        — FPS, audio status, keyboard shortcut hint.


BOUNDING BOX COORDINATE REMAPPING
-----------------------------------
YOLO detections are returned in raw camera-frame pixel coordinates.
When the camera frame is letterboxed onto the canvas a scale factor and
x/y offsets are applied. _draw_bboxes() remaps every bbox using the
transform cached by _letterbox_camera():

    display_x = int(raw_x × scale) + offset_x
    display_y = int(raw_y × scale) + offset_y

main.py passes raw detection dicts unchanged; the renderer handles all
coordinate conversion internally.


CAMERA OVERLAYS
---------------
  _draw_bboxes(canvas, active_detections, keypoint_ok, stage_name)
    Draws a coloured rectangle and confidence label for each active
    detection. Green when keypoints are verified; amber when only YOLO
    has fired. Coordinates are remapped from raw to display space.

  _draw_stage_prompt(canvas, stage_name, cw, ch)
    Displays a semi-transparent prompt at the bottom of the camera area
    telling the operator what PPE item to present next.

  _draw_complete_flash(canvas, cw, ch, comp_cd)
    Overlays a green-tinted "ALL CLEAR" banner with an auto-reset
    countdown timer when all PPE stages are verified.


DRAWING PRIMITIVES
------------------
All primitives are static methods and operate directly on numpy arrays.

  _fill(img, x1, y1, x2, y2, color, alpha)
    Filled rectangle. alpha < 1.0 blends with the existing canvas via
    cv2.addWeighted. Coordinates are clamped to image bounds.

  _text(img, text, x, y, font, scale, color, thickness)
    Thin wrapper around cv2.putText with LINE_AA anti-aliasing.

  _hdivider(img, x1, x2, y, color, thickness)
    Horizontal divider line.

  _progress_bar(img, x, y, w, h, fraction, bg_color, fill_color, border_color)
    Filled progress bar. fraction is clamped to 0.0–1.0.

  _pulse(period)
    Returns a 0.5–1.0 brightness multiplier that oscillates sinusoidally
    at the given period. Used for the no-person warning animation.

  _draw_gear_icon(img, cx, cy, r, n_teeth, fg, bg)
    Draws a filled gear icon centred at (cx, cy) using cv2.fillPoly for
    teeth and cv2.circle for the body and centre hole.

  _abbrev(name, max_len)
    Returns a short tile label from _ABBREV_MAP, or the first max_len
    characters of the name uppercased if the name is not in the map.


DEPENDENCIES
------------
  cv2 (opencv-python)           — all drawing, rotation, blending
  numpy                         — canvas array creation and ROI blending
  math                          — gear icon geometry (cos, sin, pi)
  time                          — pulse animation timing
  core/sequential_checker.py    — PPECheckStage (imported for type reference)
  core/verifier.py              — draw_verification_points (imported on demand
                                  inside the layout builders when show_debug
                                  is True, to avoid a circular import)
"""

import cv2
import numpy as np
import math
import time

from core.sequential_checker import PPECheckStage


# ==============================================================================
#   UIConfig  —  edit ONLY this class to change anything visual
# ==============================================================================

class UIConfig:
    ORIENTATION: str = "portrait"
    OUTPUT_W: int = 720
    OUTPUT_H: int = 1280
    PANEL_HEIGHT_PORTRAIT: int = 310
    PANEL_WIDTH_LANDSCAPE: int = 340
    CAMERA_ROTATION: int = 90
    FULLSCREEN: bool = True

    # ── Colors (all BGR tuples) ────────────────────────────────────────────────

    # Backgrounds
    C_PANEL_BG: tuple = (18,  20,  26)     # Info panel background
    C_CAM_BG:   tuple = (12,  14,  18)     # Letterbox fill around camera

    # Accent hierarchy
    C_ACCENT:      tuple = (0,  200, 110)  # Green  <- verified / all good
    C_ACCENT_WARN: tuple = (0,  160, 240)  # Amber  <- working on it
    C_ACCENT_BAD:  tuple = (50,  50, 220)  # Red    <- missing / error

    # Text hierarchy
    C_TEXT_PRIMARY:   tuple = (220, 228, 238)
    C_TEXT_SECONDARY: tuple = (135, 146, 163)
    C_TEXT_DIM:       tuple = (58,   66,  82)

    # Checklist item states
    C_ITEM_VERIFIED: tuple = (0,  200, 110)   # tick
    C_ITEM_ACTIVE:   tuple = (255, 210,  50)  # arrow  <- currently checking
    C_ITEM_PENDING:  tuple = (44,   52,  68)  # dot    <- not yet reached

    # Panel chrome
    C_DIVIDER:    tuple = (35,  42,  54)
    C_ROW_ACTIVE: tuple = (25,  32,  44)
    C_ROW_DONE:   tuple = (15,  34,  24)
    C_CARD_BG:    tuple = (22,  28,  38)

    # Camera feed bounding boxes
    C_BBOX_OK:       tuple = (0,  200, 110)  # Green <- keypoint verified
    C_BBOX_CHECKING: tuple = (0,  160, 240)  # Amber <- detected, kp pending

    # Completion overlay
    C_COMPLETE_OVERLAY: tuple = (0,   50,  24)
    C_COMPLETE_TEXT:    tuple = (0,  200, 110)

    # ── Fonts ──────────────────────────────────────────────────────────────────
    #   cv2.FONT_HERSHEY_SIMPLEX        clean, modern
    #   cv2.FONT_HERSHEY_DUPLEX         bolder simplex
    #   cv2.FONT_HERSHEY_COMPLEX        serif-ish
    #   cv2.FONT_HERSHEY_TRIPLEX        heavy serif
    #   cv2.FONT_HERSHEY_PLAIN          compact / monospace-like
    #   cv2.FONT_HERSHEY_COMPLEX_SMALL  small serif
    #   Append | cv2.FONT_ITALIC to italicise any of the above.
    FONT_TITLE: int = cv2.FONT_HERSHEY_DUPLEX
    FONT_LABEL: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL: int = cv2.FONT_HERSHEY_SIMPLEX

    # ── Font scales ────────────────────────────────────────────────────────────
    FS_HERO:    float = 3.2    # "ALL CLEAR" flash on camera
    FS_HEADING: float = 0.52   # Panel section headings
    FS_STAGE:   float = 3      # Current-stage name in card
    FS_ITEM:    float = 2      # Checklist item labels
    FS_PILL:    float = 0.31   # Small pills (DETECTED / PASS)
    FS_FOOTER:  float = 0.28   # Footer metadata

    # ── Font thicknesses ───────────────────────────────────────────────────────
    FT_HERO:    int = 3
    FT_HEADING: int = 2
    FT_STAGE:   int = 5
    FT_ITEM:    int = 1
    FT_PILL:    int = 1
    FT_FOOTER:  int = 1

    # ── Spacing & geometry ─────────────────────────────────────────────────────
    PAD:             int = 18   # Left/right padding inside panel (pixels)
    ITEM_H:          int = 30   # Checklist row height (pixels)
    BAR_H_MAIN:      int = 14   # Overall progress bar height
    BAR_H_STAB:      int = 5    # Stability bar height
    BORDER_ACCENT_W: int = 4    # Left-border accent strip thickness on cards
    BBOX_THICKNESS:  int = 2    # Detection bounding box line thickness

    # ── Display text ───────────────────────────────────────────────────────────
    #   Change any string here; no renderer code changes needed.
    LABEL_TITLE:        str = "PPE VERIFICATION"
    LABEL_SUBTITLE:     str = "SEQUENTIAL CHECK SYSTEM"
    LABEL_PROGRESS:     str = "PROGRESS"
    LABEL_NOW_CHECKING: str = "NOW CHECKING"
    LABEL_STABILITY:    str = "STABILITY"
    LABEL_STAGE_TIME:   str = "Stage time"
    LABEL_CHECKLIST:    str = "CHECKLIST"
    LABEL_COMPLETE:     str = "ALL PPE VERIFIED"
    LABEL_ENTRY_OK:     str = "ENTRY PERMITTED"
    LABEL_AUDIO_ON:     str = "AUDIO ON"
    LABEL_AUDIO_OFF:    str = "AUDIO OFF"
    LABEL_PAUSED:       str = "PAUSED"
    LABEL_NO_PERSON:    str = "NO PERSON DETECTED"
    LABEL_RESET_IN:     str = "Reset in"
    LABEL_AUTO_RESET:   str = "Auto-reset in"
    LABEL_PASS:         str = "PASS"
    LABEL_DETECTED:     str = "DETECTED"
    LABEL_NOT_FOUND:    str = "NOT FOUND"
    LABEL_POS_OK:       str = "POS OK"
    LABEL_POS_BAD:      str = "POS --"
    LABEL_KEYS:         str = "Q:quit  R:reset  A:audio  D:debug  G:config"
    LABEL_ALL_CLEAR:    str = "ALL CLEAR"
    LABEL_FPS_PREFIX:   str = "FPS"
    LABEL_SHOW_PROMPT:  str = "Show your white colored"

    # ── Icons ──────────────────────────────────────────────────────────────────
    #   Use ASCII ("V", ">", "o") if Unicode renders as boxes on your Pi.
    ICON_VERIFIED: str = "v"
    ICON_ACTIVE:   str = ">"
    ICON_PENDING:  str = "o"

    # ── Animation / misc ───────────────────────────────────────────────────────
    PULSE_PERIOD:   float = 1.2    # Warning-pulse period in seconds
    COMPLETE_ALPHA: float = 0.16   # Completion overlay alpha (0.0 - 1.0)


# ==============================================================================
#   UIRenderer
# ==============================================================================

class UIRenderer:
    """
    Produces a composite display frame each loop iteration.
    Internal state is limited to _cam_transform (set per frame) so bounding
    boxes can be remapped after letterboxing.

    GEAR_RECT — screen-space bounding box of the settings gear button,
                updated every time a portrait frame is built.
                main.py reads this to detect touch/click on the gear.
    """

    # Class-level rect so main.py can hit-test without a reference to the frame
    # Format: (x1, y1, x2, y2) in screen pixels
    GEAR_RECT: tuple = (0, 0, 0, 0)

    def __init__(self, cfg: UIConfig = None):
        self.cfg = cfg or UIConfig()
        self._cam_transform: tuple = (1.0, 0, 0)   # (scale, ox, oy) set per frame

    # --------------------------------------------------------------------------
    #  Camera rotation  (call in main.py right after cv2.flip)
    # --------------------------------------------------------------------------

    @staticmethod
    def rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate a camera frame by `angle` degrees clockwise.
        0/90/180/270 use the fast cv2.rotate() path.
        Any other value uses cv2.warpAffine (may clip corners).
        """
        angle = int(angle) % 360
        if angle == 0:
            return frame
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Arbitrary angle
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    # --------------------------------------------------------------------------
    #  Public entry point
    # --------------------------------------------------------------------------

    def build_frame(self, cam_frame, checker, fps, audio_enabled,
                    detection_lost_cd, completion_cd,
                    yolo_ok, keypoint_ok, active_detections,
                    show_debug, pose_result, paused) -> np.ndarray:
        """Return the complete composite display frame."""
        is_complete = checker.is_complete()
        if self.cfg.ORIENTATION == "portrait":
            return self._build_portrait(
                cam_frame, checker, fps, audio_enabled,
                detection_lost_cd, completion_cd,
                yolo_ok, keypoint_ok, active_detections,
                show_debug, pose_result, paused, is_complete)
        else:
            return self._build_landscape(
                cam_frame, checker, fps, audio_enabled,
                detection_lost_cd, completion_cd,
                yolo_ok, keypoint_ok, active_detections,
                show_debug, pose_result, paused, is_complete)

    # --------------------------------------------------------------------------
    #  Letterbox helper  (shared by both orientations)
    # --------------------------------------------------------------------------

    def _letterbox_camera(self, cam_frame: np.ndarray,
                          canvas_w: int, canvas_h: int) -> np.ndarray:
        """
        Scale cam_frame to fit canvas_w x canvas_h preserving aspect ratio.
        Records (scale, ox, oy) in self._cam_transform for bbox remapping.
        """
        fh, fw = cam_frame.shape[:2]
        scale  = min(canvas_w / fw, canvas_h / fh)
        new_w  = int(fw * scale)
        new_h  = int(fh * scale)
        ox     = (canvas_w - new_w) // 2
        oy     = (canvas_h - new_h) // 2

        self._cam_transform = (scale, ox, oy)   # <-- key fix for bbox offset

        resized = cv2.resize(cam_frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((canvas_h, canvas_w, 3),
                         self.cfg.C_CAM_BG, dtype=np.uint8)
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        return canvas

    # --------------------------------------------------------------------------
    #  PORTRAIT layout
    # --------------------------------------------------------------------------

    def _build_portrait(self, cam_frame, checker, fps, audio_enabled,
                        det_cd, comp_cd, yolo_ok, keypoint_ok,
                        active_detections, show_debug, pose_result, paused, is_complete):
        cfg   = self.cfg
        ow    = cfg.OUTPUT_W
        oh    = cfg.OUTPUT_H
        ph    = cfg.PANEL_HEIGHT_PORTRAIT
        cam_h = oh - ph

        panel = self._make_portrait_panel(
            checker, fps, audio_enabled,
            det_cd, comp_cd, yolo_ok, keypoint_ok, paused, is_complete,
            width=ow, height=ph)

        cam_canvas = self._letterbox_camera(cam_frame, ow, cam_h)

        if not is_complete:
            self._draw_bboxes(cam_canvas, active_detections, keypoint_ok,
                              checker.get_current_stage_name())
            if not yolo_ok:
                self._draw_stage_prompt(cam_canvas,
                                        checker.get_current_stage_name(), ow, cam_h)
            if show_debug and pose_result:
                from core.verifier import draw_verification_points
                draw_verification_points(cam_canvas, pose_result, ow, cam_h)
        else:
            self._draw_complete_flash(cam_canvas, ow, cam_h, comp_cd)

        cv2.line(panel, (0, ph - 1), (ow, ph - 1), cfg.C_DIVIDER, 2)
        return np.vstack([panel, cam_canvas])

    def _make_portrait_panel(self, checker, fps, audio_enabled,
                              det_cd, comp_cd, yolo_ok, keypoint_ok,
                              paused, is_complete, width, height):
        cfg    = self.cfg
        pw, ph = width, height
        panel  = np.full((ph, pw, 3), cfg.C_PANEL_BG, dtype=np.uint8)
        px     = cfg.PAD
        pr     = pw - cfg.PAD
        y      = 0

        # Header
        HDR_H = 40
        self._fill(panel, 0, 0, pw, HDR_H, (22, 26, 35))
        self._text(panel, cfg.LABEL_TITLE, px, 27,
                   cfg.FONT_TITLE, cfg.FS_HEADING, cfg.C_TEXT_PRIMARY, cfg.FT_HEADING)

        GEAR_SLOT  = 52        
        gear_cx    = pw - GEAR_SLOT // 2
        gear_cy    = HDR_H // 2
        gear_x1    = pw - GEAR_SLOT
        gear_y1    = 0
        gear_x2    = pw
        gear_y2    = HDR_H
        self._fill(panel, gear_x1, gear_y1, gear_x2, gear_y2, (28, 34, 48))
        cv2.line(panel, (gear_x1, 0), (gear_x1, HDR_H), (40, 48, 64), 1)
        self._draw_gear_icon(panel, cx=gear_cx, cy=gear_cy,
                             r=12, n_teeth=8,
                             fg=(155, 170, 190), bg=(28, 34, 48))
        
        UIRenderer.GEAR_RECT = (gear_x1, gear_y1, gear_x2, gear_y2)

        
        fps_color = (cfg.C_ACCENT      if fps >= 20 else
                     cfg.C_ACCENT_WARN if fps >= 10 else cfg.C_ACCENT_BAD)
        fps_str = f"{cfg.LABEL_FPS_PREFIX} {fps:4.1f}"
        (fw2, _), _ = cv2.getTextSize(fps_str, cfg.FONT_SMALL,
                                       cfg.FS_FOOTER, cfg.FT_FOOTER)
        
        pr_fps = pw - GEAR_SLOT - 8
        self._text(panel, fps_str, pr_fps - fw2, 27,
                   cfg.FONT_SMALL, cfg.FS_FOOTER, fps_color, cfg.FT_FOOTER)
        audio_color = cfg.C_ACCENT if audio_enabled else cfg.C_TEXT_DIM
        cv2.circle(panel, (pr_fps - fw2 - 14, 22), 5, audio_color, -1)

        y = HDR_H
        self._hdivider(panel, 0, pw, y)
        y += 8

        # Progress bar
        total    = len(checker.check_order) - 1
        verified = sum(1 for s in checker.check_order[:-1]
                       if checker.verified_items.get(s, False))
        frac     = verified / total if total > 0 else 0.0

        self._text(panel, f"{cfg.LABEL_PROGRESS}  {verified}/{total}",
                   px, y + 11, cfg.FONT_SMALL, cfg.FS_PILL,
                   cfg.C_TEXT_SECONDARY, cfg.FT_PILL)
        bar_x  = px + 115
        bar_fill = cfg.C_ACCENT if not is_complete else (0, 200, 80)
        self._progress_bar(panel, bar_x, y + 4, pr - bar_x, cfg.BAR_H_MAIN,
                           frac, (34, 40, 52), bar_fill, (50, 58, 72))
        y += 20

        # Checklist tiles (horizontal)
        stages = checker.check_order[:-1]
        n = len(stages)
        if n > 0:
            TILE_H = cfg.ITEM_H + 18
            tile_w = max(1, (pw - 2 * px) // n)
            tile_y = y + 4

            for i, stage in enumerate(stages):
                is_ver = checker.verified_items.get(stage, False)
                is_cur = (stage == checker.current_stage and not is_complete)
                name   = checker._get_stage_name(stage)
                tx     = px + i * tile_w

                bg = (cfg.C_ROW_DONE   if is_ver else
                      cfg.C_ROW_ACTIVE if is_cur else cfg.C_PANEL_BG)
                self._fill(panel, tx, tile_y, tx + tile_w - 1,
                           tile_y + TILE_H, bg)

                item_color = (cfg.C_ITEM_VERIFIED if is_ver else
                              cfg.C_ITEM_ACTIVE   if is_cur else cfg.C_ITEM_PENDING)
                cv2.rectangle(panel, (tx, tile_y),
                              (tx + tile_w - 1, tile_y + 3), item_color, -1)

                icon = (cfg.ICON_VERIFIED if is_ver else
                        cfg.ICON_ACTIVE   if is_cur else cfg.ICON_PENDING)
                (iw, _), _ = cv2.getTextSize(icon, cfg.FONT_SMALL,
                                              cfg.FS_PILL, cfg.FT_PILL)
                self._text(panel, icon,
                           tx + max(0, (tile_w - 1 - iw) // 2), tile_y + 20,
                           cfg.FONT_SMALL, cfg.FS_PILL, item_color, cfg.FT_PILL)

                abbrev = self._abbrev(name)
                (lw, _), _ = cv2.getTextSize(abbrev, cfg.FONT_SMALL,
                                              cfg.FS_FOOTER, cfg.FT_FOOTER)
                self._text(panel, abbrev,
                           tx + max(0, (tile_w - 1 - lw) // 2),
                           tile_y + TILE_H - 5,
                           cfg.FONT_SMALL, cfg.FS_FOOTER, item_color, cfg.FT_FOOTER)

            y = tile_y + TILE_H + 6

        self._hdivider(panel, px, pr, y)
        y += 8

        # Status strip (remaining height)
        remaining = ph - y - 4

        if is_complete:
            self._fill(panel, px - 4, y, pr + 4, y + remaining, (16, 44, 26))
            cv2.rectangle(panel, (px - 4, y),
                          (px - 4 + cfg.BORDER_ACCENT_W, y + remaining),
                          cfg.C_ACCENT, -1)
            mid = y + remaining // 2
            self._text(panel, cfg.LABEL_COMPLETE, px + 8, mid - 4,
                       cfg.FONT_TITLE, cfg.FS_HEADING, cfg.C_ACCENT, cfg.FT_HEADING)
            self._text(panel, cfg.LABEL_ENTRY_OK, px + 8, mid + 18,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_SECONDARY, cfg.FT_PILL)
            if comp_cd is not None:
                self._text(panel, f"{cfg.LABEL_AUTO_RESET} {comp_cd:.0f}s",
                           px + 8, y + remaining - 8,
                           cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
        else:
            self._fill(panel, px - 4, y, pr + 4, y + remaining, cfg.C_CARD_BG)
            bord = cfg.C_ITEM_ACTIVE if yolo_ok else cfg.C_TEXT_DIM
            cv2.rectangle(panel, (px - 4, y),
                          (px - 4 + cfg.BORDER_ACCENT_W, y + remaining), bord, -1)

            mid = y + remaining // 2 - 4
            self._text(panel, cfg.LABEL_NOW_CHECKING, px + 8, y + 14,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
            self._text(panel, checker.get_current_stage_name().upper(),
                       px + 8, mid,
                       cfg.FONT_TITLE, cfg.FS_STAGE, cfg.C_ITEM_ACTIVE, cfg.FT_STAGE)

            det_str = cfg.LABEL_DETECTED  if yolo_ok    else cfg.LABEL_NOT_FOUND
            kp_str  = cfg.LABEL_POS_OK    if keypoint_ok else cfg.LABEL_POS_BAD
            det_col = cfg.C_ACCENT        if yolo_ok    else cfg.C_ACCENT_BAD
            kp_col  = cfg.C_ACCENT        if keypoint_ok else cfg.C_TEXT_DIM
            cur_r = pr
            for lbl, col in reversed([(det_str, det_col), (kp_str, kp_col)]):
                (lw2, _), _ = cv2.getTextSize(lbl, cfg.FONT_SMALL,
                                               cfg.FS_PILL, cfg.FT_PILL)
                self._text(panel, lbl, cur_r - lw2, mid + 16,
                           cfg.FONT_SMALL, cfg.FS_PILL, col, cfg.FT_PILL)
                cur_r -= lw2 + 14

            stab_frac  = checker.stability_counter / max(checker.required_stability, 1)
            stab_color = cfg.C_ACCENT if stab_frac >= 1.0 else cfg.C_ACCENT_WARN
            stab_y     = mid + 28
            self._text(panel, cfg.LABEL_STABILITY, px + 8, stab_y + 8,
                       cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
            self._progress_bar(panel, px + 84, stab_y + 2, pr - px - 84,
                               cfg.BAR_H_STAB, stab_frac,
                               (34, 40, 52), stab_color, (50, 58, 72))

            warn_y = y + remaining - 14
            if det_cd is not None and det_cd > 0:
                pulse = self._pulse(cfg.PULSE_PERIOD)
                wc = tuple(int(c * pulse) for c in cfg.C_ACCENT_BAD)
                self._text(panel,
                           f"! {cfg.LABEL_NO_PERSON}  |  "
                           f"{cfg.LABEL_RESET_IN} {det_cd:.1f}s",
                           px + 8, warn_y,
                           cfg.FONT_SMALL, cfg.FS_FOOTER, wc, cfg.FT_FOOTER)
            if comp_cd is not None:
                self._text(panel, f"{cfg.LABEL_AUTO_RESET} {comp_cd:.0f}s",
                           px + 8, warn_y,
                           cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
            if paused:
                (pw3, _), _ = cv2.getTextSize(cfg.LABEL_PAUSED, cfg.FONT_SMALL,
                                               cfg.FS_PILL, cfg.FT_PILL)
                self._text(panel, cfg.LABEL_PAUSED, pr - pw3, warn_y,
                           cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_ACCENT_WARN, cfg.FT_PILL)

        return panel

    # --------------------------------------------------------------------------
    #  LANDSCAPE layout
    # --------------------------------------------------------------------------

    def _build_landscape(self, cam_frame, checker, fps, audio_enabled,
                         det_cd, comp_cd, yolo_ok, keypoint_ok,
                         active_detections, show_debug, pose_result, paused, is_complete):
        cfg   = self.cfg
        ow    = cfg.OUTPUT_W
        oh    = cfg.OUTPUT_H
        pw    = cfg.PANEL_WIDTH_LANDSCAPE
        cam_w = ow - pw

        cam_canvas = self._letterbox_camera(cam_frame, cam_w, oh)

        if not is_complete:
            self._draw_bboxes(cam_canvas, active_detections, keypoint_ok,
                              checker.get_current_stage_name())
            if not yolo_ok:
                self._draw_stage_prompt(cam_canvas,
                                        checker.get_current_stage_name(), cam_w, oh)
            if show_debug and pose_result:
                from core.verifier import draw_verification_points
                draw_verification_points(cam_canvas, pose_result, cam_w, oh)
        else:
            self._draw_complete_flash(cam_canvas, cam_w, oh, comp_cd)

        panel = self._make_landscape_panel(
            checker, fps, audio_enabled,
            det_cd, comp_cd, yolo_ok, keypoint_ok, paused, is_complete,
            width=pw, height=oh)

        cv2.line(cam_canvas, (cam_w - 1, 0), (cam_w - 1, oh), cfg.C_DIVIDER, 2)
        return np.hstack([cam_canvas, panel])

    def _make_landscape_panel(self, checker, fps, audio_enabled,
                               det_cd, comp_cd, yolo_ok, keypoint_ok,
                               paused, is_complete, width, height):
        cfg    = self.cfg
        pw, ph = width, height
        panel  = np.full((ph, pw, 3), cfg.C_PANEL_BG, dtype=np.uint8)
        px, pr = cfg.PAD, pw - cfg.PAD
        y = 0

        HDR_H = 65
        self._fill(panel, 0, 0, pw, HDR_H, (22, 26, 35))
        self._text(panel, cfg.LABEL_TITLE, px, 28,
                   cfg.FONT_TITLE, cfg.FS_HEADING, cfg.C_TEXT_PRIMARY, cfg.FT_HEADING)
        self._text(panel, cfg.LABEL_SUBTITLE, px, 50,
                   cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
        self._hdivider(panel, 0, pw, HDR_H)
        y = HDR_H + 14

        total    = len(checker.check_order) - 1
        verified = sum(1 for s in checker.check_order[:-1]
                       if checker.verified_items.get(s, False))
        frac     = verified / total if total > 0 else 0.0
        cnt_str  = f"{verified}/{total}"
        (cw2, _), _ = cv2.getTextSize(cnt_str, cfg.FONT_SMALL, cfg.FS_PILL, cfg.FT_PILL)
        self._text(panel, cfg.LABEL_PROGRESS, px, y + 10,
                   cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_SECONDARY, cfg.FT_PILL)
        self._text(panel, cnt_str, pr - cw2, y + 10,
                   cfg.FONT_SMALL, cfg.FS_PILL,
                   cfg.C_ACCENT if verified > 0 else cfg.C_TEXT_DIM, cfg.FT_PILL)
        y += 16
        bar_fill = cfg.C_ACCENT if not is_complete else (0, 200, 80)
        self._progress_bar(panel, px, y, pr - px, cfg.BAR_H_MAIN,
                           frac, (34, 40, 52), bar_fill, (50, 58, 72))
        y += 18
        self._hdivider(panel, px, pr, y)
        y += 12

        if not is_complete:
            card_h = 80
            self._fill(panel, px - 4, y, pr + 4, y + card_h, cfg.C_CARD_BG)
            bord = cfg.C_ITEM_ACTIVE if yolo_ok else cfg.C_TEXT_DIM
            cv2.rectangle(panel, (px - 4, y),
                          (px - 4 + cfg.BORDER_ACCENT_W, y + card_h), bord, -1)
            self._text(panel, cfg.LABEL_NOW_CHECKING, px + 6, y + 18,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
            self._text(panel, checker.get_current_stage_name().upper(),
                       px + 6, y + 52,
                       cfg.FONT_TITLE, cfg.FS_STAGE, cfg.C_ITEM_ACTIVE, cfg.FT_STAGE)
            det_str = cfg.LABEL_DETECTED  if yolo_ok    else cfg.LABEL_NOT_FOUND
            kp_str  = cfg.LABEL_POS_OK    if keypoint_ok else cfg.LABEL_POS_BAD
            det_col = cfg.C_ACCENT        if yolo_ok    else cfg.C_ACCENT_BAD
            kp_col  = cfg.C_ACCENT        if keypoint_ok else cfg.C_TEXT_DIM
            self._text(panel, det_str, px + 6, y + 70,
                       cfg.FONT_SMALL, cfg.FS_PILL, det_col, cfg.FT_PILL)
            self._text(panel, "|", px + 90, y + 70,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
            self._text(panel, kp_str, px + 100, y + 70,
                       cfg.FONT_SMALL, cfg.FS_PILL, kp_col, cfg.FT_PILL)
            y += card_h + 10

            stab_frac  = checker.stability_counter / max(checker.required_stability, 1)
            stab_color = cfg.C_ACCENT if stab_frac >= 1.0 else cfg.C_ACCENT_WARN
            self._text(panel, cfg.LABEL_STABILITY, px, y + 8,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
            self._progress_bar(panel, px + 84, y + 3, pr - px - 84, cfg.BAR_H_STAB,
                               stab_frac, (34, 40, 52), stab_color, (50, 58, 72))
            y += 16
            self._text(panel,
                       f"{cfg.LABEL_STAGE_TIME}:  {checker.get_elapsed_time():.1f}s",
                       px, y + 8, cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
            y += 20
        else:
            card_h = 80
            self._fill(panel, px - 4, y, pr + 4, y + card_h, (16, 44, 26))
            cv2.rectangle(panel, (px - 4, y),
                          (px - 4 + cfg.BORDER_ACCENT_W, y + card_h), cfg.C_ACCENT, -1)
            self._text(panel, cfg.LABEL_COMPLETE, px + 6, y + 40,
                       cfg.FONT_TITLE, cfg.FS_STAGE, cfg.C_ACCENT, cfg.FT_STAGE)
            self._text(panel, cfg.LABEL_ENTRY_OK, px + 6, y + 62,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_SECONDARY, cfg.FT_PILL)
            y += card_h + 12

        self._hdivider(panel, px, pr, y)
        y += 12
        self._text(panel, cfg.LABEL_CHECKLIST, px, y + 8,
                   cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_TEXT_DIM, cfg.FT_PILL)
        y += 20

        for stage in checker.check_order[:-1]:
            is_ver = checker.verified_items.get(stage, False)
            is_cur = (stage == checker.current_stage and not is_complete)
            name   = checker._get_stage_name(stage).upper()
            ry2    = y + cfg.ITEM_H
            if is_ver:
                self._fill(panel, px - 4, y, pr + 4, ry2, cfg.C_ROW_DONE)
            elif is_cur:
                self._fill(panel, px - 4, y, pr + 4, ry2, cfg.C_ROW_ACTIVE)
            icon  = (cfg.ICON_VERIFIED if is_ver else
                     cfg.ICON_ACTIVE   if is_cur else cfg.ICON_PENDING)
            color = (cfg.C_ITEM_VERIFIED if is_ver else
                     cfg.C_ITEM_ACTIVE   if is_cur else cfg.C_ITEM_PENDING)
            self._text(panel, icon, px + 2, y + 20,
                       cfg.FONT_SMALL, cfg.FS_ITEM, color, cfg.FT_ITEM)
            self._text(panel, name, px + 22, y + 20,
                       cfg.FONT_SMALL, cfg.FS_ITEM, color, cfg.FT_ITEM)
            if is_ver:
                (pw4, _), _ = cv2.getTextSize(cfg.LABEL_PASS, cfg.FONT_SMALL,
                                               cfg.FS_PILL, cfg.FT_PILL)
                self._text(panel, cfg.LABEL_PASS, pr - pw4, y + 20,
                           cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_ITEM_VERIFIED, cfg.FT_PILL)
            y += cfg.ITEM_H

        self._hdivider(panel, px, pr, y)
        y += 8

        if det_cd is not None and det_cd > 0:
            pulse = self._pulse(cfg.PULSE_PERIOD)
            wc    = tuple(int(c * pulse) for c in cfg.C_ACCENT_BAD)
            self._text(panel, f"! {cfg.LABEL_NO_PERSON}",
                       px, y + 14, cfg.FONT_SMALL, cfg.FS_PILL, wc, cfg.FT_PILL)
            self._text(panel, f"  {cfg.LABEL_RESET_IN} {det_cd:.1f}s",
                       px, y + 30, cfg.FONT_SMALL, cfg.FS_FOOTER, wc, cfg.FT_FOOTER)
            y += 38
        if comp_cd is not None:
            self._text(panel, f"{cfg.LABEL_AUTO_RESET} {comp_cd:.0f}s",
                       px, y + 14,
                       cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
            y += 22
        if paused:
            self._text(panel, cfg.LABEL_PAUSED, px, y + 14,
                       cfg.FONT_SMALL, cfg.FS_PILL, cfg.C_ACCENT_WARN, cfg.FT_PILL)
            y += 22

        foot_y = ph - 50
        self._hdivider(panel, 0, pw, foot_y)
        fy = foot_y + 14
        fps_color = (cfg.C_ACCENT      if fps >= 20 else
                     cfg.C_ACCENT_WARN if fps >= 10 else cfg.C_ACCENT_BAD)
        self._text(panel, f"{cfg.LABEL_FPS_PREFIX} {fps:4.1f}", px, fy,
                   cfg.FONT_SMALL, cfg.FS_FOOTER, fps_color, cfg.FT_FOOTER)
        audio_str   = cfg.LABEL_AUDIO_ON  if audio_enabled else cfg.LABEL_AUDIO_OFF
        audio_color = cfg.C_ACCENT        if audio_enabled else cfg.C_TEXT_DIM
        self._text(panel, audio_str, pw // 2 - 28, fy,
                   cfg.FONT_SMALL, cfg.FS_FOOTER, audio_color, cfg.FT_FOOTER)
        self._text(panel, cfg.LABEL_KEYS, px, foot_y + 36,
                   cfg.FONT_SMALL, cfg.FS_FOOTER, cfg.C_TEXT_DIM, cfg.FT_FOOTER)
        return panel

    # --------------------------------------------------------------------------
    #  Camera overlays
    # --------------------------------------------------------------------------

    def _draw_bboxes(self, canvas: np.ndarray, active_detections: list,
                     keypoint_ok: bool, stage_name: str):
        """
        Draw detection bboxes on the letterboxed canvas.
        Raw YOLO coordinates are remapped via self._cam_transform.
        """
        if not active_detections:
            return
            
        cfg = self.cfg
        scale, ox, oy = self._cam_transform

        for det in active_detections:
            x1r, y1r, x2r, y2r = det["bbox"]
            conf = det["confidence"]

            # Remap: raw camera space -> display canvas space
            x1 = int(x1r * scale) + ox
            y1 = int(y1r * scale) + oy
            x2 = int(x2r * scale) + ox
            y2 = int(y2r * scale) + oy

            color = cfg.C_BBOX_OK if keypoint_ok else cfg.C_BBOX_CHECKING
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, cfg.BBOX_THICKNESS)

            label = f"{stage_name}  {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cfg.FONT_SMALL, cfg.FS_PILL, cfg.FT_PILL)
            bg_y1 = max(y1 - lh - 10, 0)
            bg_y2 = max(y1 - 2, lh + 10)
            self._fill(canvas, x1, bg_y1, x1 + lw + 10, bg_y2, color, alpha=0.82)
            self._text(canvas, label, x1 + 4, bg_y2 - 4,
                       cfg.FONT_SMALL, cfg.FS_PILL, (12, 12, 12), cfg.FT_PILL)

    def _draw_stage_prompt(self, canvas, stage_name, cw, ch):
        cfg  = self.cfg
        text = f"{cfg.LABEL_SHOW_PROMPT} {stage_name}"
        (tw, th), _ = cv2.getTextSize(text, cfg.FONT_LABEL, 0.62, 2)
        bx = cw // 2 - tw // 2
        by = ch - 36
        self._fill(canvas, bx - 12, by - th - 6, bx + tw + 12, by + 8,
                   (14, 18, 22), alpha=0.80)
        self._text(canvas, text, bx, by, cfg.FONT_LABEL, 0.62,
                   cfg.C_ITEM_ACTIVE, 2)

    def _draw_complete_flash(self, canvas, cw, ch, comp_cd):
        cfg = self.cfg
        self._fill(canvas, 0, 0, cw, ch, cfg.C_COMPLETE_OVERLAY,
                   alpha=cfg.COMPLETE_ALPHA)
        text = cfg.LABEL_ALL_CLEAR
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
                                        cfg.FS_HERO, cfg.FT_HERO)
        bx = cw // 2 - tw // 2
        by = ch // 2
        self._fill(canvas, bx - 20, by - th - 18, bx + tw + 20, by + 12,
                   (14, 50, 28), alpha=0.92)
        cv2.putText(canvas, text, (bx, by),
                    cv2.FONT_HERSHEY_DUPLEX, cfg.FS_HERO,
                    cfg.C_COMPLETE_TEXT, cfg.FT_HERO, cv2.LINE_AA)
        if comp_cd is not None:
            sub = f"{cfg.LABEL_AUTO_RESET} {comp_cd:.0f}s"
            (sw, _), _ = cv2.getTextSize(sub, cfg.FONT_SMALL, 0.58, 1)
            self._text(canvas, sub, cw // 2 - sw // 2, by + 40,
                       cfg.FONT_SMALL, 0.58, cfg.C_TEXT_SECONDARY, 1)

    # --------------------------------------------------------------------------
    #  Drawing primitives
    # --------------------------------------------------------------------------

    @staticmethod
    def _fill(img, x1, y1, x2, y2, color, alpha=1.0):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        if alpha >= 1.0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            roi  = img[y1:y2, x1:x2]
            rect = np.full(roi.shape, color, dtype=np.uint8)
            cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0, roi)
            img[y1:y2, x1:x2] = roi

    @staticmethod
    def _text(img, text, x, y, font, scale, color, thickness):
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hdivider(img, x1, x2, y, color=(35, 42, 54), thickness=1):
        cv2.line(img, (x1, y), (x2, y), color, thickness)

    @staticmethod
    def _progress_bar(img, x, y, w, h, fraction,
                       bg_color, fill_color, border_color=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        if border_color:
            cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)
        fill_w = max(0, int(w * min(fraction, 1.0)))
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)

    @staticmethod
    def _pulse(period=1.2) -> float:
        t = time.time() % period
        return 0.5 + 0.5 * abs(np.sin(np.pi * t / period))

    @staticmethod
    def _draw_gear_icon(img, cx, cy, r=13, n_teeth=8,
                        fg=(160, 175, 195), bg=(30, 36, 50)):
        """Filled gear icon centred at (cx, cy). bg = hole fill colour."""
        tooth_len  = int(r * 0.44)
        tooth_half = int(r * 0.28)
        cv2.circle(img, (cx, cy), r, fg, -1)
        step = 2 * math.pi / n_teeth
        for i in range(n_teeth):
            a  = i * step
            c, s  = math.cos(a), math.sin(a)
            px2, py2 = -s, c
            ix, iy = cx + r * c,            cy + r * s
            ox, oy = cx + (r + tooth_len) * c, cy + (r + tooth_len) * s
            pts = np.array([
                [int(ix - tooth_half * px2), int(iy - tooth_half * py2)],
                [int(ix + tooth_half * px2), int(iy + tooth_half * py2)],
                [int(ox + tooth_half * px2), int(oy + tooth_half * py2)],
                [int(ox - tooth_half * px2), int(oy - tooth_half * py2)],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], fg)
        cv2.circle(img, (cx, cy), max(1, r // 2), bg, -1)

    _ABBREV_MAP = {
        "Hair Cap":    "HAIR",
        "Mask":        "MASK",
        "Long Sleeves":"SLVS",
        "Apron":       "APRON",
        "Left Glove":  "L.GLV",
        "Right Glove": "R.GLV",
        "Left Boot":   "L.BOOT",
        "Right Boot":  "R.BOOT",
    }

    @classmethod
    def _abbrev(cls, name: str, max_len: int = 6) -> str:
        return cls._ABBREV_MAP.get(name, name[:max_len].upper())
