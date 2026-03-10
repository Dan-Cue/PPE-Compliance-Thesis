"""
ui/loading.py — Startup Loading Screen
=======================================

OVERVIEW
--------
Displays an animated full-screen loading screen while main.py initialises
its subsystems one by one. The screen stays visible from the moment the
application launches until all components are ready, replacing a blank
window or terminal output with a polished operator-facing UI.

All cv2 calls happen inside methods, never at module level or in __init__,
to avoid Qt / display initialisation order crashes on Raspberry Pi / Linux.
The window created here is the same window reused by main.py's main loop,
so there is no flicker or re-open on transition.


USAGE  (in main.py)
-------------------
    from ui.loading import LoadingScreen
    loader = LoadingScreen(ui_cfg)

    loader.set_step("YOLO Detector")
    detector = PPEDetector(...)
    loader.mark_ok()

    loader.set_step("Camera")
    ...
    loader.mark_ok()

    loader.finish()    # green flash, then returns — window stays open

If a step raises an exception, call loader.mark_fail(str(e)) and then
loader.wait() to block until the operator presses Q.


PUBLIC API
----------
  set_step(name, detail="")
    Declare the next component about to load. Call this BEFORE the
    blocking import or model initialisation so the screen immediately
    shows a LOADING... badge for that step. Any previously active step
    is automatically promoted to OK.

  mark_ok(detail="")
    Mark the current step as successfully loaded. Plays a short animated
    progress-bar fill sweep before snapping the row badge to OK.

  mark_skip(detail="disabled")
    Mark the current step as intentionally skipped (e.g. classifier
    disabled in config). Badge shows SKIP in a dimmed colour.

  mark_fail(detail="")
    Mark the current step as failed. Switches the screen to the error
    state (red accents, INITIALISATION FAILED heading).

  finish()
    Call after all steps succeed. Shows a "SYSTEM READY" overlay with
    a spinning dot animation held for 1.8 seconds, then returns.
    The cv2 window remains open for main.py to reuse.

  wait()
    Blocks until the operator presses Q. Call after mark_fail() to hold
    the error screen open before the process exits.


SCREEN LAYOUT  (720 × 1280 portrait)
--------------------------------------
All vertical positions are absolute pixel values defined as module-level
constants, audited to fit safely within 720 × 1280 without overflow.

  Header (y ≈ 115–215)
    Two-line branding text centred on the canvas:
      Line 1 — "PERSONAL PROTECTIVE EQUIPMENT" in amber DUPLEX font.
      Line 2 — "VERIFICATION SYSTEM" in white TRIPLEX font (heavier).
    A thin divider separates the header from the body.

  Spinner (y ≈ 305–485)
    A background ring with an orbiting dot that advances each frame.
    Dot colour changes with global state:
      Loading — amber
      Ready   — green
      Failed  — red
    A small centre dot and a status text line sit below the ring.
    An elapsed-time counter updates every redraw.

  Component list card (y ≈ 520 onwards)
    A dark card listing each step declared via set_step(). Row count
    is capped so the card never overflows into the progress bar area.
    Each row contains:
      Left accent strip  — colour-coded by state (amber/green/red/dim).
      Active row tint    — blue highlight on the currently loading row.
      Component name     — uppercased, primary text colour.
      Detail sub-line    — shown only after the step completes or fails.
      Right status badge — LOADING... / OK / SKIP / FAIL.
    Rows are separated by thin dividers.

  Progress bar (below card)
    A 580 × 14 px bar centred on the canvas. Fraction = completed or
    skipped steps / total steps declared so far. A percentage label
    sits to the right of the "OVERALL PROGRESS" caption. The bar
    sweeps smoothly to its new value when mark_ok() is called.

  SYSTEM READY overlay (finish() state only)
    A semi-transparent green overlay covers the entire canvas, with a
    centred "SYSTEM READY" banner on an opaque dark-green backing card.

  Footer (y ≈ 1238–1258)
    A thin divider and a single centred line:
      Loading / ready state — "Please wait..."
      Failed state          — "Press Q to exit"


ANIMATION
---------
  Orbiting dot
    self._angle advances by 0.08–0.12 radians each redraw. The dot
    position is computed as (cx + R·cos θ, cy + R·sin θ).

  Progress bar fill sweep
    _animate_fill() redraws the screen 12 times with 18 ms delay,
    advancing the spinner angle each frame for a smooth sweep effect.

  SYSTEM READY hold
    finish() loops for 1.8 seconds (_READY_HOLD), advancing the spinner
    angle at 0.12 rad/frame at ~30 fps via cv2.waitKey(33).


DRAWING PRIMITIVES
------------------
Static methods with identical signatures to UIRenderer, allowing
copy-paste reuse without a shared base class:

  _fill(img, x1, y1, x2, y2, color, alpha)
    Filled and optionally alpha-blended rectangle. Coordinates clamped
    to image bounds.

  _text(img, text, x, y, font, scale, color, thickness)
    cv2.putText wrapper with LINE_AA anti-aliasing.

  _centered(img, text, y, font, scale, color, thickness, W)
    Like _text but horizontally centres the string within width W.

  _hdivider(img, x1, x2, y, color)
    Single-pixel horizontal divider line.


STEP STATES
-----------
  _ACTIVE  — currently loading; badge shows LOADING..., row highlighted.
  _OK      — loaded successfully; badge shows OK, green accent strip.
  _FAIL    — failed to load; badge shows FAIL, red accent strip.
  _SKIP    — intentionally skipped; badge shows SKIP, dim accent strip.


DEPENDENCIES
------------
  cv2 (opencv-python)  — window management, drawing, waitKey
  numpy                — canvas array creation and alpha blending
  math                 — orbiting dot position (cos, sin)
  time                 — elapsed timer and finish() hold duration
  ui/ui.py             — UIConfig (colours, fonts, spacing constants)
"""

import cv2
import numpy as np
import math
import time

from ui.ui import UIConfig

# ── Step states ───────────────────────────────────────────────────────────────
_ACTIVE = "active"
_OK     = "ok"
_SKIP   = "skip"
_FAIL   = "fail"

# ── Text strings ──────────────────────────────────────────────────────────────
_TXT_HDR1     = "PERSONAL PROTECTIVE EQUIPMENT"   # friend's bold header line 1
_TXT_HDR2     = "VERIFICATION SYSTEM"             # friend's bold header line 2
_TXT_LOADING  = "INITIALISING SYSTEM..."
_TXT_READY    = "SYSTEM READY"
_TXT_FAILED   = "INITIALISATION FAILED"
_TXT_WAIT     = "LOADING..."
_TXT_OK       = "OK"
_TXT_SKIP_LBL = "SKIP"
_TXT_FAIL_LBL = "FAIL"
_TXT_PROGRESS = "OVERALL PROGRESS"
_TXT_COMPS    = "COMPONENTS"
_TXT_WAIT_MSG = "Please wait..."
_TXT_QUIT_MSG = "Press Q to exit"

# ── Layout — every position is an absolute pixel value for 720 x 1280 ────────
_HDR1_Y      = 115    # "PERSONAL PROTECTIVE EQUIPMENT" baseline
_HDR2_Y      = 185    # "VERIFICATION SYSTEM" baseline
_HDR_DIV_Y   = 215    # thin divider below header
_SPIN_CY     = 385    # spinner centre Y
_SPIN_R      = 80     # background ring radius
_SPIN_ORB    = 62     # orbiting dot distance from centre
_SPIN_DOT    = 12     # orbiting dot radius
_STATUS_Y    = 485    # status text baseline (below spinner)
_CARD_Y1     = 520    # top of component list card
_STEP_H      = 48     # component row height
_BAR_W       = 580    # progress bar width  (friend's wider style)
_BAR_H       = 14     # progress bar height (friend's taller style)
_BAR_Y_OFF   = 28     # gap between card bottom and progress bar label
_FOOT_DIV_Y  = 1238   # footer divider
_FOOT_Y      = 1258   # footer text baseline

# "SYSTEM READY" flash — scale 2.0 → ~471 px, fits safely inside 720 px
_READY_SCALE = 2.0
_READY_THICK = 3
_READY_HOLD  = 1.8    # seconds


class LoadingScreen:
    """
    Startup loading screen — all cv2 calls on the main thread.
    Pass the UIConfig instance already created in main.py.
    """

    def __init__(self, cfg=None, window_name="PPE Verification System"):
        self.cfg      = cfg if cfg is not None else UIConfig()
        self.win      = window_name
        self._steps   = []
        self._start_t = time.time()
        self._angle   = 0.0      # spinner orbit angle (radians), advances each redraw

        # Open window exactly as main.py would — no imshow/waitKey here
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        if self.cfg.FULLSCREEN:
            cv2.setWindowProperty(
                self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.win,
                             self.cfg.OUTPUT_W, self.cfg.OUTPUT_H)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_step(self, name, detail=""):
        """
        Declare the next component about to load.
        Call BEFORE the blocking import / model load.
        Screen updates immediately to show LOADING... for this step.
        """
        for s in self._steps:
            if s["state"] == _ACTIVE:
                s["state"] = _OK
        self._steps.append({"name": name, "state": _ACTIVE,
                             "detail": detail, "_done_frac": 0.0})
        self._redraw("loading")

    def mark_ok(self, detail=""):
        """
        Mark current step as done.
        Plays a short animated progress-bar fill (friend's style) before
        snapping the badge to OK.
        """
        self._mark_last(_OK, detail)
        self._animate_fill()   # smooth fill animation

    def mark_skip(self, detail="disabled"):
        """Mark current step as intentionally skipped."""
        self._mark_last(_SKIP, detail)
        self._redraw("loading")

    def mark_fail(self, detail=""):
        """Mark current step as failed."""
        self._mark_last(_FAIL, detail)
        self._redraw("failed", fail_reason=detail)

    def finish(self):
        """
        All components loaded. Show animated SYSTEM READY flash, then return.
        The cv2 window stays open so main.py reuses it without flicker.
        """
        for s in self._steps:
            if s["state"] == _ACTIVE:
                s["state"] = _OK

        # Spin the orbiting dot while the READY flash is shown
        deadline = time.time() + _READY_HOLD
        while time.time() < deadline:
            self._angle += 0.12   # animate spinner during the hold
            self._redraw("ready")
            cv2.waitKey(33)       # ~30 fps

    def wait(self):
        """Block until Q is pressed. Call after a fatal failure."""
        while True:
            self._redraw("failed")
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_last(self, state, detail):
        for s in reversed(self._steps):
            if s["state"] == _ACTIVE:
                s["state"] = state
                if detail:
                    s["detail"] = detail
                return

    def _animate_fill(self, steps=12, delay_ms=18):
        """
        Smoothly animate the overall progress bar fill over `steps` frames.
        Friend's style: the bar visibly sweeps to the new value.
        """
        for _ in range(steps):
            self._angle += 0.08
            self._redraw("loading")
            cv2.waitKey(delay_ms)

    def _redraw(self, global_state, fail_reason=""):
        canvas = self._build(global_state, fail_reason)
        cv2.imshow(self.win, canvas)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Frame builder — combined design
    # ------------------------------------------------------------------

    def _build(self, global_state, fail_reason=""):
        cfg     = self.cfg
        W       = cfg.OUTPUT_W        # 720
        H       = cfg.OUTPUT_H        # 1280
        PAD     = cfg.PAD             # 18
        elapsed = time.time() - self._start_t

        # Dark background (friend's charcoal feel via UIConfig panel bg)
        canvas = np.full((H, W, 3), cfg.C_PANEL_BG, dtype=np.uint8)

        # Subtle border (friend's style)
        cv2.rectangle(canvas, (12, 12), (W - 12, H - 12), (50, 50, 50), 2)

        # ── Bold two-line branding header (friend's style) ─────────────
        # Line 1: PERSONAL PROTECTIVE EQUIPMENT — DUPLEX, medium weight
        t1, f1, s1, k1 = _TXT_HDR1, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2
        (w1, _), _ = cv2.getTextSize(t1, f1, s1, k1)
        cv2.putText(canvas, t1, ((W - w1) // 2, _HDR1_Y),
                    f1, s1, cfg.C_ACCENT_WARN, k1, cv2.LINE_AA)

        # Line 2: VERIFICATION SYSTEM — TRIPLEX, heavier weight
        t2, f2, s2, k2 = _TXT_HDR2, cv2.FONT_HERSHEY_TRIPLEX, 1.3, 3
        (w2, _), _ = cv2.getTextSize(t2, f2, s2, k2)
        cv2.putText(canvas, t2, ((W - w2) // 2, _HDR2_Y),
                    f2, s2, cfg.C_TEXT_PRIMARY, k2, cv2.LINE_AA)

        self._hdivider(canvas, PAD, W - PAD, _HDR_DIV_Y)

        # ── Orbiting-dot spinner (friend's style, positioned for 720x1280) ──
        cx, cy = W // 2, _SPIN_CY

        # Background ring
        ring_col = (40, 40, 40)
        cv2.circle(canvas, (cx, cy), _SPIN_R, ring_col, 4, cv2.LINE_AA)

        if global_state == "loading":
            spin_col = cfg.C_ACCENT_WARN
        elif global_state == "ready":
            spin_col = cfg.C_ACCENT
        else:
            spin_col = cfg.C_ACCENT_BAD

        # Orbiting dot — angle advances each _redraw call
        dot_x = int(cx + _SPIN_ORB * math.cos(self._angle))
        dot_y = int(cy + _SPIN_ORB * math.sin(self._angle))
        cv2.circle(canvas, (dot_x, dot_y), _SPIN_DOT, spin_col, -1, cv2.LINE_AA)

        # Small centre dot
        cv2.circle(canvas, (cx, cy), 6, spin_col, -1, cv2.LINE_AA)

        # Status text below spinner
        if global_state == "loading":
            status_txt = _TXT_LOADING
            status_col = cfg.C_ACCENT_WARN
        elif global_state == "ready":
            status_txt = "ALL SYSTEMS GO"
            status_col = cfg.C_ACCENT
        else:
            status_txt = _TXT_FAILED
            status_col = cfg.C_ACCENT_BAD

        self._centered(canvas, status_txt, _STATUS_Y,
                       cfg.FONT_SMALL, cfg.FS_PILL, status_col, cfg.FT_PILL, W)

        if global_state == "failed" and fail_reason:
            self._centered(canvas, fail_reason[:55], _STATUS_Y + 22,
                           cfg.FONT_SMALL, cfg.FS_FOOTER,
                           cfg.C_TEXT_SECONDARY, cfg.FT_FOOTER, W)

        # Elapsed timer
        self._centered(canvas, "{:.1f}s".format(elapsed), _STATUS_Y + 42,
                       cfg.FONT_SMALL, cfg.FS_FOOTER,
                       cfg.C_TEXT_DIM, cfg.FT_FOOTER, W)

        # ── Component list card (my design) ───────────────────────────
        card_x1 = PAD
        card_x2 = W - PAD

        # Cap rows so card never overflows above the progress bar area
        max_rows = (_FOOT_DIV_Y - _BAR_Y_OFF - _BAR_H - 20 - _CARD_Y1 - 22) // _STEP_H
        rows     = min(len(self._steps), max(max_rows, 1))
        card_h   = 22 + rows * _STEP_H + 8
        card_y2  = _CARD_Y1 + card_h

        self._fill(canvas, card_x1, _CARD_Y1, card_x2, card_y2, cfg.C_CARD_BG)
        self._hdivider(canvas, card_x1, card_x2, _CARD_Y1)
        self._hdivider(canvas, card_x1, card_x2, card_y2)

        self._text(canvas, _TXT_COMPS,
                   card_x1 + 10, _CARD_Y1 + 15,
                   cfg.FONT_SMALL, cfg.FS_FOOTER,
                   cfg.C_TEXT_DIM, cfg.FT_FOOTER)

        row_y = _CARD_Y1 + 22
        for step in self._steps[:rows]:
            state  = step["state"]
            name   = step["name"]
            detail = step.get("detail", "")

            if state == _ACTIVE:
                bar_col = cfg.C_ACCENT_WARN
                txt_col = cfg.C_TEXT_PRIMARY
                badge   = _TXT_WAIT
            elif state == _OK:
                bar_col = cfg.C_ACCENT
                txt_col = cfg.C_TEXT_PRIMARY
                badge   = _TXT_OK
            elif state == _FAIL:
                bar_col = cfg.C_ACCENT_BAD
                txt_col = cfg.C_ACCENT_BAD
                badge   = _TXT_FAIL_LBL
            else:   # SKIP
                bar_col = cfg.C_TEXT_DIM
                txt_col = cfg.C_TEXT_DIM
                badge   = _TXT_SKIP_LBL

            # Left accent strip (matches UIRenderer style)
            self._fill(canvas,
                       card_x1, row_y,
                       card_x1 + cfg.BORDER_ACCENT_W, row_y + _STEP_H - 4,
                       bar_col)

            # Active row highlight
            if state == _ACTIVE:
                self._fill(canvas,
                           card_x1 + cfg.BORDER_ACCENT_W, row_y,
                           card_x2, row_y + _STEP_H - 4,
                           cfg.C_ROW_ACTIVE)

            # Component name
            self._text(canvas, name.upper(),
                       card_x1 + 14, row_y + 22,
                       cfg.FONT_LABEL, cfg.FS_PILL, txt_col, cfg.FT_PILL)

            # Detail sub-line (hidden while still loading)
            if detail and state != _ACTIVE:
                self._text(canvas, detail,
                           card_x1 + 14, row_y + 38,
                           cfg.FONT_SMALL, cfg.FS_FOOTER,
                           cfg.C_TEXT_DIM, cfg.FT_FOOTER)

            # Right-side status badge
            (bw, _), _ = cv2.getTextSize(
                badge, cfg.FONT_SMALL, cfg.FS_PILL, cfg.FT_PILL)
            self._text(canvas, badge,
                       card_x2 - bw - 12, row_y + 22,
                       cfg.FONT_SMALL, cfg.FS_PILL, bar_col, cfg.FT_PILL)

            self._hdivider(canvas,
                           card_x1 + 10, card_x2 - 10,
                           row_y + _STEP_H - 4)
            row_y += _STEP_H

        # ── Animated progress bar (friend's wider/taller style) ────────
        n_steps = max(len(self._steps), 1)
        done    = sum(1 for s in self._steps if s["state"] in (_OK, _SKIP))
        frac    = done / n_steps

        bar_fill = (cfg.C_ACCENT_BAD if global_state == "failed" else
                    cfg.C_ACCENT     if global_state == "ready"  else
                    cfg.C_ACCENT_WARN)

        bar_y  = card_y2 + _BAR_Y_OFF
        bar_x  = (W - _BAR_W) // 2     # centred, friend's style

        # Label row: "OVERALL PROGRESS" left, "XX%" right — friend's placement
        pct_str = "{}%".format(int(frac * 100))
        self._text(canvas, _TXT_PROGRESS,
                   bar_x, bar_y,
                   cfg.FONT_SMALL, cfg.FS_FOOTER,
                   cfg.C_TEXT_DIM, cfg.FT_FOOTER)
        (pw, _), _ = cv2.getTextSize(
            pct_str, cfg.FONT_SMALL, 0.8, 2)          # friend's larger % text
        cv2.putText(canvas, pct_str,
                    (bar_x + _BAR_W - pw, bar_y),
                    cfg.FONT_SMALL, 0.8, bar_fill, 2, cv2.LINE_AA)

        # Bar body (friend's taller, centred bar)
        bar_top = bar_y + 10
        cv2.rectangle(canvas,
                      (bar_x, bar_top),
                      (bar_x + _BAR_W, bar_top + _BAR_H),
                      (60, 60, 60), -1)
        fill_w = max(0, int(_BAR_W * frac))
        if fill_w:
            cv2.rectangle(canvas,
                          (bar_x, bar_top),
                          (bar_x + fill_w, bar_top + _BAR_H),
                          bar_fill, -1)

        # ── SYSTEM READY overlay ───────────────────────────────────────
        # Scale 2.0 → 471 px wide, centred safely inside 720 px.
        if global_state == "ready":
            overlay = np.full((H, W, 3),
                              cfg.C_COMPLETE_OVERLAY, dtype=np.uint8)
            cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

            (tw, th), _ = cv2.getTextSize(
                _TXT_READY, cfg.FONT_TITLE, _READY_SCALE, _READY_THICK)
            bx = max(PAD, (W - tw) // 2)
            by = H // 2 + th // 2

            self._fill(canvas,
                       bx - 24, by - th - 20,
                       bx + tw + 24, by + 16,
                       (14, 50, 28), alpha=0.95)
            cv2.putText(canvas, _TXT_READY, (bx, by),
                        cfg.FONT_TITLE, _READY_SCALE,
                        cfg.C_COMPLETE_TEXT, _READY_THICK, cv2.LINE_AA)

        # ── Footer ─────────────────────────────────────────────────────
        self._hdivider(canvas, 0, W, _FOOT_DIV_Y)
        foot_msg = _TXT_QUIT_MSG if global_state == "failed" else _TXT_WAIT_MSG
        foot_col = cfg.C_TEXT_SECONDARY if global_state == "failed" else cfg.C_TEXT_DIM
        self._centered(canvas, foot_msg, _FOOT_Y,
                       cfg.FONT_SMALL, cfg.FS_FOOTER,
                       foot_col, cfg.FT_FOOTER, W)

        return canvas

    # ------------------------------------------------------------------
    # Drawing primitives — identical signatures to UIRenderer
    # ------------------------------------------------------------------

    @staticmethod
    def _fill(img, x1, y1, x2, y2, color, alpha=1.0):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
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
        cv2.putText(img, str(text), (int(x), int(y)),
                    font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _centered(img, text, y, font, scale, color, thickness, W):
        (tw, _), _ = cv2.getTextSize(str(text), font, scale, thickness)
        x = max(0, (W - tw) // 2)
        cv2.putText(img, str(text), (x, int(y)),
                    font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hdivider(img, x1, x2, y, color=(35, 42, 54)):
        cv2.line(img, (int(x1), int(y)), (int(x2), int(y)), color, 1)
