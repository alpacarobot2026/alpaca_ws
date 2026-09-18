#!/usr/bin/env python3
"""
Simple on-screen face display for Alpaca.

Uses tkinter in a separate process to avoid Tcl/Tk thread-affinity crashes.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Optional

try:
    import tkinter as tk
except Exception:  # pragma: no cover
    tk = None


class AlpacaFaceDisplay:
    """Minimal expressive face for a robot screen."""

    def __init__(self, fullscreen: bool = True):
        self.fullscreen = fullscreen
        self._enabled = tk is not None
        self._proc: Optional[mp.Process] = None
        self._cmd_q: Optional[mp.Queue] = None
        self._last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self) -> None:
        if not self._enabled or self._proc is not None:
            return
        self._cmd_q = mp.Queue(maxsize=100)
        self._proc = mp.Process(
            target=_face_ui_process,
            args=(self._cmd_q, self.fullscreen),
            daemon=True,
        )
        self._proc.start()

    def close(self) -> None:
        if self._cmd_q is not None:
            try:
                self._cmd_q.put_nowait(("__close__", "", ""))
            except Exception:
                pass

        if self._proc is not None:
            self._proc.join(timeout=1.5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
            self._proc = None

        if self._cmd_q is not None:
            try:
                self._cmd_q.close()
            except Exception:
                pass
            self._cmd_q = None

    def set_expression(self, expression: str, subtitle: str = "") -> None:
        if self._cmd_q is None:
            return
        try:
            self._cmd_q.put_nowait(("expression", expression, subtitle))
        except queue.Full:
            pass
        except Exception:
            pass

    def set_gaze(self, gaze_x: float) -> None:
        if self._cmd_q is None:
            return
        try:
            self._cmd_q.put_nowait(("gaze", float(gaze_x), ""))
        except queue.Full:
            pass
        except Exception:
            pass

    def set_rollout_preview(self, points) -> None:
        # Rollout preview has been removed from the face display.
        # Keep this method as a no-op for compatibility with existing callers.
        _ = points


def _face_ui_process(cmd_q: mp.Queue, fullscreen: bool) -> None:
    if tk is None:
        return

    root = tk.Tk()
    root.title("Alpaca Face")
    root.configure(bg="#f6f1e9")
    if fullscreen:
        root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _e: root.destroy())

    canvas = tk.Canvas(root, bg="#f6f1e9", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    state = {
        "expression": "idle",
        "subtitle": "Ready",
        "blink_until": 0.0,
        "gaze_target_x": 0.0,
        "gaze_render_x": 0.0,
    }

    def draw() -> None:
        _draw_face(canvas, state)

    def tick() -> None:
        while True:
            try:
                cmd_type, v1, v2 = cmd_q.get_nowait()
            except queue.Empty:
                break
            if cmd_type == "__close__":
                root.destroy()
                return
            if cmd_type == "expression":
                state["expression"] = str(v1)
                state["subtitle"] = str(v2)
            elif cmd_type == "gaze":
                g = max(-1.0, min(1.0, float(v1)))
                state["gaze_target_x"] = g

        # Smooth gaze: low-pass + slew-rate clamp + tiny deadband.
        alpha = 0.18
        max_step = 0.04
        prev = float(state["gaze_render_x"])
        target = float(state["gaze_target_x"])
        next_v = prev + alpha * (target - prev)
        delta = next_v - prev
        if delta > max_step:
            next_v = prev + max_step
        elif delta < -max_step:
            next_v = prev - max_step
        if abs(next_v) < 0.02:
            next_v = 0.0
        state["gaze_render_x"] = next_v

        draw()
        root.after(100, tick)

    root.after(0, tick)
    root.mainloop()


def _draw_face(c: "tk.Canvas", state: dict) -> None:
    c.delete("all")
    w = max(c.winfo_width(), 100)
    h = max(c.winfo_height(), 100)
    side = min(w, h)
    cx = w / 2
    cy = h / 2
    face_offset_y = -side * 0.06

    # Minimal, neutral HRI-style palette
    bg = "#141a22"
    panel = "#1b2430"
    eye_default = "#f0f2f4"
    mouth_color = "#b7cadf"
    subtitle_color = "#8ba0b8"

    c.configure(bg=bg)
    c.create_rectangle(0, 0, w, h, fill=panel, outline="")

    now = time.time()
    expression = state["expression"]
    subtitle = state["subtitle"] or ""
    if expression in {"idle", "listening"} and int(now * 2.0) % 19 == 0:
        state["blink_until"] = now + 0.15
    eye_open = now > state["blink_until"]

    eye_color = eye_default
    eye_outline = ""
    if expression == "listening":
        subtitle = subtitle or "Listening..."
    elif expression == "thinking":
        subtitle_color = "#b7c8de"  # cool steel-blue
        subtitle = subtitle or "Thinking..."
    elif expression == "speaking":
        subtitle = subtitle or "Speaking..."
    elif expression == "guiding":
        subtitle_color = "#e2c48a"  # warm sand-gold
        subtitle = subtitle or "Guiding you"
    elif expression == "arrived":
        subtitle_color = "#b9d7b2"  # soft sage
        subtitle = subtitle or "Arrived"
    else:
        subtitle = subtitle or "Ready"

    # Keep geometry aligned to the square center region even on wide screens.
    eye_y = cy - side * 0.12 + face_offset_y
    left_x = cx - side * 0.24
    right_x = cx + side * 0.24
    eye_w = side * 0.30
    eye_h = side * 0.14
    gaze_shift = side * 0.09 * float(state.get("gaze_render_x", 0.0))
    sclera_left_x = left_x
    sclera_right_x = right_x

    if eye_open:
        c.create_rectangle(
            sclera_left_x - eye_w / 2, eye_y - eye_h / 2, sclera_left_x + eye_w / 2, eye_y + eye_h / 2,
            fill=eye_color, outline="", width=1
        )
        c.create_rectangle(
            sclera_right_x - eye_w / 2, eye_y - eye_h / 2, sclera_right_x + eye_w / 2, eye_y + eye_h / 2,
            fill=eye_color, outline="", width=1
        )
        # Iris and pupil move inside each eye.
        iris_w = side * 0.030
        iris_h = side * 0.038
        pupil_r = side * 0.012
        iris_dx = gaze_shift * 0.80
        iris_bound = eye_w * 0.42
        iris_dx = max(-iris_bound, min(iris_bound, iris_dx))
        iris_color = "#000000"
        pupil_color = "#000000"
        li = sclera_left_x + iris_dx
        ri = sclera_right_x + iris_dx
        c.create_rectangle(li - iris_w, eye_y - iris_h, li + iris_w, eye_y + iris_h, fill=iris_color, outline="")
        c.create_rectangle(ri - iris_w, eye_y - iris_h, ri + iris_w, eye_y + iris_h, fill=iris_color, outline="")
    else:
        c.create_line(sclera_left_x - eye_w / 2, eye_y, sclera_left_x + eye_w / 2, eye_y, fill=eye_color, width=5)
        c.create_line(sclera_right_x - eye_w / 2, eye_y, sclera_right_x + eye_w / 2, eye_y, fill=eye_color, width=5)

    # Minimal mouth behavior
    mouth_y = cy + side * 0.23 + face_offset_y
    mouth_w = side * 0.26
    smile_width = max(10, int(side * 0.020))
    if expression == "speaking":
        phase = int(time.time() * 9) % 3
        h_open = [side * 0.03, side * 0.06, side * 0.04][phase]
        c.create_rectangle(
            cx - mouth_w * 0.45, mouth_y - h_open / 2,
            cx + mouth_w * 0.45, mouth_y + h_open / 2,
            fill=mouth_color, outline=""
        )
    elif expression == "thinking":
        c.create_rectangle(
            cx - mouth_w * 0.45, mouth_y - side * 0.010,
            cx + mouth_w * 0.45, mouth_y + side * 0.010,
            fill=mouth_color, outline=""
        )
    elif expression in {"guiding", "arrived"}:
        c.create_arc(
            cx - mouth_w * 0.65, mouth_y - side * 0.02,
            cx + mouth_w * 0.65, mouth_y + side * 0.10,
            start=200, extent=140, style=tk.ARC, outline=mouth_color, width=smile_width
        )
    elif expression == "listening":
        c.create_rectangle(
            cx - mouth_w * 0.16, mouth_y - side * 0.015,
            cx + mouth_w * 0.16, mouth_y + side * 0.045,
            fill=mouth_color, outline=""
        )
    else:
        # Default state: smiling.
        c.create_arc(
            cx - mouth_w * 0.55, mouth_y - side * 0.02,
            cx + mouth_w * 0.55, mouth_y + side * 0.09,
            start=200, extent=140, style=tk.ARC, outline=mouth_color, width=smile_width
        )

    if len(subtitle) > 42:
        subtitle = subtitle[:39].rstrip() + "..."
    subtitle_size = max(13, int(side * 0.032))
    # Subtle shadow/highlight for readability.
    c.create_text(
        cx + 2,
        h * 0.95 + 2,
        text=subtitle,
        fill="#0d1219",
        font=("Helvetica", subtitle_size, "bold"),
    )
    c.create_text(
        cx,
        h * 0.95,
        text=subtitle,
        fill=subtitle_color,
        font=("Helvetica", subtitle_size, "bold"),
    )
