"""Cinematic OpenCV interface inspired by the generated UI concept."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Palette:
    background: tuple[int, int, int] = (20, 11, 7)
    panel: tuple[int, int, int] = (33, 24, 16)
    panel_light: tuple[int, int, int] = (45, 33, 22)
    cyan: tuple[int, int, int] = (232, 214, 57)
    cyan_dim: tuple[int, int, int] = (105, 87, 26)
    white: tuple[int, int, int] = (242, 246, 247)
    muted: tuple[int, int, int] = (170, 179, 185)
    green: tuple[int, int, int] = (88, 240, 54)
    gold: tuple[int, int, int] = (90, 182, 232)


class CinematicUI:
    width = 1600
    height = 900
    camera_rect = (20, 78, 1180, 740)
    panel_rect = (1220, 78, 360, 800)

    def __init__(self) -> None:
        self.colors = Palette()
        self.button_rects = {
            "snap": (1240, 594, 320, 58),
            "foreground": (1240, 663, 320, 58),
            "cloak": (1240, 732, 320, 58),
            "recapture": (1240, 805, 154, 46),
            "mask": (1406, 805, 154, 46),
        }

    @staticmethod
    def _rounded_rectangle(
        image: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        radius: int = 16,
        thickness: int = -1,
    ) -> None:
        x, y, w, h = rect
        if thickness < 0:
            cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), color, -1)
            cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), color, -1)
            for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                           (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
                cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)
        else:
            cv2.line(image, (x + radius, y), (x + w - radius, y), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x + radius, y + h), (x + w - radius, y + h), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x, y + radius), (x, y + h - radius), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x + w, y + radius), (x + w, y + h - radius), color, thickness, cv2.LINE_AA)
            for center, start, end in (
                ((x + radius, y + radius), 180, 270), ((x + w - radius, y + radius), 270, 360),
                ((x + radius, y + h - radius), 90, 180), ((x + w - radius, y + h - radius), 0, 90),
            ):
                cv2.ellipse(image, center, (radius, radius), 0, start, end, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = np.zeros((height, width, 3), np.uint8)
        x = (width - resized.shape[1]) // 2
        y = (height - resized.shape[0]) // 2
        result[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return result

    def _status_card(
        self, canvas: np.ndarray, y: int, label: str, value: float, ready: bool
    ) -> None:
        self._rounded_rectangle(canvas, (1240, y, 320, 78), self.colors.panel_light, 18)
        self._rounded_rectangle(canvas, (1240, y, 320, 78), self.colors.cyan_dim, 18, 1)
        cv2.rectangle(canvas, (1260, y + 21), (1296, y + 57), self.colors.muted, 2, cv2.LINE_AA)
        cv2.circle(canvas, (1278, y + 34), 7, self.colors.white, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (1278, y + 53), (12, 10), 180, 0, 180, self.colors.white, -1, cv2.LINE_AA)
        self._text(canvas, label, (1314, y + 33), 0.52, self.colors.white)
        cv2.rectangle(canvas, (1314, y + 51), (1536, y + 57), (54, 57, 58), -1)
        cv2.rectangle(canvas, (1314, y + 51), (1314 + int(222 * np.clip(value, 0, 1)), y + 57), self.colors.cyan, -1)
        cv2.circle(canvas, (1538, y + 23), 6, self.colors.green if ready else self.colors.gold, -1, cv2.LINE_AA)

    def _mode_button(self, canvas: np.ndarray, mode: str, active: bool) -> None:
        rect = self.button_rects[mode]
        fill = (50, 38, 24) if active else self.colors.panel_light
        border = self.colors.cyan if active else (102, 94, 78)
        if active:
            glow = canvas.copy()
            self._rounded_rectangle(glow, (rect[0] - 4, rect[1] - 4, rect[2] + 8, rect[3] + 8), self.colors.cyan_dim, 17, 4)
            cv2.addWeighted(glow, 0.25, canvas, 0.75, 0, canvas)
        self._rounded_rectangle(canvas, rect, fill, 14)
        self._rounded_rectangle(canvas, rect, border, 14, 2 if active else 1)
        icon_x, icon_y = rect[0] + 44, rect[1] + rect[3] // 2
        if mode == "snap":
            points = np.array([[icon_x, icon_y - 17], [icon_x - 10, icon_y + 2], [icon_x, icon_y + 2], [icon_x - 5, icon_y + 18], [icon_x + 12, icon_y - 5], [icon_x + 2, icon_y - 5]])
            cv2.fillPoly(canvas, [points], self.colors.white)
            label = "SNAP MODE"
        elif mode == "foreground":
            cv2.circle(canvas, (icon_x, icon_y - 9), 8, self.colors.white, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (icon_x, icon_y + 13), (14, 15), 180, 0, 180, self.colors.white, -1, cv2.LINE_AA)
            label = "FOREGROUND"
        else:
            cv2.circle(canvas, (icon_x, icon_y - 9), 8, self.colors.white, 2, cv2.LINE_AA)
            cv2.ellipse(canvas, (icon_x, icon_y + 10), (14, 18), 180, 180, 360, self.colors.white, 2, cv2.LINE_AA)
            label = "CLOAK"
        self._text(canvas, label, (rect[0] + 90, rect[1] + 37), 0.58, self.colors.white, 1)

    def hit_test(self, x: int, y: int) -> str | None:
        for name, (bx, by, bw, bh) in self.button_rects.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return name
        return None

    def render(
        self,
        camera_frame: np.ndarray,
        mode: str,
        yolo_active: bool,
        background_ready: bool,
        confidence: float,
        fps: float,
        gesture_status: str,
        animation_progress: float | None,
        mask_preview: bool,
    ) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), self.colors.background, np.uint8)
        # Header
        self._text(canvas, "I N V I S I B L E", (30, 46), 0.92, self.colors.white, 2)
        cv2.line(canvas, (398, 35), (1182, 35), self.colors.cyan_dim, 1, cv2.LINE_AA)
        cv2.circle(canvas, (1430, 34), 7, self.colors.green, -1, cv2.LINE_AA)
        self._text(canvas, "LIVE", (1447, 41), 0.58, self.colors.white)

        # Camera stage
        x, y, w, h = self.camera_rect
        fitted = self._fit(camera_frame, w, h)
        canvas[y : y + h, x : x + w] = fitted
        self._rounded_rectangle(canvas, self.camera_rect, self.colors.cyan, 16, 2)
        cv2.line(canvas, (x + 28, y + h + 35), (x + w - 28, y + h + 35), self.colors.cyan_dim, 2, cv2.LINE_AA)
        timeline = animation_progress if animation_progress is not None else 0.0
        cv2.line(canvas, (x + 28, y + h + 35), (x + 28 + int((w - 56) * timeline), y + h + 35), self.colors.cyan, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x + 28 + int((w - 56) * timeline), y + h + 35), 6, self.colors.cyan, -1, cv2.LINE_AA)

        # Right panel and snap status
        self._rounded_rectangle(canvas, self.panel_rect, self.colors.panel, 22)
        self._rounded_rectangle(canvas, self.panel_rect, (43, 55, 62), 22, 1)
        center = (1400, 205)
        cv2.circle(canvas, center, 94, self.colors.cyan_dim, 1, cv2.LINE_AA)
        cv2.circle(canvas, center, 76, self.colors.cyan, 3, cv2.LINE_AA)
        cv2.circle(canvas, center, 55, (35, 54, 60), 1, cv2.LINE_AA)
        # Simple snap-hand glyph.
        cv2.ellipse(canvas, (1393, 207), (25, 38), 18, 190, 350, self.colors.white, 3, cv2.LINE_AA)
        cv2.line(canvas, (1387, 203), (1426, 179), self.colors.white, 3, cv2.LINE_AA)
        cv2.line(canvas, (1392, 214), (1429, 194), self.colors.white, 3, cv2.LINE_AA)
        label = "SNAP TO DISAPPEAR" if mode == "snap" else mode.upper() + " MODE"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
        self._text(canvas, label, (center[0] - text_size[0] // 2, 329), 0.55, self.colors.white)
        short_status = gesture_status.upper()[:35]
        status_size = cv2.getTextSize(short_status, cv2.FONT_HERSHEY_DUPLEX, 0.35, 1)[0]
        self._text(canvas, short_status, (center[0] - status_size[0] // 2, 352), 0.35, self.colors.muted)

        self._status_card(canvas, 374, "AI DETECTION", confidence if yolo_active else 0.0, yolo_active)
        self._status_card(canvas, 466, "BACKGROUND READY", 1.0 if background_ready else 0.25, background_ready)
        self._mode_button(canvas, "snap", mode == "snap")
        self._mode_button(canvas, "foreground", mode == "foreground")
        self._mode_button(canvas, "cloak", mode == "cloak")

        for name, label_text in (("recapture", "RECAPTURE"), ("mask", "MASK ON" if mask_preview else "MASK")):
            rect = self.button_rects[name]
            self._rounded_rectangle(canvas, rect, self.colors.panel_light, 12)
            self._rounded_rectangle(canvas, rect, self.colors.cyan if name == "mask" and mask_preview else (91, 84, 72), 12, 1)
            self._text(canvas, label_text, (rect[0] + 19, rect[1] + 29), 0.42, self.colors.white)

        self._text(canvas, f"FPS  {fps:4.1f}", (42, 882), 0.48, self.colors.cyan)
        self._text(canvas, f"CONFIDENCE  {confidence:0.2f}", (220, 882), 0.48, self.colors.muted)
        self._text(canvas, "Q / ESC  EXIT", (1055, 882), 0.43, self.colors.muted)
        return canvas
