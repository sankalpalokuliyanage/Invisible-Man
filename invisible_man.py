"""Real-time invisibility-cloak effect using OpenCV."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


WINDOW_NAME = "Invisible Man"


@dataclass
class CloakColor:
    """HSV colour selected from the camera feed."""

    hue: int = 0
    saturation: int = 180
    value: int = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an invisibility-cloak effect with a webcam and OpenCV."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height")
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the camera image"
    )
    return parser.parse_args()


def hue_mask(hsv: np.ndarray, color: CloakColor) -> np.ndarray:
    """Build an HSV mask while correctly handling OpenCV's hue wrap-around."""

    hue_tolerance = cv2.getTrackbarPos("Hue tolerance", WINDOW_NAME)
    min_saturation = cv2.getTrackbarPos("Min saturation", WINDOW_NAME)
    min_value = cv2.getTrackbarPos("Min brightness", WINDOW_NAME)

    low_hue = color.hue - hue_tolerance
    high_hue = color.hue + hue_tolerance
    lower_sv = (min_saturation, min_value)

    if low_hue < 0:
        first = cv2.inRange(hsv, (0, *lower_sv), (high_hue, 255, 255))
        second = cv2.inRange(hsv, (180 + low_hue, *lower_sv), (179, 255, 255))
        return cv2.bitwise_or(first, second)

    if high_hue > 179:
        first = cv2.inRange(hsv, (low_hue, *lower_sv), (179, 255, 255))
        second = cv2.inRange(hsv, (0, *lower_sv), (high_hue - 180, 255, 255))
        return cv2.bitwise_or(first, second)

    return cv2.inRange(hsv, (low_hue, *lower_sv), (high_hue, 255, 255))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove speckles, fill small gaps, and soften mask edges."""

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.GaussianBlur(mask, (7, 7), 0)


def capture_background(
    camera: cv2.VideoCapture, mirror: bool, seconds: float = 2.5
) -> np.ndarray | None:
    """Average several empty-scene frames to produce a stable background."""

    started = time.monotonic()
    samples: list[np.ndarray] = []

    while time.monotonic() - started < seconds:
        ok, frame = camera.read()
        if not ok:
            return None
        if mirror:
            frame = cv2.flip(frame, 1)

        remaining = max(0, int(np.ceil(seconds - (time.monotonic() - started))))
        preview = frame.copy()
        cv2.putText(
            preview,
            f"Step out of view - capturing in {remaining}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(1)

        if time.monotonic() - started > seconds * 0.45:
            samples.append(frame.astype(np.float32))

    if not samples:
        return None
    return np.mean(samples[-30:], axis=0).astype(np.uint8)


def draw_help(frame: np.ndarray, background_ready: bool, color: CloakColor) -> None:
    status = "READY" if background_ready else "PRESS B TO CAPTURE BACKGROUND"
    lines = [
        status,
        "Click cloak to select colour | B: background | M: mask | Q/Esc: quit",
        f"Selected HSV: {color.hue}, {color.saturation}, {color.value}",
    ]
    for index, line in enumerate(lines):
        y = 32 + index * 28
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)


def main() -> int:
    args = parse_args()
    mirror = not args.no_mirror
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1 (or another index).")
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Hue tolerance", WINDOW_NAME, 10, 45, lambda _value: None)
    cv2.createTrackbar("Min saturation", WINDOW_NAME, 80, 255, lambda _value: None)
    cv2.createTrackbar("Min brightness", WINDOW_NAME, 55, 255, lambda _value: None)

    selected = CloakColor()
    latest_hsv: np.ndarray | None = None
    background: np.ndarray | None = None
    show_mask = False

    def select_color(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or latest_hsv is None:
            return
        height, width = latest_hsv.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        radius = 4
        patch = latest_hsv[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ]
        median = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        selected.hue, selected.saturation, selected.value = median.tolist()

    cv2.setMouseCallback(WINDOW_NAME, select_color)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame could not be read.")
                return 1
            if mirror:
                frame = cv2.flip(frame, 1)

            latest_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = clean_mask(hue_mask(latest_hsv, selected))

            if background is not None:
                if background.shape != frame.shape:
                    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
                alpha = (mask.astype(np.float32) / 255.0)[..., None]
                output = (background * alpha + frame * (1.0 - alpha)).astype(np.uint8)
            else:
                output = frame.copy()

            draw_help(output, background is not None, selected)
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else output
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("b"):
                background = capture_background(camera, mirror)
                if background is None:
                    print("Background capture failed.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
