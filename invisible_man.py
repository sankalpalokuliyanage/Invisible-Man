"""Real-time invisibility-cloak effect using OpenCV."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cinematic_ui import CinematicUI
from snap_effect import DustAnimation, SnapDetector


WINDOW_NAME = "Invisible Man"
HUE_TOLERANCE = 10
MIN_SATURATION = 80
MIN_BRIGHTNESS = 55
SCENE_DIFFERENCE = 28


class PersonSegmenter:
    """Optional YOLO person segmenter with a safe OpenCV-only fallback."""

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model = None
        self.error: str | None = None
        if not enabled:
            self.error = "disabled"
            return
        try:
            from ultralytics import YOLO

            print(f"Loading segmentation model: {model_name}")
            self.model = YOLO(model_name)
        except Exception as exc:  # Model/network/runtime failures must not stop the camera.
            self.error = str(exc)
            print(f"YOLO unavailable; using background vision only: {exc}")

    @property
    def active(self) -> bool:
        return self.model is not None

    def person_mask(self, frame: np.ndarray) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], np.uint8)
        if self.model is None:
            return mask
        try:
            results = self.model.predict(
                frame, classes=[0], conf=0.32, imgsz=640, verbose=False
            )
            result = results[0]
            if result.masks is None:
                return mask
            instances = result.masks.data.cpu().numpy()
            if len(instances):
                combined = np.max(instances, axis=0)
                mask = cv2.resize(
                    (combined * 255).astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        except Exception as exc:
            self.error = str(exc)
            print(f"YOLO inference failed; continuing with background vision: {exc}")
            self.model = None
        return mask


@dataclass
class CloakColor:
    """HSV colour selected from the camera feed."""

    hue: int = 0
    saturation: int = 180
    value: int = 180


@dataclass
class AutoColorState:
    """Smoothed automatic colour detection state."""

    color: CloakColor
    confidence: float = 0.0
    locked: bool = False


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
    parser.add_argument(
        "--background-seconds",
        type=float,
        default=3.0,
        help="Seconds used for automatic empty-background capture",
    )
    parser.add_argument(
        "--effect",
        choices=("snap", "foreground", "cloak"),
        default="snap",
        help="Snap-triggered dissolve, immediate foreground, or detected cloak colour",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="Ultralytics segmentation model used for person masks",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Use lightweight background vision without loading a YOLO model",
    )
    return parser.parse_args()


def hue_mask(hsv: np.ndarray, color: CloakColor) -> np.ndarray:
    """Build an HSV mask while correctly handling OpenCV's hue wrap-around."""

    hue_tolerance = HUE_TOLERANCE
    min_saturation = MIN_SATURATION
    min_value = MIN_BRIGHTNESS

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


def keep_large_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Discard tiny disconnected detections while preserving cloth-sized regions."""

    binary = np.where(mask > 40, 255, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    result = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            result[labels == label] = 255
    return result


def stabilise_mask(mask: np.ndarray, history: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Blend consecutive masks to remove edge shimmer and one-frame dropouts."""

    sample = mask.astype(np.float32) / 255.0
    if history is None or history.shape != sample.shape:
        history = sample
    else:
        # Acquire new moving pixels immediately but release old pixels quickly.
        # This removes the trailing/ghosting visible during fast movement.
        rising = sample >= history
        history[rising] = history[rising] * 0.12 + sample[rising] * 0.88
        history[~rising] = history[~rising] * 0.48 + sample[~rising] * 0.52
    stable = np.clip(history * 255.0, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(stable, (5, 5), 0), history


def foreground_mask(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Find pixels that differ from the captured empty scene."""

    difference = cv2.absdiff(frame, background)
    difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    difference = cv2.GaussianBlur(difference, (7, 7), 0)
    threshold = SCENE_DIFFERENCE
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def estimate_cloak_color(
    hsv: np.ndarray, changed: np.ndarray, current: AutoColorState
) -> AutoColorState:
    """Estimate the dominant vivid colour among objects absent from the background."""

    height, width = hsv.shape[:2]
    # Ignore the outer frame, where lighting changes and camera noise are common.
    region = np.zeros((height, width), np.uint8)
    region[int(height * 0.12) : int(height * 0.96), int(width * 0.12) : int(width * 0.88)] = 255
    vivid = cv2.inRange(hsv, (0, 85, 55), (179, 255, 255))
    candidates = cv2.bitwise_and(changed, cv2.bitwise_and(vivid, region))

    # Prefer the largest coherent vivid object. This prevents a small logo,
    # screen, or decoration from becoming the automatically selected cloak.
    component_floor = max(500, int(height * width * 0.004))
    coherent = keep_large_regions(candidates, component_floor)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(coherent, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        candidates = np.where(labels == largest, 255, 0).astype(np.uint8)

    candidate_count = cv2.countNonZero(candidates)
    minimum_pixels = max(800, int(height * width * 0.008))
    if candidate_count < minimum_pixels:
        current.confidence = max(0.0, current.confidence - 0.04)
        return current

    histogram = cv2.calcHist([hsv], [0], candidates, [180], [0, 180]).reshape(-1)
    # Smooth neighbouring hue bins so cloth shadows do not split one colour peak.
    wrapped = np.concatenate((histogram[-5:], histogram, histogram[:5]))
    kernel = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smoothed = np.convolve(wrapped, kernel, mode="same")[5:-5]
    hue = int(np.argmax(smoothed))
    peak_share = float(smoothed[hue] / max(1.0, histogram.sum()))

    hue_distance = abs(hue - current.color.hue)
    hue_distance = min(hue_distance, 180 - hue_distance)
    agrees = hue_distance <= 12 or current.confidence < 0.15
    if agrees:
        current.color.hue = hue
        selected_pixels = hsv[:, :, 0]
        near_hue = np.minimum(
            np.abs(selected_pixels.astype(np.int16) - hue),
            180 - np.abs(selected_pixels.astype(np.int16) - hue),
        ) <= 8
        sample = hsv[(candidates > 0) & near_hue]
        if sample.size:
            current.color.saturation = int(np.median(sample[:, 1]))
            current.color.value = int(np.median(sample[:, 2]))
        current.confidence = min(1.0, current.confidence + 0.08 + peak_share * 0.15)
    else:
        current.confidence = max(0.0, current.confidence - 0.05)

    current.locked = current.confidence >= 0.45
    return current


def capture_background(
    camera: cv2.VideoCapture,
    mirror: bool,
    seconds: float = 2.5,
    ui: CinematicUI | None = None,
) -> np.ndarray | None:
    """Average several empty-scene frames to produce a stable background."""

    started = time.monotonic()
    samples: deque[np.ndarray] = deque(maxlen=30)

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
        if ui is not None:
            capture_progress = np.clip((time.monotonic() - started) / seconds, 0.0, 1.0)
            preview = ui.render(
                preview,
                "snap",
                False,
                False,
                0.0,
                0.0,
                f"STEP OUT - CAPTURING IN {remaining}",
                float(capture_progress),
                False,
            )
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(1)

        if time.monotonic() - started > seconds * 0.45:
            samples.append(frame.astype(np.float32))

    if not samples:
        return None
    # A median ignores brief movement that would otherwise become a ghost in
    # the captured background.
    return np.median(np.stack(tuple(samples), axis=0), axis=0).astype(np.uint8)


def draw_help(
    frame: np.ndarray,
    background_ready: bool,
    auto_color: AutoColorState,
    manual_color: bool,
    fps: float,
    effect: str,
    yolo_active: bool,
) -> None:
    if not background_ready:
        status = "CAPTURING EMPTY BACKGROUND..."
    elif effect == "snap":
        status = "READY - SNAP YOUR FINGERS TO DISAPPEAR"
    elif effect == "foreground" or auto_color.locked or manual_color:
        status = "INVISIBILITY ACTIVE"
    else:
        status = "SHOW A BRIGHT SINGLE-COLOUR CLOTH - DETECTING..."
    color = auto_color.color
    lines = [
        status,
        "S: test snap | F: foreground | C: cloak | R: recapture | M: mask | Q/Esc: quit",
        f"Mode: {effect} | YOLO: {'ON' if yolo_active else 'OFF'} | confidence: {auto_color.confidence:.0%} | {fps:.0f} FPS",
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
    cv2.resizeWindow(WINDOW_NAME, CinematicUI.width, CinematicUI.height)
    ui = CinematicUI()

    auto_color = AutoColorState(CloakColor())
    effect = args.effect
    segmenter = PersonSegmenter(args.model, enabled=not args.no_yolo)
    snap_detector = SnapDetector(Path(__file__).parent / "models" / "hand_landmarker.task")
    latest_hsv: np.ndarray | None = None
    background_seconds = max(1.5, args.background_seconds)
    background: np.ndarray | None = capture_background(
        camera, mirror, seconds=background_seconds, ui=ui
    )
    if background is None:
        print("Automatic background capture failed.")
        camera.release()
        cv2.destroyAllWindows()
        return 1
    show_mask = False
    manual_color = False
    frame_number = 0
    mask_history: np.ndarray | None = None
    fps = 0.0
    previous_tick = time.monotonic()
    person = np.zeros(background.shape[:2], np.uint8)
    animation: DustAnimation | None = None
    snapped_invisible = False
    pending_action: list[str] = []

    def select_color(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = ui.hit_test(x, y)
        if action is not None:
            pending_action.append(action)
            return
        if latest_hsv is None or effect != "cloak":
            return
        # Convert a click in the letterboxed UI camera viewport to source pixels.
        vx, vy, vw, vh = ui.camera_rect
        scale = min(vw / latest_hsv.shape[1], vh / latest_hsv.shape[0])
        shown_w = int(latest_hsv.shape[1] * scale)
        shown_h = int(latest_hsv.shape[0] * scale)
        offset_x = vx + (vw - shown_w) // 2
        offset_y = vy + (vh - shown_h) // 2
        x = int((x - offset_x) / scale)
        y = int((y - offset_y) / scale)
        height, width = latest_hsv.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        radius = 4
        patch = latest_hsv[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ]
        median = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        auto_color.color.hue, auto_color.color.saturation, auto_color.color.value = median.tolist()
        auto_color.confidence = 1.0
        auto_color.locked = True
        nonlocal manual_color
        manual_color = True

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
            changed = foreground_mask(frame, background)
            # Neural inference is intentionally sampled; temporal masks bridge
            # intermediate frames and keep CPU-only machines responsive.
            if segmenter.active and frame_number % 2 == 0:
                person = segmenter.person_mask(frame)
            if not manual_color and effect == "cloak" and frame_number % 3 == 0:
                auto_color = estimate_cloak_color(latest_hsv, changed, auto_color)
            frame_number += 1

            foreground = cv2.bitwise_or(changed, person)
            if effect in ("foreground", "snap"):
                # Background subtraction catches every new object, including
                # unknown classes. YOLO fills the complete person silhouette.
                mask = foreground
                mask = clean_mask(mask)
                minimum_area = max(500, int(frame.shape[0] * frame.shape[1] * 0.002))
                mask = keep_large_regions(mask, minimum_area)
                mask, mask_history = stabilise_mask(mask, mask_history)
            elif auto_color.locked or manual_color:
                colour = hue_mask(latest_hsv, auto_color.color)
                # Scene-difference gating prevents matching objects in the static
                # background from accidentally disappearing too.
                mask = clean_mask(cv2.bitwise_and(colour, changed))
                minimum_area = max(350, int(frame.shape[0] * frame.shape[1] * 0.0015))
                mask = keep_large_regions(mask, minimum_area)
                mask, mask_history = stabilise_mask(mask, mask_history)
            else:
                mask = np.zeros(frame.shape[:2], np.uint8)
                mask_history = None

            if effect == "snap" and not snapped_invisible and animation is None:
                effect_mask = np.zeros_like(mask)
            else:
                effect_mask = mask

            if background is not None:
                if background.shape != frame.shape:
                    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
                alpha = (effect_mask.astype(np.float32) / 255.0)[..., None]
                output = (background * alpha + frame * (1.0 - alpha)).astype(np.uint8)
            else:
                output = frame.copy()

            now = time.monotonic()
            timestamp_ms = int(now * 1000)
            snap_triggered = False
            if (
                effect == "snap"
                and animation is None
                and not snapped_invisible
                and frame_number % 2 == 0
            ):
                snap_triggered = snap_detector.update(frame, timestamp_ms)
            if snap_triggered:
                animation = DustAnimation(frame.copy(), mask.copy(), now)
            if animation is not None:
                output = animation.render(frame, background)
                if animation.complete:
                    animation = None
                    snapped_invisible = True
            instant_fps = 1.0 / max(0.001, now - previous_tick)
            fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1
            previous_tick = now
            camera_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else output
            animation_progress = animation.progress if animation is not None else None
            display = ui.render(
                camera_display,
                effect,
                segmenter.active,
                background is not None,
                auto_color.confidence if effect == "cloak" else (0.92 if segmenter.active else 0.65),
                fps,
                snap_detector.status,
                animation_progress,
                show_mask,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            action = pending_action.pop(0) if pending_action else None
            if key in (ord("q"), 27):
                break
            if key == ord("m") or action == "mask":
                show_mask = not show_mask
            if key == ord("a"):
                auto_color = AutoColorState(CloakColor())
                manual_color = False
                mask_history = None
            if key == ord("f") or action == "foreground":
                effect = "foreground"
                mask_history = None
                animation = None
                snapped_invisible = False
            if key == ord("c") or action == "cloak":
                effect = "cloak"
                auto_color = AutoColorState(CloakColor())
                manual_color = False
                mask_history = None
                animation = None
                snapped_invisible = False
            if action == "snap":
                effect = "snap"
                animation = None
                snapped_invisible = False
                mask_history = None
            if key == ord("s"):
                effect = "snap"
                animation = DustAnimation(frame.copy(), mask.copy(), time.monotonic())
                snapped_invisible = False
            if key in (ord("b"), ord("r")) or action == "recapture":
                new_background = capture_background(
                    camera, mirror, seconds=background_seconds, ui=ui
                )
                if new_background is not None:
                    background = new_background
                    auto_color = AutoColorState(CloakColor())
                    manual_color = False
                    mask_history = None
                    animation = None
                    snapped_invisible = False
                else:
                    print("Background capture failed; keeping the previous background.")
    finally:
        snap_detector.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
