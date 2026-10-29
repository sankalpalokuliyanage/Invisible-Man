"""Hand-snap recognition and a procedural cinematic dust dissolve effect."""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class SnapDetector:
    """Recognise thumb/middle-finger contact followed by a fast release."""

    def __init__(self, model_path: Path) -> None:
        self.landmarker = None
        self.error: str | None = None
        self.was_primed = False
        self.primed_at = 0.0
        self.previous_middle: np.ndarray | None = None
        self.previous_time = 0.0
        self.cooldown_until = 0.0
        self.status = "loading hand tracker"
        try:
            import mediapipe as mp

            model_path.parent.mkdir(parents=True, exist_ok=True)
            if not model_path.exists():
                print("Downloading MediaPipe hand landmark model...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
            self.mp = mp
            self.status = "show hand and snap"
        except Exception as exc:
            self.error = str(exc)
            self.status = "hand tracker unavailable - press S"
            print(f"Hand snap tracking unavailable: {exc}")

    def update(self, bgr: np.ndarray, timestamp_ms: int) -> bool:
        if self.landmarker is None:
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        now = time.monotonic()
        if not result.hand_landmarks:
            self.status = "show hand and snap"
            self.was_primed = False
            self.previous_middle = None
            return False

        points = result.hand_landmarks[0]
        thumb = np.array((points[4].x, points[4].y), np.float32)
        middle = np.array((points[12].x, points[12].y), np.float32)
        wrist = np.array((points[0].x, points[0].y), np.float32)
        middle_base = np.array((points[9].x, points[9].y), np.float32)
        palm = max(0.025, float(np.linalg.norm(wrist - middle_base)))
        gap = float(np.linalg.norm(thumb - middle)) / palm

        speed = 0.0
        if self.previous_middle is not None and self.previous_time:
            elapsed = max(0.001, now - self.previous_time)
            speed = float(np.linalg.norm(middle - self.previous_middle)) / palm / elapsed
        self.previous_middle = middle
        self.previous_time = now

        if gap < 0.38 and now >= self.cooldown_until:
            self.was_primed = True
            self.primed_at = now
            self.status = "snap primed - release middle finger"
            return False

        recently_primed = self.was_primed and now - self.primed_at < 0.55
        triggered = recently_primed and gap > 0.62 and speed > 1.15
        if triggered:
            self.was_primed = False
            self.cooldown_until = now + 4.0
            self.status = "SNAP DETECTED"
            return True

        if self.was_primed and now - self.primed_at >= 0.55:
            self.was_primed = False
        self.status = "tracking hand"
        return False

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()


@dataclass
class DustAnimation:
    source: np.ndarray
    subject_mask: np.ndarray
    started_at: float
    duration: float = 3.2

    def __post_init__(self) -> None:
        height, width = self.subject_mask.shape
        rng = np.random.default_rng(8675309)
        noise = rng.random((max(2, height // 5), max(2, width // 5)), dtype=np.float32)
        self.noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        ys, xs = np.where(self.subject_mask > 100)
        if len(xs):
            sample_count = min(2400, max(300, len(xs) // 45))
            selected = rng.choice(len(xs), sample_count, replace=len(xs) < sample_count)
            self.px = xs[selected].astype(np.float32)
            self.py = ys[selected].astype(np.float32)
            self.colors = self.source[ys[selected], xs[selected]].copy()
            self.vx = rng.uniform(25, 135, sample_count).astype(np.float32)
            self.vy = rng.uniform(-55, 28, sample_count).astype(np.float32)
            self.delay = rng.uniform(0.0, 0.72, sample_count).astype(np.float32)
            self.size = rng.integers(1, 4, sample_count)
        else:
            self.px = self.py = self.vx = self.vy = self.delay = np.array([])
            self.colors = np.empty((0, 3), np.uint8)
            self.size = np.array([], np.int32)

    @property
    def progress(self) -> float:
        return float(np.clip((time.monotonic() - self.started_at) / self.duration, 0.0, 1.0))

    @property
    def complete(self) -> bool:
        return self.progress >= 1.0

    def render(self, live: np.ndarray, background: np.ndarray) -> np.ndarray:
        progress = self.progress
        # A left-to-right noisy erosion creates a more organic disintegration.
        width = self.subject_mask.shape[1]
        horizontal = np.linspace(0.0, 0.32, width, dtype=np.float32)[None, :]
        threshold = np.clip(progress * 1.32 - horizontal, 0.0, 1.0)
        vanished = (self.noise < threshold) & (self.subject_mask > 0)
        alpha = cv2.GaussianBlur(vanished.astype(np.float32), (5, 5), 0)[..., None]
        output = (background * alpha + live * (1.0 - alpha)).astype(np.uint8)

        height, frame_width = output.shape[:2]
        for index in range(len(self.px)):
            local = progress - float(self.delay[index])
            if not 0.0 < local < 0.52:
                continue
            x = int(self.px[index] + self.vx[index] * local)
            y = int(self.py[index] + self.vy[index] * local + 45 * local * local)
            if 0 <= x < frame_width and 0 <= y < height:
                fade = 1.0 - local / 0.52
                colour = tuple(int(channel * fade) for channel in self.colors[index])
                cv2.circle(output, (x, y), int(self.size[index]), colour, -1, cv2.LINE_AA)
        return output
