"""Cheap causal-refresh criterion for the Layered PhaseRail video path.

The expensive person image is generated occasionally and transported cheaply by
PhaseRail between keyframes.  This module answers one narrow control question:

    has the transported appearance become a poor enough explanation of the
    current live structure that buying another diffusion keyframe is justified?

It deliberately does *not* compare raw RGB. A prompted marble/robot/etc person
should not be penalised merely for having different colour from the webcam.
The primary residual is a contrast-normalised blurred edge/structure map inside
the current ownership mask. PhaseRail confidence and motion saturation are
small secondary terms.

This is a heuristic controller. It is testable and tuneable; it is not a proof
of optimality or a calibrated probability of failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-7


def _gray(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    if x.ndim == 2:
        return x
    if x.shape[-1] == 1:
        return x[..., 0]
    # Inputs in AIvideoFX are BGR float images.
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def _edge_signature(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Low-cost style-tolerant structure signature in [0,1]."""
    g = _gray(image)
    g = cv2.GaussianBlur(g, (0, 0), 1.25)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    if mask is None:
        values = mag.ravel()
    else:
        m = np.asarray(mask, np.float32)
        values = mag[m > 0.08]
        if values.size < 16:
            values = mag.ravel()
    scale = float(np.percentile(values, 90)) if values.size else 0.0
    if scale < EPS:
        return np.zeros_like(mag, np.float32)
    return np.clip(mag / scale, 0.0, 1.0).astype(np.float32)


def normalized_structure_mismatch(
    live: np.ndarray,
    carried: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute difference between separately contrast-normalised edges."""
    if live.shape[:2] != carried.shape[:2]:
        carried = cv2.resize(carried, (live.shape[1], live.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask is not None and mask.shape[:2] != live.shape[:2]:
        mask = cv2.resize(mask, (live.shape[1], live.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = _edge_signature(live, mask)
    b = _edge_signature(carried, mask)
    diff = np.abs(a - b).astype(np.float32)
    if mask is None:
        return float(np.mean(diff))
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    denom = float(np.sum(m))
    if denom < EPS:
        return float(np.mean(diff))
    return float(np.sum(diff * m) / denom)


@dataclass
class RefreshReading:
    structural: float
    confidence_penalty: float
    motion_penalty: float
    instant: float
    evidence: float
    threshold: float
    triggered: bool


class CausalRefreshController:
    """Leaky accumulated mismatch with an explicit keyframe-age gate."""

    def __init__(
        self,
        *,
        decay: float = 0.94,
        threshold: float = 0.80,
        min_keyframe_age: float = 0.80,
    ) -> None:
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.min_keyframe_age = float(min_keyframe_age)
        self.evidence = 0.0
        self.last: RefreshReading | None = None

    def configure(self, *, decay: float, threshold: float, min_keyframe_age: float) -> None:
        self.decay = float(np.clip(decay, 0.0, 0.9995))
        self.threshold = max(1e-6, float(threshold))
        self.min_keyframe_age = max(0.0, float(min_keyframe_age))

    def reset(self) -> None:
        self.evidence = 0.0
        self.last = None

    def update(
        self,
        live: np.ndarray,
        carried: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        phase_confidence: float = 1.0,
        motion: float = 0.0,
        max_motion: float = 3.5,
        keyframe_age: float = 0.0,
    ) -> RefreshReading:
        structural = normalized_structure_mismatch(live, carried, mask)
        confidence_penalty = float(np.clip(1.0 - float(phase_confidence), 0.0, 1.0))
        ratio = float(motion) / max(0.25, float(max_motion))
        motion_penalty = float(np.clip((ratio - 0.60) / 0.40, 0.0, 1.0))

        # Structure dominates. Confidence/motion only help identify cases where
        # the transport solver itself is telling us not to trust its update.
        instant = float(np.clip(
            0.72 * structural + 0.20 * confidence_penalty + 0.08 * motion_penalty,
            0.0, 1.0,
        ))

        # A leaky sum, not a monotone scientific D_T^2. Quiet frames forget old
        # trouble; persistent mismatch eventually crosses the spend threshold.
        self.evidence = self.decay * self.evidence + instant * instant
        triggered = bool(
            float(keyframe_age) >= self.min_keyframe_age
            and self.evidence >= self.threshold
        )
        self.last = RefreshReading(
            structural=structural,
            confidence_penalty=confidence_penalty,
            motion_penalty=motion_penalty,
            instant=instant,
            evidence=float(self.evidence),
            threshold=float(self.threshold),
            triggered=triggered,
        )
        return self.last
