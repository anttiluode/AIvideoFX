"""Cheap baseline-relative causal-refresh criterion for Layered PhaseRail.

The expensive person image is generated occasionally and transported cheaply by
PhaseRail between keyframes.  This module asks one narrow control question:

    has the transported appearance become *worse than its own normal resting
    mismatch* with the live person, enough to justify buying another diffusion
    keyframe?

That distinction matters for stylised video.  A marble/robot/etc keyframe has a
permanent non-zero mismatch with webcam appearance even when transport is
working perfectly.  Accumulating the absolute residual therefore creates a
positive floor and eventually refreshes forever.

The primary instantaneous residual is still a contrast-normalised blurred
edge/structure mismatch inside the person ownership mask, with smaller terms for
PhaseRail confidence and motion saturation.  The controller now calibrates a
per-keyframe baseline and runs a one-sided, leaky CUSUM-like change detector on
excess above that baseline.

This is a heuristic controller, not a calibrated probability or proof of
optimality.  Its purpose is testable: stay quiet on a stable non-zero style gap,
but fire when the mismatch shifts upward after calibration.
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
    baseline: float
    excess: float
    evidence: float
    threshold: float
    calibrated: bool
    triggered: bool


class CausalRefreshController:
    """Per-keyframe baseline + one-sided leaky change detector.

    Calibration deliberately happens after every fresh generated keyframe.  A
    short warm-up estimates the normal style/transport floor with a median.  The
    detector then accumulates only positive change above::

        baseline * (1 + baseline_margin)

    using::

        evidence = max(0, decay * evidence + excess)

    A constant non-zero mismatch therefore settles near zero evidence instead of
    mathematically guaranteeing a refresh.  While the detector is quiet, the
    baseline follows very slowly so gradual lighting/style drift can be treated
    as nuisance rather than as an event.
    """

    def __init__(
        self,
        *,
        decay: float = 0.94,
        threshold: float = 0.30,
        min_keyframe_age: float = 0.80,
        baseline_alpha: float = 0.02,
        baseline_margin: float = 0.10,
        warmup_samples: int = 12,
    ) -> None:
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.min_keyframe_age = float(min_keyframe_age)
        self.baseline_alpha = float(baseline_alpha)
        self.baseline_margin = float(baseline_margin)
        self.warmup_samples = max(3, int(warmup_samples))
        self.evidence = 0.0
        self.baseline: float | None = None
        self._warmup: list[float] = []
        self.last: RefreshReading | None = None

    def configure(self, *, decay: float, threshold: float, min_keyframe_age: float) -> None:
        self.decay = float(np.clip(decay, 0.0, 0.9995))
        self.threshold = max(1e-6, float(threshold))
        self.min_keyframe_age = max(0.0, float(min_keyframe_age))

    def reset(self) -> None:
        self.evidence = 0.0
        self.baseline = None
        self._warmup = []
        self.last = None

    def _change_update(self, instant: float) -> tuple[float, float, bool]:
        """Return baseline, signed excess, calibrated flag for one sample."""
        if len(self._warmup) < self.warmup_samples:
            self._warmup.append(float(instant))
            self.baseline = float(np.median(self._warmup))
            self.evidence = 0.0
            return float(self.baseline), 0.0, False

        assert self.baseline is not None
        baseline_before = float(self.baseline)
        excess = float(instant - baseline_before * (1.0 + self.baseline_margin))
        self.evidence = max(0.0, self.decay * self.evidence + excess)

        # Adapt the nuisance floor only while the detector is convincingly
        # quiet.  Once a change begins accumulating, freeze the reference so it
        # cannot chase away the event it is supposed to detect.
        if self.evidence < 0.20 * self.threshold:
            a = float(np.clip(self.baseline_alpha, 0.0, 1.0))
            self.baseline = (1.0 - a) * baseline_before + a * float(instant)

        return float(self.baseline), excess, True

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

        baseline, excess, calibrated = self._change_update(instant)
        triggered = bool(
            calibrated
            and float(keyframe_age) >= self.min_keyframe_age
            and self.evidence >= self.threshold
        )
        self.last = RefreshReading(
            structural=structural,
            confidence_penalty=confidence_penalty,
            motion_penalty=motion_penalty,
            instant=instant,
            baseline=baseline,
            excess=excess,
            evidence=float(self.evidence),
            threshold=float(self.threshold),
            calibrated=calibrated,
            triggered=triggered,
        )
        return self.last
