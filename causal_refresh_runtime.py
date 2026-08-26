"""Adaptive refresh scheduling for expensive generative video keyframes.

This is the Oppository/Concrete idea specialized for asynchronous image generation:

* the cheap certificate is keyframe-relative low-resolution luminance drift;
* an expensive refresh is requested only when predicted visible keyframe error
  crosses a tolerance (or a bounded age/audit rule says to check);
* audits compare the carried output that would have remained visible against the
  newly generated image, turning mistakes into new sensitivity evidence.

The controller knows nothing about Tk, diffusers, or PhaseRail.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque

import cv2
import numpy as np


@dataclass
class RefreshConfig:
    output_tolerance: float = 0.12
    bootstrap_input_threshold: float = 0.045
    min_observations: int = 4
    history: int = 64
    quantile: float = 0.90
    safety_margin: float = 1.50
    decision_hz: float = 4.0
    audit_rate: float = 0.05
    min_audit_rate: float = 0.01
    failure_boost: float = 0.25
    min_key_age: float = 0.60
    max_key_age: float = 8.0
    sketch_width: int = 48
    sketch_height: int = 27
    probe_size: int = 64
    seed: int = 0
    eps: float = 1e-6


@dataclass
class RefreshStats:
    frames: int = 0
    decisions: int = 0
    refresh_requests: int = 0
    reuses: int = 0
    audits: int = 0
    unsafe_audits: int = 0
    observations: int = 0
    last_action: str = "COLD"
    last_reason: str = "no keyframe"
    last_input_drift: float = math.inf
    last_predicted_error: float = math.inf
    last_observed_error: float = math.inf
    learned_gain: float = math.inf
    key_age: float = 0.0

    @property
    def refresh_fraction(self) -> float:
        return self.refresh_requests / self.decisions if self.decisions else 0.0

    @property
    def reuse_fraction(self) -> float:
        return self.reuses / self.decisions if self.decisions else 0.0

    @property
    def unsafe_rate(self) -> float:
        return self.unsafe_audits / self.audits if self.audits else 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(
            refresh_fraction=self.refresh_fraction,
            reuse_fraction=self.reuse_fraction,
            unsafe_rate=self.unsafe_rate,
        )
        return out


@dataclass
class RefreshDecision:
    action: str
    reason: str
    sketch: np.ndarray
    input_drift: float
    predicted_error: float
    key_age: float

    @property
    def requests_refresh(self) -> bool:
        return self.action in {"REFRESH", "AUDIT"}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    xs = sorted(values)
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def frame_sketch(frame_bgr: np.ndarray, cfg: RefreshConfig) -> np.ndarray:
    """Cheap CPU certificate for uint8 or float BGR frames."""
    x = np.asarray(frame_bgr)
    if x.ndim == 2:
        gray = x.astype(np.float32, copy=False)
    else:
        xf = x.astype(np.float32, copy=False)
        gray = (
            np.float32(0.0722) * xf[..., 0]
            + np.float32(0.7152) * xf[..., 1]
            + np.float32(0.2126) * xf[..., 2]
        )
    tiny = cv2.resize(
        gray,
        (int(cfg.sketch_width), int(cfg.sketch_height)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    if tiny.size and float(np.nanmax(tiny)) > 1.5:
        tiny *= np.float32(1.0 / 255.0)
    return np.clip(tiny, 0.0, 1.0)


def sketch_drift(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return math.inf
    d = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(d * d)))


def output_probe(image_bgr: np.ndarray, cfg: RefreshConfig) -> np.ndarray:
    """Small probe of the image a viewer would actually see."""
    x = np.asarray(image_bgr, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    small = cv2.resize(
        x,
        (int(cfg.probe_size), int(cfg.probe_size)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    if small.size and float(np.nanmax(small)) > 1.5:
        small *= np.float32(1.0 / 255.0)
    return np.clip(small, 0.0, 1.0)


def probe_error(a: np.ndarray, b: np.ndarray) -> float:
    """RMS visible-image change on [0,1] probes."""
    if a.shape != b.shape:
        return math.inf
    d = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(d * d)))


class AdaptiveRefreshPolicy:
    """Learn how much expensive generated output changes for live-input drift."""

    def __init__(self, config: RefreshConfig | None = None) -> None:
        self.config = config or RefreshConfig()
        self.stats = RefreshStats()
        self._rng = random.Random(self.config.seed)
        self._gains: Deque[float] = deque(maxlen=self.config.history)
        self.anchor_sketch: np.ndarray | None = None
        self.anchor_time: float | None = None
        self.last_decision_time: float = -math.inf

    def reset(self, keep_learning: bool = True) -> None:
        self.anchor_sketch = None
        self.anchor_time = None
        self.last_decision_time = -math.inf
        self.stats = RefreshStats()
        if not keep_learning:
            self._gains.clear()

    def configure(self, config: RefreshConfig) -> None:
        old_seed = self.config.seed
        self.config = config
        self._gains = deque(self._gains, maxlen=self.config.history)
        if old_seed != self.config.seed:
            self._rng = random.Random(self.config.seed)

    def learned_gain(self) -> float:
        if len(self._gains) < max(1, int(self.config.min_observations)):
            return math.inf
        base = _quantile(list(self._gains), self.config.quantile)
        return max(0.0, base * self.config.safety_margin)

    def audit_probability(self) -> float:
        base = max(self.config.min_audit_rate, self.config.audit_rate)
        return min(1.0, max(0.0, base + self.config.failure_boost * self.stats.unsafe_rate))

    def accept_initial(self, frame_bgr: np.ndarray, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.anchor_sketch = frame_sketch(frame_bgr, self.config)
        self.anchor_time = now
        self.last_decision_time = now
        self.stats.last_action = "KEYFRAME"
        self.stats.last_reason = "initial generated keyframe"
        self.stats.key_age = 0.0

    def _predict(self, drift: float) -> float:
        gain = self.learned_gain()
        self.stats.learned_gain = gain
        if math.isfinite(gain):
            return drift * gain
        threshold = max(self.config.bootstrap_input_threshold, self.config.eps)
        return self.config.output_tolerance * (drift / threshold)

    def decide(
        self,
        frame_bgr: np.ndarray,
        *,
        has_keyframe: bool,
        pending: bool,
        now: float | None = None,
    ) -> RefreshDecision:
        now = time.monotonic() if now is None else float(now)
        self.stats.frames += 1
        sketch = frame_sketch(frame_bgr, self.config)

        if not has_keyframe or self.anchor_sketch is None or self.anchor_time is None:
            self.stats.last_action = "WAIT"
            self.stats.last_reason = "waiting for first generated keyframe"
            return RefreshDecision("WAIT", self.stats.last_reason, sketch, math.inf, math.inf, 0.0)

        drift = sketch_drift(sketch, self.anchor_sketch)
        age = max(0.0, now - self.anchor_time)
        predicted = self._predict(drift)
        self.stats.last_input_drift = drift
        self.stats.last_predicted_error = predicted
        self.stats.key_age = age

        if pending:
            self.stats.last_action = "REUSE"
            self.stats.last_reason = "refresh already generating"
            return RefreshDecision("REUSE", self.stats.last_reason, sketch, drift, predicted, age)

        period = 1.0 / max(0.1, float(self.config.decision_hz))
        if now - self.last_decision_time < period:
            self.stats.last_action = "REUSE"
            self.stats.last_reason = "between decision ticks"
            return RefreshDecision("REUSE", self.stats.last_reason, sketch, drift, predicted, age)

        self.last_decision_time = now
        self.stats.decisions += 1

        if self.config.max_key_age > 0 and age >= self.config.max_key_age:
            self.stats.refresh_requests += 1
            self.stats.last_action = "REFRESH"
            self.stats.last_reason = "maximum keyframe age"
            return RefreshDecision("REFRESH", self.stats.last_reason, sketch, drift, predicted, age)

        if age >= self.config.min_key_age and (
            not math.isfinite(predicted) or predicted > self.config.output_tolerance
        ):
            self.stats.refresh_requests += 1
            self.stats.last_action = "REFRESH"
            self.stats.last_reason = "predicted visible change"
            return RefreshDecision("REFRESH", self.stats.last_reason, sketch, drift, predicted, age)

        if age >= self.config.min_key_age and self._rng.random() < self.audit_probability():
            self.stats.refresh_requests += 1
            self.stats.audits += 1
            self.stats.last_action = "AUDIT"
            self.stats.last_reason = "blind-spot audit"
            return RefreshDecision("AUDIT", self.stats.last_reason, sketch, drift, predicted, age)

        self.stats.reuses += 1
        self.stats.last_action = "REUSE"
        self.stats.last_reason = "predicted change below tolerance"
        return RefreshDecision("REUSE", self.stats.last_reason, sketch, drift, predicted, age)

    def observe_refresh(
        self,
        *,
        request_sketch: np.ndarray,
        request_time: float,
        input_drift: float,
        old_output_probe: np.ndarray | None,
        new_output_probe: np.ndarray | None,
        was_audit: bool,
    ) -> float:
        observed = math.inf
        if old_output_probe is not None and new_output_probe is not None:
            observed = probe_error(old_output_probe, new_output_probe)

        self.stats.last_observed_error = observed
        if math.isfinite(input_drift) and math.isfinite(observed):
            denom = max(float(input_drift), self.config.eps)
            gain = observed / denom
            if math.isfinite(gain):
                self._gains.append(float(gain))
                self.stats.observations += 1

        if was_audit and math.isfinite(observed) and observed > self.config.output_tolerance:
            self.stats.unsafe_audits += 1

        self.anchor_sketch = request_sketch.copy()
        self.anchor_time = float(request_time)
        self.stats.learned_gain = self.learned_gain()
        self.stats.key_age = 0.0
        return observed

    def summary(self) -> dict:
        out = self.stats.to_dict()
        out["learned_observations"] = len(self._gains)
        out["audit_probability"] = self.audit_probability()
        return out
