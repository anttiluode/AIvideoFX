import sys
import types
import unittest

import cv2
import numpy as np

from causal_refresh import CausalRefreshController, normalized_structure_mismatch
from fx_core import FXContext, MapStore


def bars(horizontal=False, size=96):
    img = np.zeros((size, size, 3), np.float32)
    if horizontal:
        img[size // 2 - 5:size // 2 + 5, 12:size - 12] = 1.0
    else:
        img[12:size - 12, size // 2 - 5:size // 2 + 5] = 1.0
    return cv2.GaussianBlur(img, (0, 0), 0.8)


class CausalRefreshTests(unittest.TestCase):
    def test_identical_structure_is_dark(self):
        a = bars(False)
        m = np.ones(a.shape[:2], np.float32)
        self.assertLess(normalized_structure_mismatch(a, a.copy(), m), 1e-7)

    def test_global_contrast_is_mostly_nuisance(self):
        a = bars(False)
        b = np.clip(0.35 + 0.45 * a, 0, 1)
        m = np.ones(a.shape[:2], np.float32)
        self.assertLess(normalized_structure_mismatch(a, b, m), 0.03)

    def test_changed_geometry_is_visible(self):
        a = bars(False)
        b = bars(True)
        m = np.ones(a.shape[:2], np.float32)
        self.assertGreater(normalized_structure_mismatch(a, b, m), 0.05)

    def test_persistent_mismatch_eventually_triggers(self):
        a = bars(False)
        b = bars(True)
        m = np.ones(a.shape[:2], np.float32)
        ctrl = CausalRefreshController(decay=0.90, threshold=0.10, min_keyframe_age=0.0)
        reading = None
        for _ in range(40):
            reading = ctrl.update(
                a, b, mask=m, phase_confidence=1.0,
                motion=0.0, max_motion=3.5, keyframe_age=10.0,
            )
            if reading.triggered:
                break
        self.assertIsNotNone(reading)
        self.assertTrue(reading.triggered)

    def test_keyframe_age_gate_blocks_early_refresh(self):
        a = bars(False)
        b = bars(True)
        ctrl = CausalRefreshController(decay=0.0, threshold=1e-6, min_keyframe_age=2.0)
        r = ctrl.update(a, b, keyframe_age=0.5, phase_confidence=0.0, motion=3.5, max_motion=3.5)
        self.assertFalse(r.triggered)
        r = ctrl.update(a, b, keyframe_age=2.5, phase_confidence=0.0, motion=3.5, max_motion=3.5)
        self.assertTrue(r.triggered)

    def test_effect_runs_with_published_person_and_background(self):
        # Exercise the actual CausalPhaseRailLayer plumbing without requiring
        # torch/CUDA in CI. A tiny fake rail has the same public contract.
        class FakeRail:
            def __init__(self, size=128, device="cpu"):
                self.size = size
                self.target = None

            def set_target(self, target):
                self.target = target.copy()

            def process(self, source, **kwargs):
                out = self.target.copy() if self.target is not None else source.copy()
                coherence = np.ones(source.shape[:2], np.float32)
                metrics = {"confidence": 1.0, "motion": 0.0, "coherence": 1.0, "removed": 0.0}
                return out.astype(np.float32), coherence, metrics

        fake = types.ModuleType("fx_phase_rail")
        fake.LayerPhaseRail = FakeRail
        old = sys.modules.get("fx_phase_rail")
        sys.modules["fx_phase_rail"] = fake
        try:
            from fx_causal_refresh import CausalPhaseRailLayer

            h, w = 96, 128
            store = MapStore()
            mask = np.ones((h, w), np.float32)
            person = np.zeros((h, w, 3), np.float32)
            person[..., 1] = 0.9
            background = np.zeros((h, w, 3), np.float32)
            background[..., 0] = 0.8
            store.put("mask", mask)
            store.put("person_style", person)
            store.put("background_style", background)

            effect = CausalPhaseRailLayer({
                "device": "cpu",
                "rail_size": "64",
                "show_refresh": True,
                "auto_refresh": True,
                "refresh_threshold": 10.0,
            })
            frame = np.zeros((h, w, 3), np.float32)
            ctx = FXContext(store, 1.0, 1, (h, w))
            out = effect.apply(frame, ctx)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(out.mean()), 0.01)
        finally:
            if old is None:
                sys.modules.pop("fx_phase_rail", None)
            else:
                sys.modules["fx_phase_rail"] = old


if __name__ == "__main__":
    unittest.main()
