import unittest

import cv2
import numpy as np

from causal_refresh import CausalRefreshController, normalized_structure_mismatch


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


if __name__ == "__main__":
    unittest.main()
