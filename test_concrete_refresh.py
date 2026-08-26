import math
import unittest

import numpy as np

from causal_refresh_runtime import (
    AdaptiveRefreshPolicy,
    RefreshConfig,
    frame_sketch,
    output_probe,
)


class ConcreteRefreshPolicyTests(unittest.TestCase):
    def frame(self, value: float):
        return np.full((90, 160, 3), value, np.float32)

    def test_cold_policy_reuses_small_drift_and_refreshes_large_drift(self):
        cfg = RefreshConfig(
            output_tolerance=0.10,
            bootstrap_input_threshold=0.05,
            min_key_age=0.0,
            max_key_age=100.0,
            decision_hz=100.0,
            audit_rate=0.0,
            min_audit_rate=0.0,
        )
        p = AdaptiveRefreshPolicy(cfg)
        base = self.frame(0.2)
        p.accept_initial(base, now=0.0)
        small = p.decide(self.frame(0.205), has_keyframe=True, pending=False, now=0.1)
        large = p.decide(self.frame(0.30), has_keyframe=True, pending=False, now=0.2)
        self.assertEqual(small.action, "REUSE")
        self.assertEqual(large.action, "REFRESH")

    def test_refresh_observation_learns_input_to_visible_output_gain(self):
        cfg = RefreshConfig(
            output_tolerance=0.10,
            min_observations=1,
            safety_margin=1.0,
            min_key_age=0.0,
            max_key_age=100.0,
            decision_hz=100.0,
            audit_rate=0.0,
            min_audit_rate=0.0,
        )
        p = AdaptiveRefreshPolicy(cfg)
        base = self.frame(0.2)
        changed = self.frame(0.25)
        p.accept_initial(base, now=0.0)
        sketch = frame_sketch(changed, cfg)
        drift = float(np.sqrt(np.mean((sketch - p.anchor_sketch) ** 2)))
        p.observe_refresh(
            request_sketch=sketch,
            request_time=0.2,
            input_drift=drift,
            old_output_probe=output_probe(base, cfg),
            new_output_probe=output_probe(changed, cfg),
            was_audit=False,
        )
        self.assertTrue(math.isfinite(p.learned_gain()))
        self.assertGreater(p.learned_gain(), 0.0)

    def test_audit_can_discover_a_bad_reuse(self):
        cfg = RefreshConfig(
            output_tolerance=0.02,
            bootstrap_input_threshold=1.0,
            min_key_age=0.0,
            max_key_age=100.0,
            decision_hz=100.0,
            audit_rate=1.0,
            min_audit_rate=0.0,
        )
        p = AdaptiveRefreshPolicy(cfg)
        base = self.frame(0.2)
        p.accept_initial(base, now=0.0)
        decision = p.decide(self.frame(0.201), has_keyframe=True, pending=False, now=0.1)
        self.assertEqual(decision.action, "AUDIT")
        p.observe_refresh(
            request_sketch=decision.sketch,
            request_time=0.1,
            input_drift=decision.input_drift,
            old_output_probe=output_probe(base, cfg),
            new_output_probe=output_probe(self.frame(0.5), cfg),
            was_audit=True,
        )
        self.assertEqual(p.stats.unsafe_audits, 1)


class RegistrationTests(unittest.TestCase):
    def test_concrete_effect_and_preset_register(self):
        import fx_concrete_refresh
        import fx_ai
        from fx_core import EFFECTS_BY_NAME, PRESETS

        self.assertIn("ConcreteDream", EFFECTS_BY_NAME)
        self.assertIn("Concrete Dream", PRESETS)
        fx = EFFECTS_BY_NAME["ConcreteDream"]()
        self.assertIn("style_concrete", fx.requires())
        self.assertTrue(getattr(fx_ai.DiffusionWorker._due, "_concrete_patch", False))


if __name__ == "__main__":
    unittest.main()
