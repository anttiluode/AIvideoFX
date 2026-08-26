"""Adaptive full-frame diffusion refresh mode for AI Video FX.

Concrete Dream uses the ordinary full-frame ``style`` diffusion channel from the
Diffusion tab.  It turns that channel into an expensive keyframe service:
cheap live-frame certificates decide when a new diffusion image is worth buying,
while PhaseRail (or a boring hold) carries the last generated keyframe between
refreshes.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

import fx_ai
from causal_refresh_runtime import AdaptiveRefreshPolicy, RefreshConfig, output_probe
from fx_core import (
    EFFECT_CLASSES, EFFECTS_BY_NAME, PRESETS, AIDream, AnttisDeepfakeLayer,
    Bloom, ColorGrade, Param, blur, luma,
)


# While this effect's request token is present, the existing style worker becomes
# a frozen keyframe service. Dropping "style" explicitly wakes diffusion again.
_STOCK_DUE = fx_ai.DiffusionWorker._due


def _due(self, key, channel):
    if key == "style" and "style_concrete" in self.needs:
        sig = self._signature(key, channel)
        return self.store.get(key) is None or self._done_signature.get(key) != sig
    return _STOCK_DUE(self, key, channel)


if not getattr(fx_ai.DiffusionWorker._due, "_concrete_patch", False):
    _due._concrete_patch = True
    fx_ai.DiffusionWorker._due = _due


class ConcreteDream(AIDream):
    name = "Concrete Dream (adaptive)"
    group = "diffusion"
    needs = {"style", "style_concrete"}
    blurb = (
        "Adaptive FULL-FRAME diffusion using the model/prompt in the Diffusion tab "
        "(not the Layers person/background prompts). Diffusion makes visible "
        "keyframes only when a cheap certificate says the old generated result may "
        "be stale. PhaseRail can carry that generated keyframe between refreshes. "
        "Use View=keyframe to inspect the raw expensive AI image with no transport."
    )
    params = AIDream.params + [
        Param("view", "View", "choice", "output", choices=("output", "keyframe", "split")),
        Param("transport", "Between refreshes", "choice", "phase", choices=("phase", "hold")),
        Param("rail_size", "Rail resolution", "choice", "128",
              choices=("64", "96", "128", "160", "192")),
        Param("device", "Rail device", "choice", "cuda", choices=("cuda", "cpu")),
        Param("phase_lock", "Phase lock", "float", 0.92, 0.0, 1.0),
        Param("rail_structure", "Live structure", "float", 0.25, 0.0, 1.0),
        Param("rail_detail", "Generated detail", "float", 0.95, 0.0, 1.0),
        Param("max_motion", "Motion radius", "float", 3.5, 0.5, 8.0),
        Param("output_tolerance", "Refresh tolerance", "float", 0.12, 0.01, 0.50),
        Param("bootstrap_input", "Cold input threshold", "float", 0.045, 0.005, 0.20),
        Param("decision_hz", "Decision rate", "float", 4.0, 0.5, 12.0),
        Param("audit_rate", "Audit chance", "float", 0.05, 0.0, 0.50),
        Param("min_key_age", "Min key age (s)", "float", 0.60, 0.0, 5.0),
        Param("max_key_age", "Max key age (s)", "float", 8.0, 0.5, 30.0),
        Param("show_refresh", "Show refresh HUD", "bool", True),
    ]

    def _cfg(self):
        audit = float(self.p("audit_rate"))
        return RefreshConfig(
            output_tolerance=float(self.p("output_tolerance")),
            bootstrap_input_threshold=float(self.p("bootstrap_input")),
            decision_hz=float(self.p("decision_hz")),
            audit_rate=audit,
            min_audit_rate=0.0,
            failure_boost=0.25 if audit > 0 else 0.0,
            min_key_age=float(self.p("min_key_age")),
            max_key_age=float(self.p("max_key_age")),
        )

    def _policy(self, st):
        p = st.get("policy")
        if p is None:
            p = AdaptiveRefreshPolicy(self._cfg())
            st["policy"] = p
        else:
            p.configure(self._cfg())
        return p

    def _rail(self, st):
        if self.p("transport") != "phase":
            return None
        size, device = int(self.p("rail_size")), str(self.p("device"))
        rail = st.get("rail")
        if rail is None or st.get("rail_size") != size or st.get("rail_device") != device:
            from fx_phase_rail import LayerPhaseRail
            rail = LayerPhaseRail(size=size, device=device)
            st.update(rail=rail, rail_size=size, rail_device=device, rail_stamp=-1.0)
        return rail

    def _carry(self, img, key, st):
        rail = self._rail(st)
        if rail is None:
            return key
        size = int(self.p("rail_size"))
        source, meta = AnttisDeepfakeLayer._letterbox(img, size)
        stamp = float(st.get("style_stamp", -1.0))
        if st.get("rail_stamp") != stamp:
            target, _ = AnttisDeepfakeLayer._letterbox(key, size)
            rail.set_target(target)
            st["rail_stamp"] = stamp
        out, _, metrics = rail.process(
            source,
            phase_lock=float(self.p("phase_lock")),
            style_strength=1.0,
            nullspace_strength=0.0,
            structure_follow=float(self.p("rail_structure")),
            detail_follow=float(self.p("rail_detail")),
            max_displacement=float(self.p("max_motion")),
        )
        st["rail_metrics"] = metrics
        return AnttisDeepfakeLayer._unletterbox(out, meta)

    @staticmethod
    def _hud(out, p, pending, view):
        s = p.stats
        hud = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        h, _ = hud.shape[:2]
        pred = s.last_predicted_error
        pt = "inf" if not math.isfinite(pred) else f"{pred:.3f}"
        obs = s.last_observed_error
        ot = "-" if not math.isfinite(obs) else f"{obs:.3f}"
        action = "WAIT" if pending else s.last_action
        text = (
            f"Concrete {action} [{view}] drift {s.last_input_drift:.4f} pred {pt}/"
            f"{p.config.output_tolerance:.3f} obs {ot} | refresh {s.refresh_requests} "
            f"reuse {s.reuses} audit {s.audits} miss {s.unsafe_audits}"
        )
        y = max(22, h - 14)
        cv2.putText(hud, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(hud, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return hud.astype(np.float32) / 255.0

    def apply(self, img, ctx):
        st = ctx.st(self)
        p = self._policy(st)
        now = time.monotonic()
        style = ctx.style()
        stamp = ctx.stamp("style") if style is not None else 0.0
        pending = bool(st.get("pending", False))
        held = st.get("held")
        old_stamp = float(st.get("style_stamp", -1.0))

        # New expensive answer arrived. Latch it immediately as the visible
        # keyframe before any later refresh request can drop the worker slot.
        if style is not None and stamp != old_stamp:
            if pending and st.get("request_sketch") is not None:
                p.observe_refresh(
                    request_sketch=st["request_sketch"],
                    request_time=float(st["request_time"]),
                    input_drift=float(st["request_drift"]),
                    old_output_probe=st.get("request_probe"),
                    new_output_probe=output_probe(style, p.config),
                    was_audit=bool(st.get("request_audit", False)),
                )
            else:
                p.accept_initial(img, now=now)
            st.update(
                held=style.copy(), style_stamp=stamp, pending=False,
                request_sketch=None, request_probe=None,
            )
            held, pending = st["held"], False
            rail = self._rail(st)
            if rail is not None:
                target, _ = AnttisDeepfakeLayer._letterbox(style, int(self.p("rail_size")))
                rail.set_target(target)
                st["rail_stamp"] = stamp

        # Never rely on the worker slot as persistent visual memory: refresh
        # requests intentionally drop it. The held copy is the durable keyframe.
        key = held if held is not None else style
        if key is None:
            return img

        try:
            carried = self._carry(img, key, st)
        except Exception as exc:
            st["rail_error"] = str(exc)
            carried = key

        decision = p.decide(img, has_keyframe=True, pending=pending, now=now)
        if decision.requests_refresh and not pending:
            st.update(
                held=key.copy(), pending=True,
                request_sketch=decision.sketch.copy(),
                request_time=now,
                request_drift=float(decision.input_drift),
                request_probe=output_probe(carried, p.config),
                request_audit=decision.action == "AUDIT",
            )
            pending = True
            ctx.store.drop("style")

        # Diagnostic views make the expensive image impossible to hide.
        view = str(self.p("view"))
        if view == "keyframe":
            out = np.clip(key, 0.0, 1.0)
        else:
            s = carried
            prev = st.get("display_prev")
            smear = float(self.p("smear"))
            if prev is not None and prev.shape == s.shape and smear > 0:
                s = prev * smear + s * (1.0 - smear)
            st["display_prev"] = s
            if self.p("keep_luma"):
                s = s * ((luma(img) + 0.05) / (luma(s) + 0.05))[..., None]
            mix = float(self.p("mix"))
            out = img * (1.0 - mix) + s * mix
            detail = float(self.p("detail"))
            if detail > 0:
                out += (img - blur(img, 3)) * (detail * 2.0)
            out = np.clip(out, 0.0, 1.0)
            if view == "split":
                half = out.shape[1] // 2
                out[:, :half] = np.clip(key[:, :half], 0.0, 1.0)
                cv2.line(out, (half, 0), (half, out.shape[0]), (1.0, 1.0, 1.0), 1)

        return self._hud(out, p, pending, view) if self.p("show_refresh") else out


def register():
    if ConcreteDream not in EFFECT_CLASSES:
        try:
            i = EFFECT_CLASSES.index(AIDream) + 1
        except ValueError:
            i = len(EFFECT_CLASSES)
        EFFECT_CLASSES.insert(i, ConcreteDream)
    EFFECTS_BY_NAME[ConcreteDream.__name__] = ConcreteDream
    PRESETS["Concrete Dream"] = [
        {"type": "ConcreteDream", "values": {
            # First-run defaults intentionally make the generated image obvious.
            "mix": 1.0, "detail": 0.0, "keep_luma": False, "smear": 0.0,
            "view": "output", "transport": "phase",
            "rail_size": "128", "device": "cuda",
            "phase_lock": 0.92, "rail_structure": 0.25, "rail_detail": 0.95,
            "output_tolerance": 0.12, "bootstrap_input": 0.045,
            "decision_hz": 4.0, "audit_rate": 0.05,
            "min_key_age": 0.60, "max_key_age": 8.0, "show_refresh": True,
        }},
        {"type": "Bloom", "values": {"threshold": 0.74, "intensity": 0.28}},
        {"type": "ColorGrade", "values": {"contrast": 1.04, "saturation": 1.05}},
    ]


register()
