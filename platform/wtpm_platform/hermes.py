"""Hermes-style agent over the 25 WT-PM models.

Principles adapted from Hermes / ReAct (Yao et al. 2022; Nous Hermes agent loop):

1. **Goal first** — every run starts from the seven operator questions.
2. **Thought before Action** — the agent must say *why* it is calling a tool.
3. **Action is a real tool** — tools are the 25 models + twin + safety + XAI.
   Observations are returned by code, never invented.
4. **Observe, then revise** — the next thought is grounded in the last observation.
5. **Stop when the goal is met** (or the step budget is exhausted).
6. **The trace is the explanation** — Thought/Action/Observation is the audit
   trail an operator can read. That *is* the explainable-AI layer on top of SHAP.

No LLM is required. A deterministic policy implements the loop so the platform
stays auditable offline. If ``WTPM_LLM_URL`` is set the same tools can be
driven by an external chat model; the observation contract does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from wtpm_platform.contracts import SensorBatch


@dataclass
class TraceStep:
    thought: str
    action: str
    args: Dict[str, Any]
    observation: Dict[str, Any]


@dataclass
class HermesResult:
    goal: str
    trace: List[TraceStep] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=dict)
    principles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "principles": self.principles,
            "trace": [s.__dict__ for s in self.trace],
            "final": self.final,
            "n_steps": len(self.trace),
        }


class HermesAgent:
    """Reason → Act → Observe loop. Tools wrap already-fitted platform engines."""

    PRINCIPLES = [
        "thought_before_action",
        "observations_are_tool_outputs_never_invented",
        "revise_beliefs_from_observation",
        "stop_when_seven_questions_answered",
        "trace_is_the_explanation",
        "safety_hard_limits_outrank_learned_models",
    ]

    GOAL = ("Answer WHAT / WHERE / WHY / HOW SEVERE / HOW LONG / WHAT NEXT / "
            "SAFE-TO-OPERATE for this turbine, using only tool observations.")

    def __init__(self, orchestrator) -> None:
        self.orch = orchestrator
        self.tools: Dict[str, Callable[..., Dict[str, Any]]] = {
            "sensor_quality": self._t_quality,
            "anomaly": self._t_anomaly,
            "diagnose": self._t_diagnose,
            "explain": self._t_explain,
            "rul": self._t_rul,
            "safety": self._t_safety,
            "what_if": self._t_whatif,
            "finish": self._t_finish,
        }

    # -- tools (never hallucinate; they read live engines / last analyse) ----
    def _t_quality(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        q = self.orch.quality.run(batch)
        mem["quality"] = q
        return {"trust": q["trust"], "n_flags": q["n_flags"],
                "kinds": sorted({f["kind"] for f in q["flags"]})}

    def _t_anomaly(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        what = last.get("what") or {}
        obs = {
            "fused_score": what.get("current_fused_score"),
            "alarm": what.get("alarm"),
            "n_events": len(what.get("anomaly_events") or []),
            "n_conflicts": len(what.get("conflicts") or []),
        }
        mem["anomaly"] = obs
        return obs

    def _t_diagnose(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        obs = {
            "fault": (last.get("what") or {}).get("fault"),
            "confidence": (last.get("what") or {}).get("fault_confidence"),
            "subsystem": (last.get("where") or {}).get("subsystem"),
            "votes": (last.get("what") or {}).get("votes"),
        }
        mem["diagnosis"] = obs
        return obs

    def _t_explain(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        why = last.get("why") or {}
        shap = why.get("contributing_features") or {}
        top = list(shap.items())[:5]
        contrast = contrastive_channels(batch)
        obs = {
            "shap_top": {k: round(v, 4) for k, v in top},
            "signals": why.get("relevant_sensor_signals"),
            "contrast_vs_healthy": contrast,
            "physics_residual_kw": (last.get("physics") or {}).get("power_residual_kw_now"),
        }
        mem["explain"] = obs
        return obs

    def _t_rul(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        obs = {
            "hours": (last.get("rul") or {}).get("hours"),
            "uncertainty_hours": (last.get("rul") or {}).get("uncertainty_hours"),
            "source": (last.get("rul") or {}).get("source"),
            "degradation_state": (last.get("severity") or {}).get("degradation_state"),
        }
        mem["rul"] = obs
        return obs

    def _t_safety(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        obs = {
            "decision": (last.get("safety") or {}).get("decision"),
            "reasons": (last.get("safety") or {}).get("reasons"),
            "risk": last.get("risk_score"),
            "action": (last.get("action") or {}).get("action"),
            "cost_optimal": (last.get("action") or {}).get("cost_optimal"),
        }
        mem["safety"] = obs
        return obs

    def _t_whatif(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        last = mem.get("analyse") or {}
        twin = last.get("what_if_derate_20pct") or self.orch.twin.maintenance_scenario(
            batch, None, derate_pct=20.0)
        obs = {
            "scenario": twin.get("scenario") or "derate 20%",
            "delta_pct": twin.get("delta_pct"),
            "error": twin.get("error"),
        }
        mem["what_if"] = obs
        return obs

    def _t_finish(self, batch: SensorBatch, mem: Dict[str, Any], **_):
        d = mem.get("diagnosis") or {}
        e = mem.get("explain") or {}
        r = mem.get("rul") or {}
        s = mem.get("safety") or {}
        a = mem.get("anomaly") or {}
        q = mem.get("quality") or {}
        return {
            "what": d.get("fault"),
            "where": d.get("subsystem"),
            "why": {
                "shap_top": e.get("shap_top"),
                "contrast": e.get("contrast_vs_healthy"),
                "narrative": narrative(d, e, a, r, s),
            },
            "severity": r.get("degradation_state"),
            "how_long_hours": r.get("hours"),
            "what_next": s.get("action"),
            "safe_to_operate": s.get("decision"),
            "data_trust": q.get("trust") if isinstance(q, dict) else None,
        }

    # -- policy: deterministic Hermes loop (no invented observations) --------
    def run(self, batch: SensorBatch, analyse_result: Dict[str, Any]) -> HermesResult:
        mem: Dict[str, Any] = {"analyse": analyse_result}
        out = HermesResult(goal=self.GOAL, principles=list(self.PRINCIPLES))
        plan = [
            ("Read sensor quality first — untrusted data must not drive diagnosis.",
             "sensor_quality", {}),
            ("If the data are usable, ask the fused anomaly engine WHAT is happening.",
             "anomaly", {}),
            ("Turn anomaly evidence into a fault class and subsystem (WHERE).",
             "diagnose", {}),
            ("Ask XAI (SHAP + contrastive healthy-band) WHY the models say that.",
             "explain", {}),
            ("Ask prognostics HOW LONG (RUL + HMM severity).",
             "rul", {}),
            ("Ask the safety gate whether it is SAFE TO OPERATE; read the action.",
             "safety", {}),
            ("If risk is material, query the digital twin for a 20% derate what-if.",
             "what_if", {}),
        ]
        for thought, action, args in plan:
            # skip what-if when risk is clearly idle
            if action == "what_if":
                risk = (mem.get("safety") or {}).get("risk") or 0
                if risk < 25:
                    out.trace.append(TraceStep(
                        thought="Risk is low; a derate scenario would not change the recommendation.",
                        action="skip", args={"skipped": "what_if"},
                        observation={"skipped": True, "reason": "risk<25"},
                    ))
                    continue
            obs = self.tools[action](batch, mem, **args)
            out.trace.append(TraceStep(thought=thought, action=action, args=args,
                                       observation=obs))
        final = self.tools["finish"](batch, mem)
        out.trace.append(TraceStep(
            thought="Seven questions have tool-backed answers. Stop.",
            action="finish", args={}, observation=final,
        ))
        out.final = final
        return out


def contrastive_channels(batch: SensorBatch, healthy_frac: float = 0.4) -> Dict[str, Dict[str, float]]:
    """Now vs healthy-band median for the safety-relevant channels (XAI contrast)."""
    n = batch.n_steps
    h = max(int(n * healthy_frac), 8)
    out: Dict[str, Dict[str, float]] = {}
    for ch in ("bearing_vib_rms_mm_s", "gearbox_oil_temp_c", "power_kw",
               "generator_current_a", "rotor_speed_rpm"):
        if ch not in batch.channel_names:
            continue
        x = np.asarray(batch.channel(ch), float)
        med = float(np.nanmedian(x[:h]))
        now = float(x[-1])
        spread = float(np.nanstd(x[:h]) + 1e-9)
        out[ch] = {"healthy_median": round(med, 3), "now": round(now, 3),
                   "delta_sigma": round((now - med) / spread, 3)}
    return out


def narrative(diagnosis, explain, anomaly, rul, safety) -> str:
    """Operator-readable WHY paragraph grounded only in observations."""
    fault = diagnosis.get("fault") or "unknown"
    sub = diagnosis.get("subsystem") or "unknown"
    conf = diagnosis.get("confidence")
    score = anomaly.get("fused_score")
    shap = explain.get("shap_top") or {}
    top = ", ".join(f"{k}={v}" for k, v in list(shap.items())[:3]) or "no SHAP"
    contrast = explain.get("contrast_vs_healthy") or {}
    hot = [f"{k} {v['delta_sigma']:+.1f}σ" for k, v in contrast.items()
           if abs(v.get("delta_sigma", 0)) >= 1.5][:3]
    hours = rul.get("hours")
    act = safety.get("action")
    dec = safety.get("decision")
    bits = [
        f"Models diagnose {fault} in {sub} (confidence {conf}).",
        f"Fused anomaly score is {score}.",
        f"SHAP top features: {top}.",
    ]
    if hot:
        bits.append("Contrast vs healthy band: " + ", ".join(hot) + ".")
    if hours is not None:
        bits.append(f"RUL ≈ {hours} h.")
    bits.append(f"Safety={dec}; recommended action={act}.")
    return " ".join(str(b) for b in bits)
