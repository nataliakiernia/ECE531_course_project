"""
CSP elements for the TinyStrawEnv (1-D robot straw delivery task).

"In-depth jittery/noisy scenario" evaluation:
- TinyStrawState exposes x (observed), sigma_obs, target_nominal, target_effective, tolerance.
- Adds ROBUST hard-safety constraints: shrink feasible set by robust_margin = robust_k * sigma_obs.
- Learns preference relative to effective target when provided (info["effective_target"] / explanation).
- Adds a second objective: maximize robust safety clearance (margin) + speed.
- Adds CBTL-style confidence helper: comfort_prob(), and robust_k setter.
- Policy termination tolerance adapts with noise (avoid thrashing under jitter).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from sklearn.neighbors import RadiusNeighborsClassifier

from multitask_personalization.csp_generation import CSPConstraintGenerator, CSPGenerator
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)

# -----------------------------
# Straw-specific dataclasses
# -----------------------------

@dataclass(frozen=True)
class TinyStrawState:
    # Observed x used by policy (x_obs)
    x: float
    # Observation noise scale
    sigma_obs: float
    # Nominal target (static preference target)
    target_nominal: float
    # Effective target (nominal + head offset drift)
    target_effective: float
    # User comfort tolerance (meters)
    tolerance: float


TinyAction = Tuple[int, float | None]  # (op, delta) where op in {0=move,1=done}

# ---------------------------------------------------------------------
# Internal helpers: geometry / robust safe sets
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _KeepOutGeometry:
    mouth_center: float = 0.000
    nose_center: float = 0.050
    eye_center: float = 0.090
    mouth_halfwidth_max: float = 0.015
    nose_radius: float = 0.012
    eye_radius: float = 0.015
    min_x: float = -0.20
    max_x: float = 0.10


def _interval(center: float, radius: float) -> tuple[float, float]:
    return center - radius, center + radius


def _is_in_interval(x: float, a: float, b: float) -> bool:
    return (x >= a) and (x <= b)


def _robust_safe_position(pos: float, geom: _KeepOutGeometry, robust_margin: float) -> bool:
    """Robustly safe feasibility check for a candidate planned position."""
    # Shrink global bounds
    min_x = geom.min_x + robust_margin
    max_x = geom.max_x - robust_margin
    if pos < min_x or pos > max_x:
        return False

    # Shrink mouth depth max
    mouth_max = geom.mouth_center + geom.mouth_halfwidth_max - robust_margin
    if pos > mouth_max:
        return False

    # Expand keep-out intervals
    nose_a, nose_b = _interval(geom.nose_center, geom.nose_radius + robust_margin)
    eye_a, eye_b = _interval(geom.eye_center, geom.eye_radius + robust_margin)

    if _is_in_interval(pos, nose_a, nose_b):
        return False
    if _is_in_interval(pos, eye_a, eye_b):
        return False

    return True


def _min_safety_margin_robust(pos: float, geom: _KeepOutGeometry, robust_margin: float) -> float:
    """
    Robust-aware clearance: minimum distance to any ROBUST boundary.
    Larger is better. Returns 0 if inside a robust keepout.
    """
    # robust-shrunk bounds
    min_x = geom.min_x + robust_margin
    max_x = geom.max_x - robust_margin
    m_bounds = min(pos - min_x, max_x - pos)

    # robust-shrunk mouth max
    mouth_max = geom.mouth_center + geom.mouth_halfwidth_max - robust_margin
    m_depth = mouth_max - pos

    # robust-expanded keepouts
    nose_a, nose_b = _interval(geom.nose_center, geom.nose_radius + robust_margin)
    eye_a, eye_b = _interval(geom.eye_center, geom.eye_radius + robust_margin)

    def margin_to_interval(x: float, a: float, b: float) -> float:
        if x < a:
            return a - x
        if x > b:
            return x - b
        return 0.0

    m_nose = margin_to_interval(pos, nose_a, nose_b)
    m_eye = margin_to_interval(pos, eye_a, eye_b)

    return float(min(m_bounds, m_depth, m_nose, m_eye))


def _get_info_effective_target(info: dict[str, Any]) -> Optional[float]:
    if "effective_target" in info:
        try:
            return float(info["effective_target"])
        except Exception:
            pass
    try:
        return float(info["explanation"]["preference"]["target_effective"])
    except Exception:
        return None


# ---------------------------------------------------------------------
# CSP POLICY
# ---------------------------------------------------------------------

class _TinyCSPPolicy(CSPPolicy[TinyStrawState, TinyAction]):
    """
    CSP-induced policy for TinyStraw.

    Termination tolerance adapts to sigma_obs and user tolerance:
      tol_done = max(user_tol, base_done_tol, robust_k * sigma_obs)
    """

    def __init__(
        self,
        csp_variables: Collection[CSPVariable],
        seed: int = 0,
        robust_k: float = 0.0,
        base_done_tol: float = 1e-3,  # 1 mm
    ) -> None:
        super().__init__(csp_variables, seed)
        self._target_position: float | None = None
        self._speed: float | None = None
        self._terminated: bool = False
        self._robust_k = float(robust_k)
        self._base_done_tol = float(base_done_tol)

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._target_position = float(self._get_value("position"))
        self._speed = float(self._get_value("speed"))
        self._terminated = False

    def step(self, obs: TinyStrawState) -> TinyAction:
        assert self._target_position is not None
        assert self._speed is not None

        x_obs = float(obs.x)

        tol_done = max(
            float(obs.tolerance),
            self._base_done_tol,
            self._robust_k * float(max(0.0, obs.sigma_obs)),
        )

        if abs(self._target_position - x_obs) < tol_done:
            self._terminated = True
            return (1, None)

        direction = float(np.clip(self._target_position - x_obs, -1.0, 1.0))
        delta = float(self._speed * direction)
        return (0, delta)

    def check_termination(self, obs: TinyStrawState) -> bool:
        return self._terminated


# ---------------------------------------------------------------------
# Preference / distance constraint generator
# ---------------------------------------------------------------------

class _TinyDistanceConstraintGenerator(CSPConstraintGenerator[TinyStrawState, TinyAction]):
    """
    Learns P(comfortable | distance_to_target_estimate).

    - At planning time: uses obs.target_effective (head drift included).
    - At learning time: uses effective_target if provided (info) else obs.target_effective.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._classifier: RadiusNeighborsClassifier | None = None
        self._training_inputs: list[NDArray] = []
        self._training_outputs: list[bool] = []

    def save(self, model_dir: Path) -> None:
        outfile = model_dir / "tiny_distance_constraint_classifier.pkl"
        with open(outfile, "wb") as f:
            import pickle as pkl
            pkl.dump(self._classifier, f)

    def load(self, model_dir: Path) -> None:
        outfile = model_dir / "tiny_distance_constraint_classifier.pkl"
        with open(outfile, "rb") as f:
            import pickle as pkl
            self._classifier = pkl.load(f)

    def generate(self, obs: TinyStrawState, csp_vars: list[CSPVariable], constraint_name: str) -> CSPConstraint:
        assert len(csp_vars) == 1
        position_var = next(iter(csp_vars))

        target_est = float(obs.target_effective)

        def _position_logprob(position: np.float_) -> float:
            if self._classifier is None:
                return 0.0
            dist = abs(float(position) - target_est)
            feats = self._featurize_input(dist)
            prob = self._classifier.predict_proba([feats])[0][1]
            return float(np.log(prob + 1e-9))

        return LogProbCSPConstraint(
            constraint_name,
            [position_var],
            _position_logprob,
            threshold=float(np.log(0.5)),
        )

    def learn_from_transition(
        self,
        obs: TinyStrawState,
        act: TinyAction,
        next_obs: TinyStrawState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if act[0] != 1:
            return

        label = bool(info.get("user_satisfaction", 0.0) > 0.0)

        # Prefer x_true if env provides it; fallback to observed
        try:
            x_at_done = float(info.get("x_true"))
        except Exception:
            x_at_done = float(obs.x)

        eff_target = _get_info_effective_target(info)
        if eff_target is None:
            eff_target = float(obs.target_effective)

        dist = abs(x_at_done - float(eff_target))

        self._training_inputs.append(self._featurize_input(dist))
        self._training_outputs.append(label)
        self._update_constraint_parameters()

    def get_metrics(self) -> dict[str, float]:
        return {}

    def _update_constraint_parameters(self) -> None:
        if len(set(self._training_outputs)) < 2:
            return
        # Radius is huge => behaves like weighted KNN on 1D feature
        self._classifier = RadiusNeighborsClassifier(radius=1000.0, weights="distance")
        self._classifier.fit(self._training_inputs, self._training_outputs)

    def _featurize_input(self, dist: float | np.floating) -> NDArray:
        return np.array([float(dist)], dtype=float)


# ---------------------------------------------------------------------
# CSP Generator
# ---------------------------------------------------------------------

class TinyStrawCSPGenerator(CSPGenerator[TinyStrawState, TinyAction]):
    """
    Generator for TinyStraw CSP.

    Additions:
      - robust hard-safety constraint on planned position
      - speed + robust safety margin objective
      - CBTL-style confidence helper: comfort_prob()
      - robust_k setter for confidence-gated conservatism
    """

    def __init__(
        self,
        robust_k: float = 0.0,
        keepout_geom: Optional[_KeepOutGeometry] = None,
        w_speed: float = 1.0,
        w_margin: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pref_gen = _TinyDistanceConstraintGenerator(seed=self._seed)
        self._robust_k = float(robust_k)
        self._geom = keepout_geom or _KeepOutGeometry()
        self._w_speed = float(w_speed)
        self._w_margin = float(w_margin)

    # ---------- Public helpers (confidence / gating) ----------

    def set_robust_k(self, k: float) -> None:
        self._robust_k = float(k)

    def get_robust_k(self) -> float:
        return float(self._robust_k)

    def comfort_prob(self, obs: TinyStrawState, position: float) -> float:
        """p(comfortable | distance to obs.target_effective)."""
        clf = self._pref_gen._classifier
        if clf is None:
            return 0.5
        dist = abs(float(position) - float(obs.target_effective))
        feats = np.array([dist], dtype=float)
        return float(clf.predict_proba([feats])[0][1])

    # ---------- Save / load ----------

    def save(self, model_dir: Path) -> None:
        self._pref_gen.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._pref_gen.load(model_dir)

    # ---------- CSP pieces ----------

    def _generate_variables(self, obs: TinyStrawState) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        position = CSPVariable("position", Box(-np.inf, np.inf, shape=(), dtype=np.float_))
        speed = CSPVariable("speed", Box(0.0, 1.0, shape=(), dtype=np.float_))

        # IMPORTANT: initialize near effective target (matches preference constraint)
        init_center = float(obs.target_effective)

        initialization = {
            position: float(self._rng.normal(loc=init_center, scale=0.01)),
            speed: float(self._rng.uniform(0.1, 1.0)),
        }
        return [position, speed], initialization

    def _generate_personal_constraints(self, obs: TinyStrawState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        position, _ = variables
        return [self._pref_gen.generate(obs, [position], "user_preference")]

    def _generate_nonpersonal_constraints(self, obs: TinyStrawState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        position, _ = variables
        robust_margin = self._robust_k * float(max(0.0, obs.sigma_obs))

        def _hard_safety_logprob(pos: np.float_) -> float:
            ok = _robust_safe_position(float(pos), self._geom, robust_margin)
            return 0.0 if ok else -1e9

        return [
            LogProbCSPConstraint(
                "hard_safety_robust",
                [position],
                _hard_safety_logprob,
                threshold=-1e6,
            )
        ]

    def _generate_exploit_cost(self, obs: TinyStrawState, variables: list[CSPVariable]) -> CSPCost | None:
        """
        Combined objective (single CSPCost):
          minimize = w_speed*(1-speed) + w_margin*(1/max(margin,eps))
        """
        position, speed = variables
        robust_margin = self._robust_k * float(max(0.0, obs.sigma_obs))

        w_speed = float(self._w_speed)
        w_margin = float(self._w_margin)

        def _combined_cost_fn(pos: np.float_, spd: np.float_) -> float:
            spd_val = float(spd)
            pos_val = float(pos)

            speed_cost = 1.0 - spd_val

            m = _min_safety_margin_robust(pos_val, self._geom, robust_margin)
            margin_cost = 1.0 / max(m, 1e-4)

            return w_speed * speed_cost + w_margin * margin_cost

        return CSPCost("speed_plus_margin", [position, speed], _combined_cost_fn)

    def _generate_samplers(self, obs: TinyStrawState, csp: CSP) -> list[CSPSampler]:
        position, speed = csp.variables

        robust_margin = self._robust_k * float(max(0.0, obs.sigma_obs))

        min_x = self._geom.min_x + robust_margin
        max_x = self._geom.max_x - robust_margin
        mouth_max = self._geom.mouth_center + self._geom.mouth_halfwidth_max - robust_margin

        # IMPORTANT: sample around effective target
        target_center = float(obs.target_effective)

        def _sample_position_fn(_: dict[CSPVariable, Any], rng: np.random.Generator) -> dict[CSPVariable, Any]:
            # biased rejection sampling
            for _ in range(200):
                cand = float(rng.normal(loc=target_center, scale=0.01))
                cand = float(np.clip(cand, min_x, min(max_x, mouth_max)))
                if _robust_safe_position(cand, self._geom, robust_margin):
                    return {position: cand}

            for _ in range(500):
                cand = float(rng.uniform(min_x, min(max_x, mouth_max)))
                if _robust_safe_position(cand, self._geom, robust_margin):
                    return {position: cand}

            return {position: float(np.clip(target_center, min_x, min(max_x, mouth_max)))}

        def _sample_speed_fn(_: dict[CSPVariable, Any], rng: np.random.Generator) -> dict[CSPVariable, Any]:
            return {speed: float(rng.uniform(0.1, 1.0))}

        return [
            FunctionalCSPSampler(_sample_position_fn, csp, {position}),
            FunctionalCSPSampler(_sample_speed_fn, csp, {speed}),
        ]

    def _generate_policy(self, obs: TinyStrawState, csp_variables: Collection[CSPVariable]) -> CSPPolicy:
        return _TinyCSPPolicy(csp_variables, seed=self._seed, robust_k=self._robust_k, base_done_tol=1e-3)

    # ---------- Online updates ----------

    def observe_transition(
        self,
        obs: TinyStrawState,
        act: TinyAction,
        next_obs: TinyStrawState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._pref_gen.learn_from_transition(obs, act, next_obs, done, info)

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()




