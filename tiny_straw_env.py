# tiny_straw_env.py
# (based on the tiny.env template from
# tomsilver/multitask-personalization/src/multitask_personalization/envs/tiny/)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import gymnasium as gym
import numpy as np


# Geometry (1-D axis): smaller x = farther from face, larger x = toward forehead
# Landmarks along this axis:
# mouth at x = 0.00 m, nose at +0.05 m, eye at +0.09 m (example numbers)
#
# Key additions for "jittery user / noisy input":
# - True state: x_true (physics)
# - Observation: x_obs (what policy/planner sees)
# - Latent head motion: head_offset (can update every timestep, toggleable)
# - Effective comfort target shifts with head_offset (preference frame drifts)
# - Safety projection can be toggled off to create failure-case baselines
# - Optional terminate_on_unsafe baseline when projection is off
# - Unified min_margin_any metric for robust-margin evaluation / plotting


@dataclass(frozen=True)
class StrawHiddenSpec:
    # User comfort/preference around mouth contact (learnable)
    desired_mouth_offset: float = 0.010   # 10 mm inside mouth
    tolerance: float = 0.004              # ±4 mm comfort window


@dataclass(frozen=True)
class FaceKeepOutSpec:
    mouth_center: float = 0.000
    nose_center: float = 0.050
    eye_center: float = 0.090

    mouth_halfwidth_max: float = 0.015    # never go deeper than 15 mm
    nose_radius: float = 0.012
    eye_radius: float = 0.015

    min_x: float = -0.20
    max_x: float = 0.10


@dataclass(frozen=True)
class JitterSpec:
    # Observation noise (sensor jitter)
    sigma_obs: float = 0.0          # meters (std dev)

    # Head motion / user jitter: AR(1) latent offset in meters
    sigma_head: float = 0.0         # meters (innovation std dev)
    rho_head: float = 0.0           # AR(1) coefficient in [0, 1)
    head_clip: float = 0.01         # clamp |head_offset| to this many meters


class TinyStrawEnv(gym.Env):
    """
    1-D 'drinking with a straw' toy environment.

    True state: x_true (float) = straw tip position along the face axis (meters).
    Observation: x_obs = x_true + N(0, sigma_obs^2) (what the policy sees).

    Latent user head motion:
      head_offset_{t+1} = rho_head * head_offset_t + N(0, sigma_head^2)

    IMPORTANT FIX (Dec 2025):
      By default, we DO NOT update head motion on the DONE step.
      Otherwise, the effective target can jump right before evaluation,
      making "done" look wrong even when the robot is at the right place.

    Actions: (op, delta)
      - op=0: move by delta in [-max_step, +max_step] (bounded)
      - op=1: declare done (evaluate success/comfort)

    Hard safety:
      - Never enter nose / eye keep-out intervals.
      - Never exceed mouth penetration limit.
      - Stay within [min_x, max_x].

    Safety projection:
      - If project_actions=True, proposed moves are projected to the nearest safe value.
      - If project_actions=False, moves are applied directly (unsafe states possible),
        enabling failure-case baselines.

    terminate_on_unsafe (baseline option):
      - If True and project_actions=False, the episode ends immediately upon entering unsafe.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        hidden: Optional[StrawHiddenSpec] = None,
        keepout: Optional[FaceKeepOutSpec] = None,
        jitter: Optional[JitterSpec] = None,
        x0: float = -0.12,
        max_step: float = 0.01,
        seed: int = 0,
        eval_mode: bool = False,
        project_actions: bool = True,
        # NEW flags for ablations / baselines
        update_head_each_step: bool = True,
        add_obs_noise: bool = True,
        terminate_on_unsafe: bool = False,
        # NEW: control whether head motion updates on DONE step
        update_head_on_done: bool = False,
        # NEW: optional "freeze head per episode" mode (recommended for TinyStraw sanity)
        sample_head_on_reset: bool = False,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.hidden = hidden or StrawHiddenSpec()
        self.keep = keepout or FaceKeepOutSpec()
        self.jitter = jitter or JitterSpec()

        self.x0 = float(x0)
        self.max_step = float(max_step)
        self.eval_mode = bool(eval_mode)

        self.project_actions = bool(project_actions)
        self.update_head_each_step = bool(update_head_each_step)
        self.add_obs_noise = bool(add_obs_noise)
        self.terminate_on_unsafe = bool(terminate_on_unsafe)
        self.update_head_on_done = bool(update_head_on_done)
        self.sample_head_on_reset = bool(sample_head_on_reset)

        # True state + latent head motion
        self.x_true = float(self.x0)
        self.head_offset = 0.0

        # action space: (op, delta)
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2),  # 0=move, 1=done
            gym.spaces.Box(low=-self.max_step, high=self.max_step, shape=(1,), dtype=np.float32),
        ))

        # observation is x_obs, clipped to [min_x, max_x] for space definition
        self.observation_space = gym.spaces.Box(
            low=np.array([self.keep.min_x], dtype=np.float32),
            high=np.array([self.keep.max_x], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

    # -------------------------
    # Jitter helpers
    # -------------------------
    def _step_head_motion(self) -> None:
        """Update latent head motion."""
        if self.jitter.sigma_head <= 0.0:
            return

        eta = float(self.rng.normal(0.0, self.jitter.sigma_head))
        self.head_offset = float(self.jitter.rho_head * self.head_offset + eta)

        # Clip for realism / preventing extreme drift
        self.head_offset = float(np.clip(self.head_offset, -self.jitter.head_clip, self.jitter.head_clip))

    def _observe(self) -> float:
        """Return (possibly) noisy observation of x_true."""
        x_obs = float(self.x_true)
        if self.add_obs_noise and (self.jitter.sigma_obs > 0.0):
            eps = float(self.rng.normal(0.0, self.jitter.sigma_obs))
            x_obs = x_obs + eps

        # Keep observations in the declared observation range
        x_obs = float(np.clip(x_obs, self.keep.min_x, self.keep.max_x))
        return x_obs

    # -------------------------
    # Safety helpers
    # -------------------------
    def _interval(self, center: float, radius: float) -> Tuple[float, float]:
        return center - radius, center + radius

    def _in_interval(self, x: float, interval: Tuple[float, float]) -> bool:
        a, b = interval
        return (x >= a) and (x <= b)

    def _safe_project(self, x_curr: float, x_next: float) -> float:
        """
        Enforce hard safety by projecting proposed x_next to the closest safe value
        if it would enter any keep-out interval.
        """
        # Always clip to global bounds
        x_next = float(np.clip(x_next, self.keep.min_x, self.keep.max_x))

        # Mouth penetration bound (do not exceed 'max depth' into mouth)
        mouth_max = self.keep.mouth_center + self.keep.mouth_halfwidth_max
        if x_next > mouth_max:
            x_next = mouth_max

        # Nose and eye keep-outs
        nose_iv = self._interval(self.keep.nose_center, self.keep.nose_radius)
        eye_iv = self._interval(self.keep.eye_center, self.keep.eye_radius)

        def project_from_left(x_try: float, interval: Tuple[float, float]) -> float:
            a, _ = interval
            if (x_curr <= a) and (x_try > a):
                return float(np.nextafter(a, -np.inf))  # just before boundary
            return x_try

        def project_from_right(x_try: float, interval: Tuple[float, float]) -> float:
            _, b = interval
            if (x_curr >= b) and (x_try < b):
                return float(np.nextafter(b, +np.inf))
            return x_try

        if x_next > x_curr:
            x_next = project_from_left(x_next, nose_iv)
            x_next = project_from_left(x_next, eye_iv)
        elif x_next < x_curr:
            x_next = project_from_right(x_next, eye_iv)
            x_next = project_from_right(x_next, nose_iv)

        for iv in (nose_iv, eye_iv):
            if self._in_interval(x_next, iv):
                a, b = iv
                x_next = a if x_curr < a else b

        return float(x_next)

    def _unsafe(self, x: float) -> bool:
        nose_iv = self._interval(self.keep.nose_center, self.keep.nose_radius)
        eye_iv = self._interval(self.keep.eye_center, self.keep.eye_radius)
        mouth_max = self.keep.mouth_center + self.keep.mouth_halfwidth_max
        return (
            self._in_interval(x, nose_iv)
            or self._in_interval(x, eye_iv)
            or (x > mouth_max)
            or (x < self.keep.min_x)
            or (x > self.keep.max_x)
        )

    def _min_margin_to_any_boundary(self, x: float) -> float:
        """
        Minimum margin to any constraint boundary.
        Positive => safely away from all boundaries.
        Negative => violation depth into a forbidden region/bound.
        """
        nose_a, nose_b = self._interval(self.keep.nose_center, self.keep.nose_radius)
        eye_a, eye_b = self._interval(self.keep.eye_center, self.keep.eye_radius)
        mouth_max = self.keep.mouth_center + self.keep.mouth_halfwidth_max

        def margin_to_interval(xv: float, a: float, b: float) -> float:
            if xv < a:
                return a - xv
            if xv > b:
                return xv - b
            return -min(xv - a, b - xv)

        m_nose = margin_to_interval(x, nose_a, nose_b)
        m_eye = margin_to_interval(x, eye_a, eye_b)

        m_minx = x - self.keep.min_x
        m_maxx = self.keep.max_x - x
        m_depth = mouth_max - x

        return float(min(m_nose, m_eye, m_minx, m_maxx, m_depth))

    # -------------------------
    # Preference helpers
    # -------------------------
    def _effective_target(self) -> float:
        return float(self.keep.mouth_center + self.hidden.desired_mouth_offset + self.head_offset)

    def _within_comfort(self, x_true: float) -> bool:
        target = self._effective_target()
        return abs(x_true - target) <= self.hidden.tolerance

    def _pref_score(self, x_true: float) -> float:
        if self._unsafe(x_true):
            return 0.0
        target = self._effective_target()
        sig2 = (self.hidden.tolerance ** 2) + 1e-8
        return float(np.exp(-((x_true - target) ** 2) / (2 * sig2)))

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.x_true = float(self.x0)
        self.head_offset = 0.0

        # Optional: sample head offset once per episode (helps solvability in TinyStraw)
        if self.sample_head_on_reset and (self.jitter.sigma_head > 0.0):
            self.head_offset = float(self.rng.normal(0.0, self.jitter.sigma_head))
            self.head_offset = float(np.clip(self.head_offset, -self.jitter.head_clip, self.jitter.head_clip))

        x_obs = self._observe()
        info = self._info(robot_indicated_done=False, x_obs=x_obs)
        return np.array([x_obs], dtype=np.float32), info

    def step(self, action: Tuple[int, np.ndarray]):
        op, val = action
        done = False

        # -----------------------------
        # IMPORTANT FIX:
        # Update head motion on MOVE steps by default,
        # and only update on DONE if explicitly requested.
        # -----------------------------
        if self.update_head_each_step:
            if op == 0:
                self._step_head_motion()
            elif (op == 1) and self.update_head_on_done:
                self._step_head_motion()

        if op == 0:
            delta = float(np.clip(val[0], -self.max_step, self.max_step))
            proposed = float(self.x_true + delta)

            if self.project_actions:
                self.x_true = self._safe_project(self.x_true, proposed)
            else:
                # Failure-case baseline: apply directly (can enter unsafe)
                self.x_true = float(np.clip(proposed, self.keep.min_x, self.keep.max_x))

                if self.terminate_on_unsafe and self._unsafe(self.x_true):
                    done = True

        elif op == 1:
            done = True
        else:
            raise ValueError("Invalid op")

        x_obs = self._observe()
        info = self._info(robot_indicated_done=(op == 1), x_obs=x_obs)

        reward = 0.0
        truncated = False
        return np.array([x_obs], dtype=np.float32), reward, done, truncated, info

    def render(self):
        pass

    # -------------------------
    # Info / explanation
    # -------------------------
    def _info(self, robot_indicated_done: bool, x_obs: Optional[float] = None) -> Dict[str, Any]:
        x_true = float(self.x_true)
        x_obs_val = float(x_obs) if x_obs is not None else float(self._observe())

        unsafe = self._unsafe(x_true)

        nose_a, nose_b = self._interval(self.keep.nose_center, self.keep.nose_radius)
        eye_a, eye_b = self._interval(self.keep.eye_center, self.keep.eye_radius)

        def margin_to_interval(x: float, a: float, b: float) -> float:
            if x < a:
                return a - x
            if x > b:
                return x - b
            return -(min(x - a, b - x))

        m_nose = margin_to_interval(x_true, nose_a, nose_b)
        m_eye = margin_to_interval(x_true, eye_a, eye_b)

        mouth_max = self.keep.mouth_center + self.keep.mouth_halfwidth_max
        m_depth = mouth_max - x_true

        min_margin_any = self._min_margin_to_any_boundary(x_true)

        if robot_indicated_done:
            if unsafe:
                sat = -1.0
                feedback = "Unsafe: near nose/eye or too deep."
            else:
                sat = 1.0 if self._within_comfort(x_true) else 0.0
                feedback = "Perfect sip." if sat > 0 else "Not comfortable yet."
        else:
            sat = 0.0
            feedback = "In progress."

        target_nominal = float(self.keep.mouth_center + self.hidden.desired_mouth_offset)
        target_effective = self._effective_target()

        explanation = {
            "active_constraints": {
                "bounds": [self.keep.min_x, self.keep.max_x],
                "nose_keepout": [nose_a, nose_b],
                "eye_keepout": [eye_a, eye_b],
                "mouth_depth_max": float(mouth_max),
                "per_step_bound": float(self.max_step),

                "project_actions": bool(self.project_actions),
                "terminate_on_unsafe": bool(self.terminate_on_unsafe),
                "update_head_each_step": bool(self.update_head_each_step),
                "update_head_on_done": bool(self.update_head_on_done),
                "sample_head_on_reset": bool(self.sample_head_on_reset),
                "add_obs_noise": bool(self.add_obs_noise),
            },
            "tightness": {
                "margin_to_nose": float(m_nose),
                "margin_to_eye": float(m_eye),
                "margin_to_depth": float(m_depth),
                "min_margin_any": float(min_margin_any),
            },
            "preference": {
                "target_nominal": float(target_nominal),
                "target_effective": float(target_effective),
                "tolerance": float(self.hidden.tolerance),
            },
            "jitter": {
                "sigma_obs": float(self.jitter.sigma_obs),
                "sigma_head": float(self.jitter.sigma_head),
                "rho_head": float(self.jitter.rho_head),
                "head_clip": float(self.jitter.head_clip),
            },
        }

        return {
            "robot_indicated_done": bool(robot_indicated_done),
            "user_satisfaction": float(sat),  # {-1,0,1}
            "feedback": str(feedback),
            "pref_score": self._pref_score(x_true),
            "explanation": explanation,
            "unsafe": bool(unsafe),

            # Key fields for analysis / CSP learning
            "x_true": float(x_true),
            "x_obs": float(x_obs_val),
            "head_offset": float(self.head_offset),
            "effective_target": float(target_effective),
            "min_margin_any": float(min_margin_any),
        }


# Minimal usage demo
if __name__ == "__main__":
    # Example: jittery user + noisy observations, with safety projection ON
    env = TinyStrawEnv(
        jitter=JitterSpec(sigma_obs=0.002, sigma_head=0.001, rho_head=0.8, head_clip=0.008),
        project_actions=True,
        update_head_each_step=True,
        update_head_on_done=False,     # IMPORTANT: keep this False
        sample_head_on_reset=False,    # set True to make TinyStraw easier/cleaner
        add_obs_noise=True,
        terminate_on_unsafe=False,
    )
    obs, info = env.reset()

    done = False
    steps = 0
    while not done and steps < 200:
        x_obs = float(obs[0])  # policy sees noisy
        target_nominal = env.keep.mouth_center + env.hidden.desired_mouth_offset

        # naive controller chases nominal target using noisy observation
        delta = float(np.clip(target_nominal - x_obs, -env.max_step, env.max_step))
        action = (0, np.array([delta], dtype=np.float32))
        obs, _, done, _, info = env.step(action)
        steps += 1

        # declare done if observed is near nominal (this may fail under jitter)
        if abs(float(obs[0]) - target_nominal) <= env.hidden.tolerance:
            obs, _, done, _, info = env.step((1, np.array([0.0], dtype=np.float32)))

    print("Finished:", info)


