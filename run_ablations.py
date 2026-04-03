"""
run_ablations.py

4-arm ablation:
  (1) NO_PERSONALIZATION: no learning, no param updates
  (2) UNFILTERED_LLM: LLM updates comfort params with minimal crash guards
  (3) CBTL_FRAMEWORK: no LLM; online preference learning + CBTL-style train/eval explore-exploit
  (4) MY_FRAMEWORK: filtered LLM updates + confidence-gated robust_k (+ optional speed cap)

CBTL baseline alignment (to original repo logic):
- In TRAIN mode with explore_method="max-entropy":
    - PERSONAL constraints are excluded (so don't "overfit" to early guess)
    - COST is maximize entropy of personal logprob constraints (no speed term)
- In EVAL mode:
    - PERSONAL constraints are included again
    - COST is exploit (comfort + optional speed bias)

NEW DEBUG ADDITIONS:
- CBTL plan debug printouts each replan: mode(train/eval) + exclude_personal + objective + median entropy/prob + frac high-entropy
- CBTL transition counter per episode
- CBTL end-of-episode metrics dump via gen.get_metrics()
"""

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tiny_straw_env import TinyStrawEnv, JitterSpec, StrawHiddenSpec
from tiny_straw_csp import TinyStrawState, TinyStrawCSPGenerator

from openai import OpenAI

client = OpenAI()


# =========================
# Ablation modes
# =========================
class Mode(str, Enum):
    NO_PERSONALIZATION = "no_personalization_baseline"
    UNFILTERED_LLM = "unfiltered_llm_updates_baseline"
    CBTL_FRAMEWORK = "cbtl_framework_baseline"
    MY_FRAMEWORK = "my_framework"


# =========================
# Logging structures
# =========================
@dataclass
class EpisodeResult:
    mode: str
    seed: int
    episode: int
    steps: int

    unsafe_steps: int
    done_satisfaction: float
    final_pref_score: float

    final_true_error: float
    avg_true_error: float

    min_margin_to_nose: float
    min_margin_to_eye: float
    min_margin_to_depth: float

    desired_offset: float
    tolerance: float

    llm_calls: int


@dataclass
class LLMStats:
    num_calls: int = 0


# =========================
# Math helpers
# =========================
def bernoulli_entropy_bits(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


# =========================
# MC "planner" (baseline-friendly)
# =========================
def solve_tiny_csp_mc(
    gen: TinyStrawCSPGenerator,
    obs: TinyStrawState,
    rng: np.random.Generator,
    num_samples: int = 600,
    speed_cap: Optional[Tuple[float, float]] = None,
    objective: str = "exploit",  # "exploit" or "entropy"
    exclude_personal_constraints: bool = False,  # CBTL-train behavior
) -> Tuple[List[Any], Dict[Any, float]]:
    """
    Emulates CSP solve in a way that matches CBTL semantics:
      - If exclude_personal_constraints=True: do NOT enforce p_ok >= 0.5 threshold.
      - If exclude_personal_constraints=False: enforce the personal constraint threshold (p_ok >= 0.5).

    Also matches CBTL max-entropy cost: entropy objective should NOT include speed.
    """
    vars_, _ = gen._generate_variables(obs)
    position_var, speed_var = vars_

    best_score = -np.inf
    best_solution: Optional[Dict[Any, float]] = None

    # Exploit gets a speed bias 
    alpha_speed_exploit = 0.05

    # Entropy exploration should match CBTL: NO speed bias.
    alpha_speed_entropy = 0.0

    sample_sigma = 0.01

    for _ in range(num_samples):
        pos = float(rng.normal(loc=float(obs.target_nominal), scale=sample_sigma))

        if speed_cap is None:
            spd = float(rng.uniform(0.1, 1.0))
        else:
            spd = float(rng.uniform(speed_cap[0], speed_cap[1]))

        p_ok = float(gen.comfort_prob(obs, pos))

        # Enforce personal constraint ONLY when not excluded (CBTL eval / exploit).
        if (not exclude_personal_constraints) and (p_ok < 0.5):
            continue

        if objective == "entropy":
            score = bernoulli_entropy_bits(p_ok) + alpha_speed_entropy * spd
        else:
            score = float(np.log(p_ok + 1e-9)) + alpha_speed_exploit * spd

        if score > best_score:
            best_score = score
            best_solution = {position_var: pos, speed_var: spd}

    # If we filtered everything out (possible early), fall back to best ignoring threshold.
    if best_solution is None:
        # fallback: allow violating personal constraint, but still exploit score
        best_score = -np.inf
        for _ in range(num_samples):
            pos = float(rng.normal(loc=float(obs.target_nominal), scale=sample_sigma))
            spd = float(rng.uniform(0.1, 1.0)) if speed_cap is None else float(rng.uniform(speed_cap[0], speed_cap[1]))
            p_ok = float(gen.comfort_prob(obs, pos))
            score = float(np.log(p_ok + 1e-9)) + 0.05 * spd
            if score > best_score:
                best_score = score
                best_solution = {position_var: pos, speed_var: spd}

    assert best_solution is not None
    return vars_, best_solution


# =========================
# My framework gating
# =========================
def robust_k_from_conf(conf: float) -> float:
    if conf < 0.6:
        return 3.0
    if conf < 0.8:
        return 2.0
    return 1.0


# =========================
# LLM prompt + updates
# =========================
def _query_llm(prompt: str, stats: Optional[LLMStats] = None) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    if stats is not None:
        stats.num_calls += 1
    return resp.choices[0].message.content.strip()


def build_update_prompt(env: TinyStrawEnv) -> str:
    current_target_mm = env.hidden.desired_mouth_offset * 1000.0
    current_tol_mm = env.hidden.tolerance * 1000.0

    safe_min_mm = 0.0
    safe_max_mm = (env.keep.mouth_halfwidth_max - 1e-3) * 1000.0

    return f"""
Output NEW comfort parameters for a drinking straw.

Current:
- desired_mouth_offset_mm = {current_target_mm:.1f}
- tolerance_mm = {current_tol_mm:.1f}

Safety bounds:
- desired_mouth_offset_mm must be in [{safe_min_mm:.1f}, {safe_max_mm:.1f}]
- tolerance_mm > 0 and tolerance_mm < {safe_max_mm:.1f}

Return JSON ONLY:
{{
  "desired_mouth_offset_mm": <float>,
  "tolerance_mm": <float>
}}
""".strip()


def verify_llm_update_safe(
    suggestion: Dict[str, float], env: TinyStrawEnv
) -> Tuple[bool, Any]:
    try:
        target = float(suggestion["desired_mouth_offset_mm"]) / 1000.0
        tol = float(suggestion["tolerance_mm"]) / 1000.0
    except Exception as e:
        return False, f"Bad JSON fields: {e}"

    if tol <= 0:
        return False, "tol must be positive"

    safe_min = 0.0
    safe_max = env.keep.mouth_halfwidth_max - 1e-4

    if not (safe_min <= target <= safe_max):
        return False, "target outside mouth-safe region"

    if (target - tol) < safe_min or (target + tol) > safe_max:
        return False, "comfort window violates mouth-safe region"

    return True, replace(env.hidden, desired_mouth_offset=target, tolerance=tol)


def apply_llm_update_unfiltered(suggestion: Dict[str, float], env: TinyStrawEnv) -> Any:
    target = float(
        suggestion.get("desired_mouth_offset_mm", env.hidden.desired_mouth_offset * 1000.0)
    ) / 1000.0
    tol = float(suggestion.get("tolerance_mm", env.hidden.tolerance * 1000.0)) / 1000.0

    if not np.isfinite(target):
        target = env.hidden.desired_mouth_offset
    if not np.isfinite(tol) or tol <= 0:
        tol = env.hidden.tolerance

    return replace(env.hidden, desired_mouth_offset=target, tolerance=tol)


# =========================
# Run one episode
# =========================
def run_episode(
    mode: Mode,
    base_seed: int,
    episode: int,
    jitter: JitterSpec,
    project_actions: bool,
    gen: TinyStrawCSPGenerator,
    hidden_in: StrawHiddenSpec,
    plan_rng: np.random.Generator,
    cbtl_train_episodes: int,
    num_steps_cap: int = 60,
    replan_every: int = 1,
) -> Tuple[EpisodeResult, StrawHiddenSpec]:
    episode_seed = int(base_seed * 1000 + episode)
    env = TinyStrawEnv(seed=episode_seed, jitter=jitter, project_actions=project_actions)

    env.hidden = replace(hidden_in)

    llm_stats = LLMStats()
    fixed_hidden = replace(env.hidden)

    cbtl_transition_count = 0
    cbtl_plans_entropy = 0
    cbtl_plans_exploit = 0

    # -------------------------
    # Helper: build TinyStrawState with new required fields
    # -------------------------
    def make_state(obs_raw, info, env) -> TinyStrawState:
        expl = info.get("explanation", {})
        pref = expl.get("preference", {})
        jit = expl.get("jitter", {})

        target_nominal = float(pref.get("target_nominal", 0.0))

        # Prefer env-provided effective target, else preference dict, else nominal.
        target_effective = info.get("effective_target", pref.get("target_effective", target_nominal))
        target_effective = float(target_effective)

        # Prefer preference tolerance, else fall back to env.hidden tolerance.
        tolerance = pref.get("tolerance", getattr(env.hidden, "tolerance", 0.01))
        tolerance = float(tolerance)

        sigma_obs = float(jit.get("sigma_obs", getattr(env.jitter, "sigma_obs", 0.0)))

        return TinyStrawState(
            x=float(obs_raw[0]),
            sigma_obs=sigma_obs,
            target_nominal=target_nominal,
            target_effective=target_effective,
            tolerance=tolerance,
        )

    obs_raw, info = env.reset()
    obs = make_state(obs_raw, info, env)

    policy = None
    unsafe_steps = 0
    errors: List[float] = []

    min_m_nose = float("inf")
    min_m_eye = float("inf")
    min_m_depth = float("inf")

    done = False
    steps = 0
    last_info = info

    # CBTL mode flags (faithful to repo)
    cbtl_is_train = False
    cbtl_exclude_personal = False
    cbtl_objective = "exploit"

    if mode == Mode.CBTL_FRAMEWORK:
        cbtl_is_train = episode < cbtl_train_episodes
        if cbtl_is_train:
            gen.train()
            cbtl_exclude_personal = True
            cbtl_objective = "entropy"
        else:
            gen.eval()
            cbtl_exclude_personal = False
            cbtl_objective = "exploit"

    while not done and steps < num_steps_cap:
        need_replan = (policy is None) or (replan_every > 0 and steps % replan_every == 0)

        if need_replan:
            speed_cap = None

            if mode == Mode.CBTL_FRAMEWORK:
                # Faithful baseline: no robust inflation; exploration excludes personal constraints.
                gen.set_robust_k(0.0)

                # Debug probes (optional)
                probe_samples = 80
                entropies: List[float] = []
                probs: List[float] = []
                for _ in range(probe_samples):
                    pos = float(plan_rng.normal(loc=float(obs.target_nominal), scale=0.01))
                    p = float(gen.comfort_prob(obs, pos))
                    probs.append(p)
                    entropies.append(bernoulli_entropy_bits(p))
                median_h = float(np.median(entropies))
                median_p = float(np.median(probs))
                frac_hi = float(np.mean(np.array(entropies) >= 0.75))

                if cbtl_objective == "entropy":
                    cbtl_plans_entropy += 1
                else:
                    cbtl_plans_exploit += 1

                print(
                    f"[CBTL][plan] seed={base_seed} ep={episode} step={steps:02d} "
                    f"phase={'train' if cbtl_is_train else 'eval'} "
                    f"exclude_personal={cbtl_exclude_personal} "
                    f"objective={cbtl_objective} medianH={median_h:.3f} "
                    f"medianP={median_p:.3f} fracHiH={frac_hi:.2f}"
                )

                vars_, solution = solve_tiny_csp_mc(
                    gen,
                    obs,
                    rng=plan_rng,
                    num_samples=600,
                    speed_cap=speed_cap,
                    objective=cbtl_objective,
                    exclude_personal_constraints=cbtl_exclude_personal,
                )

            elif mode == Mode.MY_FRAMEWORK:
                vars_tmp, sol_tmp = solve_tiny_csp_mc(
                    gen,
                    obs,
                    rng=plan_rng,
                    num_samples=300,
                    objective="exploit",
                    exclude_personal_constraints=False,
                )
                pos_tmp = float(sol_tmp[vars_tmp[0]])
                conf = float(gen.comfort_prob(obs, pos_tmp))

                gen.set_robust_k(robust_k_from_conf(conf))
                if conf < 0.6:
                    speed_cap = (0.1, 0.4)

                vars_, solution = solve_tiny_csp_mc(
                    gen,
                    obs,
                    rng=plan_rng,
                    num_samples=600,
                    speed_cap=speed_cap,
                    objective="exploit",
                    exclude_personal_constraints=False,
                )

            else:
                # NO_PERSONALIZATION and UNFILTERED_LLM: plain exploit, no robust inflation
                gen.set_robust_k(0.0)
                vars_, solution = solve_tiny_csp_mc(
                    gen,
                    obs,
                    rng=plan_rng,
                    num_samples=600,
                    speed_cap=None,
                    objective="exploit",
                    exclude_personal_constraints=False,
                )

            policy = gen._generate_policy(obs, vars_)
            policy.reset(solution)

        # ---- Act ----
        act = policy.step(obs)
        gym_action = (
            act[0],
            np.array([act[1]], dtype=np.float32) if act[0] == 0 else np.array([0.0], dtype=np.float32),
        )

        next_obs_raw, _, env_done, _, info = env.step(gym_action)
        last_info = info

        if bool(info.get("unsafe", False)):
            unsafe_steps += 1

        x_true = float(info["x_true"])
        eff_target = float(info["effective_target"])
        errors.append(abs(x_true - eff_target))

        tight = info["explanation"]["tightness"]
        min_m_nose = min(min_m_nose, float(tight["margin_to_nose"]))
        min_m_eye = min(min_m_eye, float(tight["margin_to_eye"]))
        min_m_depth = min(min_m_depth, float(tight["margin_to_depth"]))

        # ---- Online learning ----
        gen.observe_transition(
            obs,
            act,
            make_state(next_obs_raw, info, env),
            env_done,
            info,
        )

        if mode == Mode.CBTL_FRAMEWORK:
            cbtl_transition_count += 1

        obs = make_state(next_obs_raw, info, env)

        done = bool(env_done) or bool(policy.check_termination(obs))
        steps += 1

    # ---- End-of-episode preference parameter updates ----
    if mode == Mode.NO_PERSONALIZATION:
        env.hidden = fixed_hidden

    elif mode == Mode.UNFILTERED_LLM:
        prompt = build_update_prompt(env)
        raw = _query_llm(prompt, stats=llm_stats)
        try:
            suggestion = json.loads(raw)
        except Exception:
            suggestion = {}
        env.hidden = apply_llm_update_unfiltered(suggestion, env)

    elif mode == Mode.CBTL_FRAMEWORK:
        # CBTL baseline: NO LLM parameter update
        pass

    elif mode == Mode.MY_FRAMEWORK:
        prompt = build_update_prompt(env)
        raw = _query_llm(prompt, stats=llm_stats)
        try:
            suggestion = json.loads(raw)
        except Exception:
            suggestion = {}
        ok, new_hidden_or_msg = verify_llm_update_safe(suggestion, env)
        if ok:
            env.hidden = new_hidden_or_msg

    done_satisfaction = float(last_info.get("user_satisfaction", 0.0))
    final_pref_score = float(last_info.get("pref_score", 0.0))

    final_true_error = float(errors[-1]) if errors else 0.0
    avg_true_error = float(np.mean(errors)) if errors else 0.0

    if mode == Mode.CBTL_FRAMEWORK:
        print(
            f"[CBTL][summary] seed={base_seed} ep={episode} "
            f"phase={'train' if cbtl_is_train else 'eval'} "
            f"plans_entropy={cbtl_plans_entropy} plans_exploit={cbtl_plans_exploit} "
            f"transitions={cbtl_transition_count} steps={steps} "
            f"final_err={final_true_error:.4f} pref_score={final_pref_score:.3f}"
        )
        try:
            print("[CBTL][metrics]", gen.get_metrics())
        except Exception as e:
            print("[CBTL][metrics] unavailable:", repr(e))

    result = EpisodeResult(
        mode=str(mode.value),
        seed=int(base_seed),
        episode=int(episode),
        steps=int(steps),
        unsafe_steps=int(unsafe_steps),
        done_satisfaction=done_satisfaction,
        final_pref_score=final_pref_score,
        final_true_error=final_true_error,
        avg_true_error=avg_true_error,
        min_margin_to_nose=float(min_m_nose if np.isfinite(min_m_nose) else 0.0),
        min_margin_to_eye=float(min_m_eye if np.isfinite(min_m_eye) else 0.0),
        min_margin_to_depth=float(min_m_depth if np.isfinite(min_m_depth) else 0.0),
        desired_offset=float(env.hidden.desired_mouth_offset),
        tolerance=float(env.hidden.tolerance),
        llm_calls=int(llm_stats.num_calls),
    )
    return result, replace(env.hidden)


# =========================
# Main ablation loop
# =========================
def main():
    jitter = JitterSpec(sigma_obs=0.002, sigma_head=0.001, rho_head=0.8, head_clip=0.008)
    project_actions = True

    seeds = [0]
    num_episodes = 5

    # CBTL: first chunk of episodes is "train/explore", rest is "eval/exploit"
    cbtl_train_episodes = max(1, num_episodes // 2)

    modes = [
        Mode.NO_PERSONALIZATION,
        Mode.UNFILTERED_LLM,
        Mode.CBTL_FRAMEWORK,
        Mode.MY_FRAMEWORK,
    ]

    results: List[EpisodeResult] = []

    gens: Dict[Tuple[Mode, int], TinyStrawCSPGenerator] = {}
    hidden_mem: Dict[Tuple[Mode, int], StrawHiddenSpec] = {}
    plan_rngs: Dict[Tuple[Mode, int], np.random.Generator] = {}

    for mode in modes:
        for s in seeds:
            # IMPORTANT: baseline generators should not get "margin objective"
            if mode in (Mode.NO_PERSONALIZATION, Mode.UNFILTERED_LLM, Mode.CBTL_FRAMEWORK):
                gens[(mode, s)] = TinyStrawCSPGenerator(
                    seed=s,
                    robust_k=0.0,
                    w_margin=0.0,   # baseline: turn off safety-margin objective
                )
            else:
                gens[(mode, s)] = TinyStrawCSPGenerator(
                    seed=s,
                    robust_k=0.0,
                    w_margin=0.5,   # my method keeps it 
                )

            if mode == Mode.NO_PERSONALIZATION:
                gens[(mode, s)]._disable_learning = True

            tmp_env = TinyStrawEnv(seed=s * 1000 + 999, jitter=jitter, project_actions=project_actions)
            tmp_env.reset()
            hidden_mem[(mode, s)] = replace(tmp_env.hidden)

            plan_rngs[(mode, s)] = np.random.default_rng(
                10_000 * s + 123 + (0 if mode != Mode.MY_FRAMEWORK else 7)
            )

    for mode in modes:
        for s in seeds:
            gen = gens[(mode, s)]
            plan_rng = plan_rngs[(mode, s)]

            for ep in range(num_episodes):
                r, new_hidden = run_episode(
                    mode=mode,
                    base_seed=s,
                    episode=ep,
                    jitter=jitter,
                    project_actions=project_actions,
                    gen=gen,
                    hidden_in=hidden_mem[(mode, s)],
                    plan_rng=plan_rng,
                    cbtl_train_episodes=cbtl_train_episodes,
                    num_steps_cap=60,
                    replan_every=1 if mode in (Mode.CBTL_FRAMEWORK, Mode.MY_FRAMEWORK) else 5,
                )
                hidden_mem[(mode, s)] = new_hidden
                results.append(r)
                print(r)

    def agg(mode_name: str) -> None:
        rows = [x for x in results if x.mode == mode_name]
        if not rows:
            return
        print("\n==============================")
        print("MODE:", mode_name)
        print("episodes:", len(rows))
        print("avg unsafe_steps:", np.mean([x.unsafe_steps for x in rows]))
        print("avg done_satisfaction:", np.mean([x.done_satisfaction for x in rows]))
        print("avg final_true_error:", np.mean([x.final_true_error for x in rows]))
        print("avg avg_true_error:", np.mean([x.avg_true_error for x in rows]))
        print("avg min_margin_to_nose:", np.mean([x.min_margin_to_nose for x in rows]))
        print("avg min_margin_to_eye:", np.mean([x.min_margin_to_eye for x in rows]))
        print("avg min_margin_to_depth:", np.mean([x.min_margin_to_depth for x in rows]))
        print("avg llm_calls:", np.mean([x.llm_calls for x in rows]))
        print("==============================\n")

    for m in modes:
        agg(m.value)


if __name__ == "__main__":
    main()


