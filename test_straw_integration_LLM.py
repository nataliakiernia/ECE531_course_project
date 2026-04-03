# test_straw_integration_LLM.py
import json
from dataclasses import dataclass, replace
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt

from tiny_straw_env import TinyStrawEnv, JitterSpec
from tiny_straw_csp import (
    TinyStrawState,
    TinyStrawCSPGenerator,
    _KeepOutGeometry,
    _min_safety_margin_robust,
)

client = OpenAI()


# ---------------------------------------------------------------------
# Small helper: build TinyStrawState from env's info dict + observation
# ---------------------------------------------------------------------
def state_from_env(obs_raw: np.ndarray, info: Dict[str, Any]) -> TinyStrawState:
    sigma_obs = float(info["explanation"]["jitter"]["sigma_obs"])
    target_nominal = float(info["explanation"]["preference"]["target_nominal"])
    # Prefer explicit env field, fallback to explanation
    target_effective = float(
        info.get("effective_target", info["explanation"]["preference"]["target_effective"])
    )
    tolerance = float(info["explanation"]["preference"]["tolerance"])

    return TinyStrawState(
        x=float(obs_raw[0]),
        sigma_obs=sigma_obs,
        target_nominal=target_nominal,
        target_effective=target_effective,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------
# Warm-start preference model
# ---------------------------------------------------------------------
def warmstart_preferences(gen: TinyStrawCSPGenerator, target_nominal: float = 0.01) -> None:
    """
    Warm-start using synthetic "done" transitions.
    We provide the fields the updated learner may look for (x_true, effective_target, explanation...).
    """

    def make_state(x_obs: float) -> TinyStrawState:
        return TinyStrawState(
            x=float(x_obs),
            sigma_obs=0.0,
            target_nominal=float(target_nominal),
            target_effective=float(target_nominal),
            tolerance=0.004,
        )

    # Positive (comfortable) near target
    for x in [target_nominal - 0.002, target_nominal + 0.003]:
        obs = make_state(x)
        act = (1, 0.0)
        info = {
            "user_satisfaction": 1.0,
            "x_true": float(x),
            "effective_target": float(target_nominal),
            "pref_score": 1.0,
            "feedback": "Perfect sip.",
            "unsafe": False,
            "explanation": {
                "preference": {
                    "target_nominal": float(target_nominal),
                    "target_effective": float(target_nominal),
                    "tolerance": 0.004,
                },
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, obs, done=True, info=info)

    # Negative (uncomfortable) farther away
    for x in [-0.05, -0.02, 0.04, 0.015]:
        obs = make_state(x)
        act = (1, 0.0)
        info = {
            "user_satisfaction": 0.0,
            "x_true": float(x),
            "effective_target": float(target_nominal),
            "pref_score": 0.0,
            "feedback": "Not comfortable yet.",
            "unsafe": False,
            "explanation": {
                "preference": {
                    "target_nominal": float(target_nominal),
                    "target_effective": float(target_nominal),
                    "tolerance": 0.004,
                },
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, obs, done=True, info=info)


# ---------------------------------------------------------------------
# Confidence metric (CBTL-ish): separation between near and far comfort probabilities
# ---------------------------------------------------------------------
def compute_confidence(gen: TinyStrawCSPGenerator, obs: TinyStrawState) -> float:
    """
    Confidence ~ how sharply the current model distinguishes near vs far.
    Returns in [0,1] (heuristic).
    """
    p_near = gen.comfort_prob(obs, float(obs.target_effective))
    far = float(obs.target_effective - 0.08)  # 8 cm away
    p_far = gen.comfort_prob(obs, far)
    sep = abs(p_near - p_far)
    return float(np.clip(sep, 0.0, 1.0))


def choose_robust_k_from_conf(conf: float) -> float:
    """Map confidence -> robust_k. Low confidence => more conservative."""
    if conf < 0.20:
        return 3.0
    if conf < 0.40:
        return 2.5
    if conf < 0.60:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------
# Monte Carlo solver with optional low-confidence safe exploration
# ---------------------------------------------------------------------
def solve_tiny_csp_mc(
    gen: TinyStrawCSPGenerator,
    obs: TinyStrawState,
    num_samples: int = 400,
    do_safe_explore: bool = True,
    conf: float = 1.0,
) -> Tuple[List[Any], Dict[Any, float]]:
    """
    MC solver:
      score = log p(comfortable) + 0.1*speed + explore_bonus (optional)

    With updated CSP, comfort_prob is centered on target_effective, so we
    sample around target_effective (not nominal).
    """
    vars_, _ = gen._generate_variables(obs)
    position_var, speed_var = vars_

    rng = np.random.default_rng(gen._seed)

    best_score = -np.inf
    best_solution: Optional[Dict[Any, float]] = None

    # Speed cap under low confidence (safer baseline)
    if conf < 0.40:
        speed_lo, speed_hi = 0.1, 0.4
    else:
        speed_lo, speed_hi = 0.1, 1.0

    center = float(obs.target_effective)

    for _ in range(num_samples):
        position = float(rng.normal(loc=center, scale=0.02))
        speed = float(rng.uniform(speed_lo, speed_hi))

        p = gen.comfort_prob(obs, position)
        logpref = float(np.log(p + 1e-9))

        explore_bonus = 0.0
        if do_safe_explore and conf < 0.40:
            explore_bonus = float(-abs(p - 0.5))

        score = logpref + 0.1 * speed + 0.2 * explore_bonus

        if score > best_score:
            best_score = score
            best_solution = {position_var: float(position), speed_var: float(speed)}

    assert best_solution is not None
    return vars_, best_solution


# ---------------------------------------------------------------------
# Paper plots
# ---------------------------------------------------------------------
def make_paper_plots(episode_logs: List[Dict[str, Any]], out_dir: str = "paper_figures") -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(episode_logs)
    if df.empty:
        print("[plots] No episode logs to plot.")
        return

    df["safe_episode"] = (df["safety_violations"] == 0).astype(int)

    # FIGURE 1: Safety vs Personalization
    plt.figure()
    sc = plt.scatter(
        df["final_true_error"],
        df["planned_min_margin"],
        c=df["confidence"],
        s=45,
    )
    plt.xlabel("Final true error |x_true - target_effective| (m)")
    plt.ylabel("Planned robust min safety margin (m)")
    plt.title("Safety vs Personalization Tradeoff")
    cbar = plt.colorbar(sc)
    cbar.set_label("Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_tradeoff_error_vs_margin.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig1_tradeoff_error_vs_margin.pdf"))
    plt.close()

    # FIGURE 2: Confidence -> robust_k
    plt.figure()
    plt.scatter(df["confidence"], df["robust_k_used"], s=45)
    plt.xlabel("Confidence (near-vs-far comfort separation)")
    plt.ylabel("robust_k used")
    plt.title("CBTL-ish Gating: Confidence → Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_confidence_vs_robustk.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig2_confidence_vs_robustk.pdf"))
    plt.close()

    # FIGURE 3: Preference score vs error
    plt.figure()
    plt.scatter(df["final_true_error"], df["final_pref_score"], s=45)
    plt.xlabel("Final true error |x_true - target_effective| (m)")
    plt.ylabel("Final preference score (pref_score)")
    plt.title("Preference Satisfaction vs Final Error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_prefscore_vs_error.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig3_prefscore_vs_error.pdf"))
    plt.close()

    # FIGURE 4: Safety violations per episode
    plt.figure()
    plt.bar(df["episode"], df["safety_violations"])
    plt.xlabel("Episode")
    plt.ylabel("Safety violations (count)")
    plt.title("Safety Violations per Episode")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_safety_violations.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig4_safety_violations.pdf"))
    plt.close()

    # FIGURE 5: Summary bars (mean ± 95% CI)
    def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        if len(x) <= 1:
            return float(np.mean(x)), 0.0
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        ci = 1.96 * se
        return m, ci

    metrics = [
        ("final_true_error", "Final true error (m)"),
        ("final_pref_score", "Final pref score"),
        ("planned_min_margin", "Planned min margin (m)"),
        ("safety_violations", "Safety violations"),
    ]

    means: List[float] = []
    cis: List[float] = []
    labels: List[str] = []
    for col, lab in metrics:
        m, ci = mean_ci95(df[col].to_numpy())
        means.append(m)
        cis.append(ci)
        labels.append(lab)

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=cis, capsize=6)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Summary Metrics (mean ± 95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_summary_bars.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig5_summary_bars.pdf"))
    plt.close()

    df.to_csv(os.path.join(out_dir, "episode_logs.csv"), index=False)
    print(f"[plots] wrote {out_dir}/episode_logs.csv and 5 figures (.png + .pdf)")


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    gen = TinyStrawCSPGenerator(seed=0, robust_k=2.0, w_speed=1.0, w_margin=0.5)

    env = TinyStrawEnv(
        jitter=JitterSpec(sigma_obs=0.002, sigma_head=0.001, rho_head=0.8, head_clip=0.008),
        project_actions=True,
        update_head_each_step=True,
        update_head_on_done=False,
        sample_head_on_reset=False,
        add_obs_noise=True,
        terminate_on_unsafe=False,
    )

    warmstart_preferences(gen, target_nominal=0.01)

    NUM_EPISODES = 20
    episode_logs: List[Dict[str, Any]] = []

    geom = _KeepOutGeometry()

    for episode in range(NUM_EPISODES):
        print("\n==============================")
        print(f"EPISODE {episode}")
        print("==============================")

        obs_raw, info = env.reset()
        obs = state_from_env(obs_raw, info)

        # CBTL-ish confidence gating
        conf = compute_confidence(gen, obs)
        robust_k = choose_robust_k_from_conf(conf)
        gen.set_robust_k(robust_k)
        print(f"[CONF] conf={conf:.3f} -> robust_k={robust_k:.2f}")

        # Solve CSP via MC
        vars_, solution = solve_tiny_csp_mc(gen, obs, num_samples=500, do_safe_explore=True, conf=conf)
        position_var, speed_var = vars_
        chosen_pos = float(solution[position_var])
        chosen_speed = float(solution[speed_var])

        robust_margin = float(gen.get_robust_k() * max(0.0, obs.sigma_obs))
        min_margin = _min_safety_margin_robust(chosen_pos, geom, robust_margin)

        print("Chosen CSP solution:", {v.name: solution[v] for v in vars_})
        print(f"[PLAN] min_margin={min_margin:.4f} m, speed={chosen_speed:.3f}")

        policy = gen._generate_policy(obs, vars_)
        policy.reset(solution)

        offset_values: List[float] = []
        distance_errors: List[float] = []
        satisfaction_values: List[float] = []
        tolerance_values: List[float] = []
        safety_violations = 0

        done = False
        step = 0
        last_info = info

        while (not done) and (step < 60):
            act = policy.step(obs)  # (op, delta)

            gym_action = (
                int(act[0]),
                np.array([float(act[1])], dtype=np.float32) if int(act[0]) == 0 else np.array([0.0], dtype=np.float32),
            )

            next_obs_raw, _, env_done, _, info = env.step(gym_action)
            next_obs = state_from_env(next_obs_raw, info)

            print(
                f"step={step:02d} x_obs={obs.x:+.4f} -> x_obs'={next_obs.x:+.4f} "
                f"act={act} sat={info['user_satisfaction']} fb={info['feedback']}"
            )

            gen.observe_transition(obs, act, next_obs, env_done, info)

            x_true = float(info["x_true"])
            target_effective = float(
                info.get("effective_target", info["explanation"]["preference"]["target_effective"])
            )
            tol = float(info["explanation"]["preference"]["tolerance"])
            soft_sat = float(info["pref_score"])

            offset = x_true - target_effective
            dist_err = abs(offset)

            offset_values.append(float(offset))
            distance_errors.append(float(dist_err))
            satisfaction_values.append(float(soft_sat))
            tolerance_values.append(float(tol))

            if bool(info.get("unsafe", False)):
                safety_violations += 1

            last_info = info
            obs = next_obs
            done = bool(env_done) or bool(policy.check_termination(obs))
            step += 1

        print("Episode finished. Last info:", last_info)

        ep_log = {
            "episode": int(episode),
            "confidence": float(conf),
            "robust_k_used": float(robust_k),
            "planned_position": float(chosen_pos),
            "planned_speed": float(chosen_speed),
            "planned_min_margin": float(min_margin),
            "avg_true_offset": float(np.mean(offset_values)) if offset_values else 0.0,
            "final_true_offset": float(offset_values[-1]) if offset_values else 0.0,
            "avg_true_error": float(np.mean(distance_errors)) if distance_errors else 0.0,
            "final_true_error": float(distance_errors[-1]) if distance_errors else 0.0,
            "avg_pref_score": float(np.mean(satisfaction_values)) if satisfaction_values else 0.0,
            "final_pref_score": float(satisfaction_values[-1]) if satisfaction_values else 0.0,
            "avg_tolerance": float(np.mean(tolerance_values)) if tolerance_values else 0.0,
            "final_tolerance": float(tolerance_values[-1]) if tolerance_values else 0.0,
            "safety_violations": int(safety_violations),
        }
        episode_logs.append(ep_log)

        print("\n=== EPISODE LOG ===")
        print(ep_log)

    print("\n=== ALL EPISODE LOGS ===")
    for log in episode_logs:
        print(log)

    # call plots AFTER logs exist 
    make_paper_plots(episode_logs, out_dir="paper_figures")


if __name__ == "__main__":
    main()
