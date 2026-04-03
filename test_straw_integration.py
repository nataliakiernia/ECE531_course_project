import numpy as np

from tiny_straw_env import TinyStrawEnv, JitterSpec
from tiny_straw_csp import TinyStrawState, TinyStrawCSPGenerator


def _get_sigma_obs(info, default=0.0) -> float:
    try:
        return float(info["explanation"]["jitter"]["sigma_obs"])
    except Exception:
        return float(default)


def _get_target_nominal(info, default=0.0) -> float:
    try:
        return float(info["explanation"]["preference"]["target_nominal"])
    except Exception:
        # older env fallback
        try:
            return float(info["explanation"]["preference"]["target"])
        except Exception:
            return float(default)


def make_state_from_env(obs_raw, info) -> TinyStrawState:
    return TinyStrawState(
        x=float(obs_raw[0]),  # observed position
        sigma_obs=_get_sigma_obs(info, default=0.0),
        target_nominal=_get_target_nominal(info, default=0.0),
    )


def warmstart_preferences(gen: TinyStrawCSPGenerator, target: float = 0.01) -> None:
    """Warm-start distance preference model so it doesn't start blind."""

    def make_state(x_obs: float) -> TinyStrawState:
        return TinyStrawState(x=float(x_obs), sigma_obs=0.0, target_nominal=float(target))

    # Positive examples near the target
    for x in [target - 0.002, target + 0.003]:
        obs = make_state(x_obs=x)
        next_obs = make_state(x_obs=x)
        act = (1, 0.0)

        info = {
            "user_satisfaction": 1.0,
            # let learner use true/effective values (even though here they match)
            "x_true": float(x),
            "effective_target": float(target),
            "explanation": {
                "preference": {
                    "target_nominal": float(target),
                    "target_effective": float(target),
                    "tolerance": 0.004,
                },
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, next_obs, done=True, info=info)

    # Negative examples farther away
    for x in [-0.05, -0.02, 0.04, 0.015]:
        obs = make_state(x_obs=x)
        next_obs = make_state(x_obs=x)
        act = (1, 0.0)

        info = {
            "user_satisfaction": 0.0,
            "x_true": float(x),
            "effective_target": float(target),
            "explanation": {
                "preference": {
                    "target_nominal": float(target),
                    "target_effective": float(target),
                    "tolerance": 0.004,
                },
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, next_obs, done=True, info=info)


def solve_tiny_csp(
    gen: TinyStrawCSPGenerator,
    obs: TinyStrawState,
    num_samples: int = 200,
):
    """
    Simple Monte Carlo CSP solver:
    - Samples (position, speed)
    - Scores using learned preference + speed
    """
    vars_, _ = gen._generate_variables(obs)
    position_var, speed_var = vars_

    rng = np.random.default_rng(gen._seed)

    best_score = -np.inf
    best_solution = None

    # Score distance to CURRENT target estimate (not to 0.0)
    target_est = float(obs.target_nominal)

    # If you're using robust safety constraints in the generator, pull them once
    nonpersonal_constraints = gen._generate_nonpersonal_constraints(obs, [position_var, speed_var])

    for _ in range(num_samples):
        position = float(rng.normal(loc=target_est, scale=0.01))
        speed = float(rng.uniform(0.1, 1.0))

        # Optional but recommended: reject if violates hard safety constraint
        hard_ok = True
        for c in nonpersonal_constraints:
            fn = getattr(c, "_fn", None)
            threshold = getattr(c, "threshold", -1e6)
            if fn is not None:
                lp = float(fn(np.float_(position)))
                if lp < float(threshold):
                    hard_ok = False
                    break
        if not hard_ok:
            continue

        clf = gen._pref_gen._classifier
        if clf is None:
            logpref = 0.0
        else:
            dist = abs(position - target_est)
            feats = np.array([dist], dtype=float)
            prob = clf.predict_proba([feats])[0][1]
            logpref = float(np.log(prob + 1e-9))

        score = logpref + 0.1 * speed

        if score > best_score:
            best_score = score
            best_solution = {position_var: position, speed_var: speed}

    return vars_, best_solution


def main():
    # If you want the jittery scenario here, set jitter + robust_k
    env = TinyStrawEnv(
        jitter=JitterSpec(sigma_obs=0.002, sigma_head=0.001, rho_head=0.8, head_clip=0.008),
        project_actions=True,
    )
    gen = TinyStrawCSPGenerator(seed=0, robust_k=2.0)

    warmstart_preferences(gen, target=0.01)

    for episode in range(3):
        print("\n==============================")
        print(f"EPISODE {episode}")
        print("==============================")

        obs_raw, info = env.reset()
        obs = make_state_from_env(obs_raw, info)

        temp_vars, _ = gen._generate_variables(obs)
        constraints = gen._generate_personal_constraints(obs, temp_vars)
        cost = gen._generate_exploit_cost(obs, temp_vars)

        print("Cost name:", cost.name if cost is not None else "None")
        print("Num personal constraints:", len(constraints))

        vars_, solution = solve_tiny_csp(gen, obs, num_samples=300)
        if solution is None:
            print("No feasible CSP solution found (too conservative / too noisy).")
            continue

        print("Chosen CSP solution:", {v.name: solution[v] for v in vars_})

        policy = gen._generate_policy(obs, vars_)
        policy.reset(solution)

        done = False
        step = 0
        while not done and step < 50:
            act = policy.step(obs)
            gym_action = (
                act[0],
                np.array([act[1]], dtype=np.float32) if act[0] == 0 else np.array([0.0], dtype=np.float32),
            )

            next_obs_raw, reward, env_done, truncated, info = env.step(gym_action)
            next_obs = make_state_from_env(next_obs_raw, info)

            print(
                f"step={step:02d} x_obs={obs.x:+.4f} -> x_obs'={next_obs.x:+.4f} "
                f"act={act} sat={info['user_satisfaction']} fb={info['feedback']}"
            )

            gen.observe_transition(obs, act, next_obs, env_done, info)

            obs = next_obs
            done = env_done or policy.check_termination(obs)
            step += 1

        print("Episode finished. Last info:", info)


if __name__ == "__main__":
    main()




