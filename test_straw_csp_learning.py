import numpy as np
from tiny_straw_csp import TinyStrawState, TinyStrawCSPGenerator


def main():
    gen = TinyStrawCSPGenerator(seed=0)

    target_nominal = 0.01

    def make_state(x_obs: float) -> TinyStrawState:
        return TinyStrawState(x=float(x_obs), sigma_obs=0.0, target_nominal=float(target_nominal))

    def p_comfort(pos: float) -> float:
        return gen.comfort_prob(make_state(target_nominal), float(pos))

    # BEFORE learning
    print("=== BEFORE LEARNING ===")
    print("p(near) =", p_comfort(target_nominal))
    print("p(far)  =", p_comfort(target_nominal - 0.10))

    # Provide transitions
    positives = [target_nominal - 0.002, target_nominal + 0.003]
    negatives = [-0.05, -0.02, 0.04]

    for x in positives:
        obs = make_state(x)
        act = (1, 0.0)
        info = {
            "user_satisfaction": 1.0,
            "x_true": float(x),
            "effective_target": float(target_nominal),
            "explanation": {
                "preference": {"target_nominal": float(target_nominal), "target_effective": float(target_nominal), "tolerance": 0.004},
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, obs, done=True, info=info)

    for x in negatives:
        obs = make_state(x)
        act = (1, 0.0)
        info = {
            "user_satisfaction": 0.0,
            "x_true": float(x),
            "effective_target": float(target_nominal),
            "explanation": {
                "preference": {"target_nominal": float(target_nominal), "target_effective": float(target_nominal), "tolerance": 0.004},
                "jitter": {"sigma_obs": 0.0},
            },
        }
        gen.observe_transition(obs, act, obs, done=True, info=info)

    print("\nClassifier after learning:", gen._pref_gen._classifier)

    # AFTER learning
    print("\n=== AFTER LEARNING ===")
    p_near = p_comfort(target_nominal)
    p_far = p_comfort(target_nominal - 0.10)
    print("p(near) =", p_near)
    print("p(far)  =", p_far)

    assert p_near > p_far, "Expected higher comfort probability near the nominal target after learning."
    assert (p_near - p_far) > 0.1, "Separation is too weak; warmstart/labels may be insufficient."

    print("\n test_straw_csp_learning passed")


if __name__ == "__main__":
    main()




