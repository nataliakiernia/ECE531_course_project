import numpy as np
from tiny_straw_env import TinyStrawEnv

def run_basic_env_test():
    env = TinyStrawEnv()
    obs, info = env.reset()
    print("RESET:")
    print("  x =", float(obs[0]))
    print("  info =", info)

    target = env.keep.mouth_center + env.hidden.desired_mouth_offset
    print("\nTarget mouth position:", target)

    done = False
    step = 0

    while not done and step < 100:
        x = float(obs[0])
        # Move toward target with bounded delta
        delta = np.clip(target - x, -env.max_step, env.max_step)
        action = (0, np.array([delta], dtype=np.float32))

        obs, reward, done, truncated, info = env.step(action)
        x = float(obs[0])

        print(
            f"step={step:02d}  x={x:+.4f}  "
            f"pref_score={info['pref_score']:.3f}  "
            f"user_sat={info['user_satisfaction']}  "
            f"feedback={info['feedback']}"
        )

        # When inside comfort, declare done
        if abs(x - target) <= env.hidden.tolerance and not done:
            print("\nInside comfort window → declaring done\n")
            obs, reward, done, truncated, info = env.step(
                (1, np.array([0.0], dtype=np.float32))
            )

        step += 1

    print("\nFINAL STATE:")
    print("  x =", float(obs[0]))
    print("  info =", info)

    # Simple sanity checks
    assert info["user_satisfaction"] in (-1.0, 0.0, 1.0)
    assert not env._unsafe(float(obs[0])), "Ended in unsafe region!"

if __name__ == "__main__":
    run_basic_env_test()
