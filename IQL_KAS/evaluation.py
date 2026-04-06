from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def compute_jerk(action_history):
    if len(action_history) < 3:
        return 0.0

    actions = np.asarray(action_history, dtype=np.float64)
    jerks = np.linalg.norm(
        actions[2:] - 2.0 * actions[1:-1] + actions[:-2],
        axis=1,
    )
    return float(np.mean(jerks))


def evaluate(
    agent: nn.Module,
    env: gym.Env,
    num_episodes: int,
    apply_kalman: bool = False,
    dt: float = 0.01,
) -> Dict[str, float]:
    stats = {"return": [], "length": []}
    jerks = []

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        if hasattr(agent, "reset_filters"):
            agent.reset_filters()

        action_history = []

        while not done:
            action = agent.sample_actions(
                observation,
                temperature=0.0,
                apply_kalman=apply_kalman,
                dt=dt,
            )
            action_history.append(np.asarray(action, dtype=np.float64).copy())
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info["episode"][k])

        if len(action_history) >= 3:
            jerks.append(compute_jerk(action_history))

    for k, v in stats.items():
        stats[k] = float(np.mean(v))

    stats["jerk"] = float(np.mean(jerks)) if len(jerks) > 0 else 0.0

    return stats