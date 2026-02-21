"""
Run a single experiment
"""
from typing import Tuple
import numpy as np

def run_experiment(agent, env, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single experiment
    Return:
        rewards at each step
        optimal)actions: 1 if optimal_action taken, else 0
    """
    rewards = np.zeros(steps)
    optimal_actions = np.zeros(steps)
    for t in range(steps):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)
        rewards[t] = reward
        optimal_actions[t] = 1 if action == env.get_optimal_action() else 0
    return rewards, optimal_actions
