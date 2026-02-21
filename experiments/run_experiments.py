"""
Run multiple experiments
"""
from typing import Tuple
import numpy as np
from multiarm_bandits.bandit_environments import BanditEnvironment
from experiments.run_single_experiment import run_experiment

def run_multiple_experiments(agent_fn,
                             num_runs: int,
                             steps: int,
                             **env_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run multiple experiments and average the results
    Args:
        agent_fn: Function that creates agent
        num_runs: Number of independent runs
        steps: Steps per run
        env_kwargs: Arguments for BanditEnvironment
    Returns:
        avg_rewards: Average reward at each step
        pct_optimal: Percentage optimal action at each step
    """
    all_rewards = np.zeros((num_runs, steps))
    all_optimal = np.zeros((num_runs, steps))
    for run in range(num_runs):
        env = BanditEnvironment(**env_kwargs)
        agent = agent_fn()
        rewards, optimal = run_experiment(agent, env, steps)
        all_rewards[run] = rewards
        all_optimal[run] = optimal
    return all_rewards.mean(axis=0), all_optimal.mean(axis=0) * 100