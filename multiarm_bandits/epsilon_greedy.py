"""
Epsilon-greedy exploration technique for Multiarm bandit environment
"""
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiments import run_multiple_experiments

class EpsilonGreedy:
    """
    Setup the Epsilon-greedy action selection
    """
    def __init__(self, k: int, epsilon: float, alpha: float = 0.0, initial_value: float = 0.0):
        """
        Args:
            k: Number of arms
            epsilon: Exploration probability
            alpha: Step size (None = sample average)
            initial_value: Initial Q estimates
        """
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_value = initial_value
        
        self.Q = np.ones(k) * initial_value
        self.N = np.zeros(k)
    
    def select_action(self) -> int:
        """Select action using epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        else:
            return np.argmax(self.Q)  # Exploit (greedy)
    
    def update(self, action: int, reward: float):
        """
        Update the Q-estimates
        """
        self.N[action] += 1
        if self.alpha is None:
            # Sample average: step_size = 1/N(a)
            step_size = 1.0 / self.N[action]
        else:
            step_size = self.alpha
        self.Q[action] += step_size * (reward - self.Q[action])

def test_epsilon_greedy():
    """
    Test the epsilon-greedy method
    """
    print("Generating the greedy vs ε-greedy")
    num_runs, steps = 2000, 1000
    # Test different epsilon values
    methods = [
        ("ε=0 (greedy)", lambda: EpsilonGreedy(k=10, epsilon=0.0)),
        ("ε=0.01", lambda: EpsilonGreedy(k=10, epsilon=0.01)),
        ("ε=0.1", lambda: EpsilonGreedy(k=10, epsilon=0.1)),
    ]
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for name, agent_fn in methods:
        avg_rewards, pct_optimal = run_multiple_experiments(agent_fn, num_runs, steps)
        
        ax1.plot(avg_rewards, label=name)
        ax2.plot(pct_optimal, label=name)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('% Optimal Action', fontsize=12)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_greedy.png', dpi=150)
    print("✓ Saved as epsilon_greedy.png")
    plt.show()

if __name__ == "__main__":
    test_epsilon_greedy()