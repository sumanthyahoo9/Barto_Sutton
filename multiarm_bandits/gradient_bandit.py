"""
Gradient Bandit
"""
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiments import run_multiple_experiments

class GradientBandit:
    """
    Gradient Bandit with baseline for comparison
    """
    def __init__(self, k: int, alpha: float, use_baseline: bool = True):
        """
        k: number of arms
        alpha: step size
        """
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.H = np.zeros(k)
        self.avg_reward = 0.0
        self.t = 0
    
    def select_action(self) -> int:
        """
        Select the next action using softmax
        """
        exp_h = np.exp(self.H - np.max(self.H)) # Stability
        self.pi = exp_h/np.sum(exp_h)
        # Sample from the probability distribution
        return np.random.choice(self.k, p=self.pi)
    
    def update(self, action: int, reward: float):
        """
        Update preferences
        """
        self.t += 1
        # Update the average reward
        self.avg_reward += (reward - self.avg_reward) / self.t
        # Baseline
        baseline = self.avg_reward if self.use_baseline else 0.0
        # Update preferences
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        # H(a) += α(R - baseline)(𝟙_a - π(a))
        self.H += self.alpha * (reward - baseline) * (one_hot - self.pi)

def test_gradient_bandit():
    """
    Test with different values of alpha
    """
    print("Generating Gradient bandit with baseline...")
    
    num_runs = 2000
    steps = 1000
    
    methods = [
        ("alpha=0.1 with baseline", lambda: GradientBandit(k=10, alpha=0.1, use_baseline=True)),
        ("alpha=0.1 without baseline", lambda: GradientBandit(k=10, alpha=0.1, use_baseline=False)),
        ("alpha=0.4 with baseline", lambda: GradientBandit(k=10, alpha=0.4, use_baseline=True)),
        ("alpha=0.4 without baseline", lambda: GradientBandit(k=10, alpha=0.4, use_baseline=False)),
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, agent_fn in methods:
        # Use mean_shift=4 as in the book
        _, pct_optimal = run_multiple_experiments(agent_fn, num_runs, steps, mean_shift=4.0)
        plt.plot(pct_optimal, label=name, linewidth=2)
    
    plt.ylabel('% Optimal Action', fontsize=12)
    plt.xlabel('Steps', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.title('Gradient Bandit: Baseline Dramatically Improves Performance', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('gradient_bandit.png', dpi=150)
    print("✓ Saved as gradient_bandit.png")
    plt.show()

if __name__ == "__main__":
    test_gradient_bandit()