"""
Compare the ε-greedy approach with UCB
"""
import matplotlib.pyplot as plt
from multiarm_bandits.epsilon_greedy import EpsilonGreedy
from multiarm_bandits.upper_confidence_bound import UCB
from experiments.run_experiments import run_multiple_experiments

def epsilon_greedy_vs_ucb():
    """
    Compare ε-greedy with UCB
    """
    print("Comparing ε-greedy with UCB")
    num_runs, steps = 2000, 1000
    methods = [
        ("ε-greedy ε=0.1", lambda: EpsilonGreedy(k=10, epsilon=0.1)),
        ("UCB c=2", lambda: UCB(k=10, c=2)),
    ]
    plt.figure(figsize=(10, 6))
    for name, agent_fn in methods:
        avg_rewards, _ = run_multiple_experiments(agent_fn, num_runs, steps)
        plt.plot(avg_rewards, label=name, linewidth=2)
    plt.ylabel('Average Reward', fontsize=12)
    plt.xlabel('Steps', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.title('UCB Performs Better Than ε-Greedy', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('ucb_vs_ε_greedy.png', dpi=150)
    print("✓ Saved as ucb_vs_ε_greedy.png")
    plt.show()

if __name__ == "__main__":
    epsilon_greedy_vs_ucb()