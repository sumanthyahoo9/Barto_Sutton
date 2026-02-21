"""
Optimistic Initialization
Better Convergence
"""
import matplotlib.pyplot as plt
from experiments.run_experiments import run_multiple_experiments
from multiarm_bandits.epsilon_greedy import EpsilonGreedy

def test_optimistic_initialization():
    """Compare optimistic initialization vs realistic ε-greedy"""
    print("Generating Optimistic initial values...")
    
    num_runs = 2000
    steps = 1000
    
    methods = [
        ("Realistic, ε-greedy", lambda: EpsilonGreedy(k=10, epsilon=0.1, alpha=0.1, initial_value=0.0)),
        ("Optimistic, greedy", lambda: EpsilonGreedy(k=10, epsilon=0.0, alpha=0.1, initial_value=5.0)),
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, agent_fn in methods:
        _, pct_optimal = run_multiple_experiments(agent_fn, num_runs, steps)
        plt.plot(pct_optimal, label=name, linewidth=2)
    
    plt.ylabel('% Optimal Action', fontsize=12)
    plt.xlabel('Steps', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.title('Optimistic Initial Values Encourage Exploration', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('optimistic_initialization.png', dpi=150)
    print("✓ Saved as optimistic_initialization.png")
    plt.show()

if __name__ == "__main__":
    test_optimistic_initialization()
