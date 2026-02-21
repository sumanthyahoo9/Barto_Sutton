"""
The bandit environment
"""
import numpy as np
np.random.seed(42)

class BanditEnvironment:
    """
    10-armed testbed bandit environment
    """
    def __init__(self, k: int = 10, stationary: bool = True, mean_shift: float = 0.0):
        """
        Args:
            k: Number of arms
            stationary: If True, true values don't change
            mean_shift: Shift mean of true values (e.g., +4 for gradient bandit test)
        """
        self.k = k
        self.stationary = stationary
        self.mean_shift = mean_shift
        
        # Initialize true action values q*(a) ~ N(mean_shift, 1)
        self.q_true = np.random.randn(k) + mean_shift
        self.optimal_action = np.argmax(self.q_true)
        
    def step(self, action: int) -> float:
        """Take action and return reward ~ N(q*(a), 1)"""
        reward = np.random.randn() + self.q_true[action]
        
        # For nonstationary problems, take random walk
        if not self.stationary:
            self.q_true += np.random.randn(self.k) * 0.01
            self.optimal_action = np.argmax(self.q_true)
            
        return reward
    
    def get_optimal_action(self) -> int:
        """
        Return the optimal action
        """
        return self.optimal_action