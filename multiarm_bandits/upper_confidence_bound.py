"""
Upper Confidence Bound method of exploration
"""
import numpy as np

class UCB:
    """
    Upper Confidence Bound
    """
    def __init__(self, k: int, c: float):
        """
        k: Number of arms
        c: Exploration parameter
        """
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0
    
    def select_action(self) -> int:
        """
        Select action using UCB
        """
        self.t += 1
        # Select untried actions first
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return np.random.choice(untried)
        # UCB: Q(a) + c * sqrt(ln(t)/N(a))
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t)/self.N)
        return np.argmax(ucb_values)
    
    def update(self, action: int, reward: float):
        """
        Update the Q estimates
        """
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]