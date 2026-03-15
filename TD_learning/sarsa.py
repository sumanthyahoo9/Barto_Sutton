"""
SARSA method of Prediction and Control
Prediction: Determining Value functions
Control: Improving a policy
This script deals with On-policy TD-control where we improve the policy we're working with
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ── Environment ──────────────────────────────────────────────────────────────
class GridWorld:
    """
    Define the environment here
    """
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (4, 4)
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        self.action_labels = ['↑', '↓', '←', '→']
    
    def reset(self):
        """
        Reset the environment
        """
        self.state = self.start
        return self.state
    
    def step(self, action):
        """
        One step in an episode
        """
        r, c = self.state
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        else:
            c += 1
        # Clip to grid boundaries
        r = max(0, min(self.size-1, r))
        c = max(0, min(self.size-1, c))
        self.state = (r, c)
        done = (self.state == self.goal)
        reward = 1.0 if done else -0.1 # +1 at the goal, small cost per step
        return self.state, reward, done

class SarsaAgent:
    """
    SARSA agent for on-policy TD-control
    """
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha # Step size
        self.gamma = gamma # discount
        self.epsilon = epsilon
        # Q-table: defaultdict so unseen (s,a) --> 0
        self.Q = defaultdict(float)
    
    def choose_action(self, state):
        """
        epsilon-greedy action selection
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions) # Explore
        q_vals = [self.Q[(state, a)]for a in self.actions]
        return self.actions[np.argmax(q_vals)] # Exploit
    
    def update(self, s, a, r, s_next, a_next, done):
        """
        Core SARSA update: uses actual next action a_next
        """
        q_next = 0.0 if done else self.Q[(s_next, a_next)]
        td_error = r + self.gamma * q_next - self.Q[(s, a)]
        self.Q[(s, a)] += self.alpha * td_error
        return td_error

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_sarsa(n_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    SARSA policy
    """
    env = GridWorld()
    agent = SarsaAgent(env.actions, alpha, gamma, epsilon)
    rewards_per_episode, steps_per_episode = [], []
    for _ in range(n_episodes):
        s = env.reset()
        a = agent.choose_action(s) # Choose the FIRST action (S, A chosen)
        total_reward = 0
        steps = 0
        while True:
            s_next, r, done = env.step(a)
            a_next = agent.choose_action(s_next) # choose next action NOW
            # SARSA update: (S, A, R, S', A')
            agent.update(s, a, r, s_next, a_next, done)
            s, a = s_next, a_next # Advance using the SAME policy
            total_reward += r
            steps += 1
            if done or steps > 200:
                break
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
    return agent, rewards_per_episode, steps_per_episode

def show_numerical_trace():
    """
    Replicates the manual example: 3-state chain, A->B->Goal
    """
    print("="*65)
    print("NUMERICAL TRACE: 3-state chain A-->B-->Goal")
    print("alpha=0.1, gamma=0.9, initial Q=0 everywhere")
    print("="*65)

    # Represent as simple indices: A=0, B=1, Goal=2 (terminal)
    # One action only, move forward
    Q = defaultdict(float)
    alpha, gamma = 0.1, 0.9
    for episode in range(1, 4):
        print(f"\n--- Episode {episode} ---")
        # Episode: A->B(R=0), B->Goal(R=+1)
        transitions = [(0, 0, 0.0, 1, 0, False),   # s,a,r,s',a',done
                       (1, 0, 1.0, 2, 0, True)]
        for (s, a, r, s_next, a_next, done) in transitions:
            q_next = 0.0 if done else Q[(s_next, a_next)]
            td_error = r + gamma * q_next - Q[(s,a)]
            Q[(s,a)] += alpha * td_error
            sname = ["A", "B", "Goal"][s]
            print(f"  Q({sname}, fwd): δ={td_error:+.4f} → Q={Q[(s,a)]:.4f}")
    print("\nFinal Q-values")
    for s,name in [(0,'A'),(1,'B')]:
        print(f"  Q({name}, fwd) = {Q[(s,0)]:.4f}")
    print("="*65)


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_results(rewards, steps, agent, env):
    """
    Plot the results on the grid
    """
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    fig.suptitle("SARSA: On-policy TD control on GridWorld", fontsize=13, fontweight='bold')
    # 1. Reward per episode
    window = 20
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(rewards, alpha=0.3, color='steelblue', label='Raw')
    axes[0].plot(range(window-1, len(rewards)), smoothed, color='steelblue', lw=2, label=f'{window}-ep avg')
    axes[0].set_title('Total Reward per Episode')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # 2. Steps per episode
    smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    axes[1].plot(steps, alpha=0.3, color='coral')
    axes[1].plot(range(window-1, len(steps)), smoothed_steps, color='coral', lw=2)
    axes[1].set_title('Steps to Reach Goal per Episode')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Steps')
    axes[1].grid(alpha=0.3)

    # 3. Learned greedy policy as arrows
    size = env.size
    grid = np.zeros((size, size))
    action_labels = ['↑', '↓', '←', '→']
    action_arrows  = [(-0.3, 0), (0.3, 0), (0, -0.3), (0, 0.3)]  # dy, dx
 
    for r in range(size):
        for c in range(size):
            s = (r, c)
            q_vals = [agent.Q[(s, a)] for a in env.actions]
            best_a = np.argmax(q_vals)
            dy, dx = action_arrows[best_a]
            axes[2].annotate('', xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    
    axes[2].set_xlim(-0.5, size - 0.5)
    axes[2].set_ylim(-0.5, size - 0.5)
    axes[2].set_xticks(range(size)); axes[2].set_yticks(range(size))
    axes[2].invert_yaxis()
    axes[2].scatter(*env.start[::-1], color='green', s=150, zorder=5, label='Start')
    axes[2].scatter(*env.goal[::-1],  color='red',   s=150, zorder=5, label='Goal')
    axes[2].set_title('Learned Greedy Policy')
    axes[2].legend(loc='upper left'); axes[2].grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('sarsa_gridworld.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: sarsa_gridworld.png")


if __name__ == "__main__":
    show_numerical_trace()
    print("\nTraining SARSA on 5x5 GridWorld, for 500 episodes")
    agent, rewards, steps = train_sarsa(n_episodes=500)
    plot_results(rewards, steps, agent, GridWorld())
    print(f"Final 50-ep avg reward: {np.mean(rewards[-50:]):.3f}")
    print(f"Final 50-ep avg steps:  {np.mean(steps[-50:]):.1f}")

