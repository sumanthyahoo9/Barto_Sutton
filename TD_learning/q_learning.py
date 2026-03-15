"""
Q-learning algorithm
Off-policy TD-control
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ── Environment ───────────────────────────────────────────────────────────────
class CliffWalking:
    """
    Cliff walking environment
    4x12 grid. Start=(3,0), Goal=(3,11).
    Cliff = bottom row cols 1-10. Fall → R=-100, back to start.
    All other steps → R=-1
    """
    def __init__(self):
        self.rows, self.cols = 4, 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = {(3, c) for c in range(1, 11)}
        self.actions = [0, 1, 2, 3] # up, down, left, right
    
    def reset(self):
        """
        Reset for new episode
        """
        self.state = self.start
        return self.state
    
    def step(self, action):
        """
        One step of an episode
        """
        r, c = self.state
        if action == 0:
            r -= 1
        elif action == 1:
            r += 1
        elif action == 2:
            c -= 1
        elif action == 3:
            c += 1
        r = max(0, min(self.rows-1, r))
        c = max(0, min(self.cols-1, c))
        self.state = (r, c)
        if self.state in self.cliff:
            self.state = self.start
            return self.state, -100, False
        if self.state == self.goal:
            return self.state, -1, True
        return self.state, -1, False

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')
 
def plot_policy(ax, Q, env, title):
    arrows = {0: (-0.3,0), 1: (0.3,0), 2: (0,-.3), 3: (0,.3)}
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            if s in env.cliff:
                ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1, color='#c0392b', alpha=0.6))
                continue
            if s == env.goal:
                ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1, color='#27ae60', alpha=0.6))
                continue
            best = max(env.actions, key=lambda a: Q[(s,a)])
            dy, dx = arrows[best]
            ax.annotate('', xy=(c+dx, r+dy), xytext=(c,r),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
    ax.scatter(*env.start[::-1], color='lime', s=120, zorder=5, label='Start')
    ax.set_xlim(-0.5, env.cols-.5); ax.set_ylim(-0.5, env.rows-.5)
    ax.set_xticks(range(env.cols)); ax.set_yticks(range(env.rows))
    ax.invert_yaxis(); ax.set_title(title, fontweight='bold')
    ax.text(5.5, 3, 'THE CLIFF', ha='center', va='center',
            color='white', fontweight='bold', fontsize=9)
 
def plot_all(sarsa_r, ql_r, Q_sarsa, Q_ql, env):
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('Sarsa vs Q-Learning: Cliff Walking (Example 6.6)', fontsize=13, fontweight='bold')
 
    # 1. Reward curves
    ax1 = fig.add_subplot(2, 2, (1, 2))
    w = 10
    ax1.plot(range(w-1, len(sarsa_r)), smooth(sarsa_r, w),
             color='steelblue', lw=2, label='Sarsa (safe path)')
    ax1.plot(range(w-1, len(ql_r)),    smooth(ql_r, w),
             color='coral',     lw=2, label='Q-learning (optimal path)')
    ax1.axhline(-13, color='steelblue', ls='--', alpha=0.5, label='Sarsa optimal ≈ -13')
    ax1.axhline(-11, color='coral',     ls='--', alpha=0.5, label='Q-learning optimal ≈ -11')
    ax1.set_xlabel('Episodes'); ax1.set_ylabel('Sum of rewards per episode')
    ax1.set_ylim(-120, 0); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title('Online Performance: Sarsa outperforms Q-learning during training\n'
                  '(Q-learning occasionally falls off cliff due to ε-exploration)')
 
    # 2. Learned policies
    ax2 = fig.add_subplot(2, 2, 3)
    plot_policy(ax2, Q_sarsa, env, 'Sarsa Learned Policy\n(safer upper path)')
 
    ax3 = fig.add_subplot(2, 2, 4)
    plot_policy(ax3, Q_ql, env, 'Q-Learning Learned Policy\n(optimal edge-of-cliff path)')
 
    plt.tight_layout()
    plt.savefig('qlearning_vs_sarsa.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("Saved: qlearning_vs_sarsa.png")


def epsilon_greedy(Q, state, actions, epsilon):
    """
    Epsilon-greedy
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    q_vals = [Q[(state, a)] for a in actions]
    return actions[np.argmax(q_vals)]

def train_qlearning(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):
    """
    Q-learning off policy
    """
    Q = defaultdict(float)
    rewards = []
    for _ in range(episodes):
        s = env.reset()
        total_r = 0
        while True:
            a = epsilon_greedy(Q, s, env.actions, epsilon)
            s_next, r, done = env.step(a)
            # Q-learning: Uses max over next actions
            q_next = 0.0 if done else max(Q[(s_next, a2)] for a2 in env.actions)
            td_err = r + gamma * q_next - Q[(s,a)]
            Q[(s, a)] += alpha * td_err
            s = s_next
            total_r += r
            if done:
                break
        rewards.append(total_r)
    return Q, rewards

def train_sarsa(env, episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1):
    """
    Train a SARSA policy
    """
    Q = defaultdict(float)
    rewards = []
 
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, env.actions, epsilon)   # choose A first
        total_r = 0
 
        while True:
            s_next, r, done = env.step(a)
            a_next = epsilon_greedy(Q, s_next, env.actions, epsilon)
 
            # Sarsa: uses ACTUAL next action a_next
            q_next = 0.0 if done else Q[(s_next, a_next)]
            td_err = r + gamma * q_next - Q[(s, a)]
            Q[(s, a)] += alpha * td_err
 
            s, a = s_next, a_next
            total_r += r
            if done: break
 
        rewards.append(total_r)
    return Q, rewards

def numerical_trace():
    """
    Compare SARSA and Q-learning
    """
    print("=" * 60)
    print("SARSA vs Q-LEARNING: Update Comparison (3 episodes)")
    print("3-state chain: A → B → Goal,  α=0.1, γ=0.9")
    print("=" * 60)
 
    Q_sarsa = defaultdict(float)
    Q_ql    = defaultdict(float)
    alpha, gamma = 0.1, 0.9
    # Only one action (fwd=0) so max = same value --> results are identical
    for ep in range(1, 4):
        print(f"\n--- Episode {ep} ---")
        transitions = [(0, 0, 0.0, 1, 0, False),
                       (1, 0, 1.0, 2, 0, True)]
        for (s, a, r, s_next, a_next, done) in transitions:
            sname = ["A", "B", "Goal"][s]
            # SARSA: next action is given
            q_next_sarsa = 0.0 if done else Q_sarsa[(s_next, a_next)]
            td_sarsa = r + gamma * q_next_sarsa - Q_sarsa[(s, a)]
            # Q-learning: takes max over next actions
            q_next_ql = 0.0 if done else max(Q_ql[(s_next, aa)] for aa in [0])
            td_ql = r + gamma * q_next_ql - Q_ql[(s, a)]
            Q_ql[(s, a)] += alpha * td_ql
            print(f"  Q({sname},fwd)  Sarsa δ={td_sarsa:+.4f} → {Q_sarsa[(s,a)]:.4f} | "
                  f"Q-learn δ={td_ql:+.4f} → {Q_ql[(s,a)]:.4f}")
    print("\n(Values identical here — difference emerges with multiple actions & ε-exploration)")
    print("=" * 60)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    numerical_trace()
    env = CliffWalking()
    print("\nRunning 500 episodes each (averaged over 50 runs)...")
 
    # Average over multiple runs for stable curves
    sarsa_avg = np.zeros(500)
    ql_avg    = np.zeros(500)
    n_runs = 50
    for _ in range(n_runs):
        _, sr = train_sarsa(CliffWalking())
        _, qr = train_qlearning(CliffWalking())
        sarsa_avg += np.array(sr)
        ql_avg    += np.array(qr)
    sarsa_avg /= n_runs
    ql_avg    /= n_runs
 
    # Final policies from single run
    Q_sarsa, _ = train_sarsa(CliffWalking(), episodes=2000)
    Q_ql,    _ = train_qlearning(CliffWalking(), episodes=2000)
 
    plot_all(sarsa_avg, ql_avg, Q_sarsa, Q_ql, env)
    print(f"Final 50-ep avg  Sarsa: {np.mean(sarsa_avg[-50:]):.1f}")
    print(f"Final 50-ep avg  Q-learning: {np.mean(ql_avg[-50:]):.1f}")