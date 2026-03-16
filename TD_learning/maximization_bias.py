"""
Maximization buas 
Compares Q-learning and Double Q-learning on the small MDP
A -left-> B -many actions-> Terminal (reward ~ N(-0.1, 1))
A -right-> Terminal (reward 0)
"""
import numpy as np
import matplotlib.pyplot as plt

# ── MDP Definition ──────────────────────────────────────────────────────────
# States: 0=A, 1=B, 2=Terminal
# Actions from A: 0=left, 1=right
# Actions from B: 0..N_B_ACTIONS-1  (all lead to terminal)

N_B_ACTIONS = 10 # actions available in state B
REWARD_MEAN = -0.1
REWARD_STD = 1.0
ALPHA = 0.1
GAMMA = 1.0
EPSILON = 0.1
N_RUNS = 10000
N_EPISODES = 300

# ── Run Experiments ───────────────────────────────────────────────────────────
ql_left = np.zeros(N_EPISODES)
dql_left = np.zeros(N_EPISODES)

def step(state, action):
    """
    One step in an episode
    """
    if state == 0:
        # State A
        if action == 1:
            return 2, 0.0
        else:
            return 1, 0.0
    elif state == 1:
        return 2, np.random.normal(REWARD_MEAN, REWARD_STD)
    else:
        raise ValueError("Stopped from terminal")

def n_actions(state):
    """
    get the action
    """
    return 2 if state == 0 else N_B_ACTIONS

def epsilon_greedy(Q, state, eps=EPSILON):
    """
    Epsilon-greedy way of choosing actions
    """
    na = n_actions(state)
    if np.random.rand() < eps:
        return np.random.randint(na)
    return int(np.argmax(Q[state][:na]))

def q_learning_episode(Q):
    """
    Q-learning episode, one run
    """
    state = 0
    chose_left = False
    while state != 2:
        action = epsilon_greedy(Q, state)
        if state == 0 and action == 0:
            chose_left = True
        next_state, reward = step(state, action)
        if next_state == 2:
            target = reward
        else:
            target = reward + GAMMA * np.max(Q[next_state][:n_actions(next_state)])
        Q[state][action] += ALPHA * (target - Q[state][action])
        state = next_state
    return chose_left

def double_q_episode(Q1, Q2):
    """
    Run ONE episode with Double Q-learning: return 1 if left chosen from A
    """
    state, chose_left = 0, False
    while state != 2:
        na = n_actions(state)
        # Behaviour: Epsilon-greedy on Q1+Q2
        if np.random.rand() < EPSILON:
            action = np.random.randint(na)
        else:
            action = int(np.argmax((Q1[state][:na] + Q2[state][:na])))
        if state == 0 and action == 0:
            chose_left = True
        next_state, reward = step(state, action)
        # Coin flip: update Q1 or Q2
        if np.random.rand() < 0.5:
            if next_state == 2:
                target = reward
            else:
                best = int(np.argmax(Q1[next_state][:n_actions(next_state)]))
                target = reward + GAMMA * Q2[next_state][best]
            Q1[state][action] += ALPHA * (target - Q1[state][action])
        else:
            if next_state == 2:
                target = reward
            else:
                best = int(np.argmax(Q2[next_state][:n_actions(next_state)]))
                target = reward + GAMMA * Q1[next_state][best]
            Q2[state][action] += ALPHA * (target - Q2[state][action])
        state = next_state
    return chose_left

for run in range(N_RUNS):
    # Q-learning tables: state 0 has 2 actions while state 1 has N_B_ACTIONS
    Q_ql = [np.zeros(max(2, N_B_ACTIONS)), np.zeros(max(2, N_B_ACTIONS)), np.zeros(1)]
    Q1_dq = [np.zeros(max(2, N_B_ACTIONS)), np.zeros(max(2, N_B_ACTIONS)), np.zeros(1)]
    Q2_dq = [np.zeros(max(2, N_B_ACTIONS)), np.zeros(max(2, N_B_ACTIONS)), np.zeros(1)]
    for episode in range(N_EPISODES):
        ql_left[episode] += q_learning_episode(Q_ql)
        dql_left[episode] += double_q_episode(Q1_dq, Q2_dq)
ql_left /= N_RUNS
dql_left /= N_RUNS

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
episodes = np.arange(1, N_EPISODES + 1)
 
ax.plot(episodes, ql_left * 100,  color='red',   label='Q-learning',        linewidth=1.5)
ax.plot(episodes, dql_left * 100, color='green', label='Double Q-learning',  linewidth=1.5)
ax.axhline(5, color='black', linestyle='--', linewidth=1, label='Optimal (5% = ε/2)')
 
ax.set_xlabel("Episodes")
ax.set_ylabel("% Left actions from A")
ax.set_title("Maximization Bias: Q-learning vs Double Q-learning\n(Replication of Sutton & Barto Fig. 6.5)")
ax.legend()
ax.set_ylim(0, 100)
ax.set_xlim(1, N_EPISODES)
ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig('maximization_bias_fig6_5.png', dpi=150, bbox_inches='tight')
print("Saved: maximization_bias_fig6_5.png")
plt.close()
 
# ── Print final stats ─────────────────────────────────────────────────────────
print(f"\nFinal % left (episode {N_EPISODES}):")
print(f"  Q-learning:        {ql_left[-1]*100:.1f}%")
print(f"  Double Q-learning: {dql_left[-1]*100:.1f}%")
print("  Optimal:            5.0% (ε-greedy minimum)")


