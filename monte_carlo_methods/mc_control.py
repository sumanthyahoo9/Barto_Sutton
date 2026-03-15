"""
Monte Carlo Control
"""
from collections import defaultdict
import numpy as np

# ─────────────────────────────────────────
# Simple GridWorld for MC Control
# 4x4, terminals at 0 and 15, reward=-1
# Now we learn q(s,a) instead of v(s)!
# ─────────────────────────────────────────
GRID_SIZE  = 4
N_STATES   = 16
N_ACTIONS  = 4
TERMINAL   = {0, 15}
ACTION_DELTA = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
ACTION_NAME = {0:'↑', 1:'↓', 2:'←', 3:'→'}

def print_policy(policy, title):
    """
    Print the policy
    """
    print(f"\n {title}")
    print(f"  {'─'*34}")
    for r in range(GRID_SIZE):
        row = " "
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL: 
                row += "  [T] "
            else:
                row += f"   {ACTION_NAME[policy[s]]}   "
        print(row)

def print_q_table(Q, title):
    """
    Best q-value per state
    """
    print(f"\n  {title}")
    print(f"  {'─'*36}")
    for r in range(GRID_SIZE):
        row = "  "
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL:
                row += "    [T]   "
            else:
                best_q = max(Q[s].values()) if Q[s] else 0
                row += f" {best_q:6.2f}"
            print(row)


def state_to_rc(s):
    """
    State to Rate control
    """
    return divmod(s, GRID_SIZE)

def rc_to_state(r, c):
    """
    Rate control to State
    """
    return r * GRID_SIZE + c

def get_next_state_reward(s, a):
    """
    Get the reward for the next state
    """
    if s in TERMINAL:
        return s, 0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTA[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c
    return rc_to_state(nr, nc), -1

def generate_episodes(policy, max_steps=200):
    """
    Generate episode
    Generate episode WITH Exploring Starts:
    → Randomly pick any non-terminal (s,a) as starting point
    → Then follow policy for rest of episode
    This ensures ALL (s,a) pairs get visited over time!
    """
    # Exploring start: random non-terminal state AND random action
    non_terminals = [s for s in range(N_STATES)if s not in TERMINAL]
    s = np.random.choice(non_terminals)
    a = np.random.randint(N_ACTIONS) # Random first action regardless of policy
    episode = []
    for _ in range(max_steps):
        s_next, reward = get_next_state_reward(s, a)
        episode.append((s, a, reward))
        if s_next in TERMINAL:
            break
        s = s_next
        a = policy[s] # Follow policy after the first step
    return episode
    

def mc_control_es(n_episodes, gamma=1.0):
    """
    MC Control with exploring starts
    Learns q(s,a) directly from episodes.
    Improves policy greedily after each episode.
    No model p(s',r|s,a) used anywhere!
    """
    # Q-table: q(s,a) estimates
    Q = {s: {a: 0.0 for a in range(N_ACTIONS)} for s in range(N_STATES)}
    returns_sum   = {s: {a: 0.0 for a in range(N_ACTIONS)} for s in range(N_STATES)}
    returns_count = {s: {a: 0   for a in range(N_ACTIONS)} for s in range(N_STATES)}

    # Start with arbitrary deterministic policy
    # Defined as a dict, where each state maps to a specific action
    policy = {s: np.random.randint(N_ACTIONS)for s in range(N_STATES)}
    for _ in range(n_episodes):
        # Step 1: Generate episode with exploring starts
        episode = generate_episodes(policy)
        # Step 2: Compute the returns backwards
        G = 0
        visited_sa = set()
        for s,a, reward in reversed(episode):
            G = reward + gamma*G
            # First-visit MC for (s,a) pairs
            if (s,a) not in visited_sa:
                visited_sa.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a]/returns_count[s][a]
                # Step 3: Improve policy greedily from q(s,a)
                # No model is needed
                if s not in TERMINAL:
                    policy[s] = max(Q[s], key=Q[s].get)
    return Q, policy


# ─────────────────────────────────────────
# RUN MC CONTROL
# ─────────────────────────────────────────
print("=" * 44)
print("  MC CONTROL WITH EXPLORING STARTS")
print("  Learning q(s,a) from episodes — no model!")
print("=" * 44)
for n_ep in [100, 500, 1000, 5000]:
    Q, pi = mc_control_es(n_episodes=n_ep)
    print_policy(pi, f"Policy after {n_ep} episodes")

# Final detailed results
Q_final, pi_final = mc_control_es(n_episodes=5000)
print_q_table(Q_final, "Best q(s,a) per state after 5000 episodes")

# ─────────────────────────────────────────
# KEY COMPARISON: DP vs MC Control
# ─────────────────────────────────────────
print(f"""
{'='*44}
  DP vs MC CONTROL
{'='*44}
  {'Aspect':<25} {'DP':>8} {'MC':>8}
  {'─'*43}
  {'Needs model p(s,r|s,a)?':<25} {'Yes':>8} {'No':>8}
  {'Value function used':<25} {'v(s)':>8} {'q(s,a)':>8}
  {'How V/Q updated':<25} {'Bellman':>8} {'Average':>8}
  {'Needs full episodes?':<25} {'No':>8} {'Yes':>8}
  {'Works model-free?':<25} {'No':>8} {'Yes':>8}
  {'Guaranteed convergence':<25} {'Yes':>8} {'Yes*':>8}
  {'─'*43}
  * With Exploring Starts & sufficient episodes

  WHY q(s,a) in MC but v(s) in DP?
  → DP has model: can compute pi'(s) = argmax_a[r + g*V(s')]
  → MC no model:  can't do lookahead, so needs q(s,a) directly
    pi'(s) = argmax_a q(s,a)  ← no model required!
{'='*44}
""")
