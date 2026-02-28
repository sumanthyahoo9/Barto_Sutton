"""
Monte Carlo prediction
Estimate v_π(s) for a given policy π by averaging actual returns 
observed from state s across many episodes
Withou a model we can't use Bellman equations directly
Using the law of large numbers, average of observed returns
converges to expected return as episodes -> ∞
First-Visit MC: Average Returns from the FIRST 
time s is visited per episode.
Every-Visit MC: Average Returns from EVERY time s is visited
per episode
BOTH converge to v_π(s)
1-D RANDOM WALK EXPERIMENT
"""
from collections import defaultdict
import numpy as np

N_STATES = 7
LEFT_TERM = 0
RIGHT_TERM = 6
STATE_NAMES = ['L', '1', '2', '3', '4', '5', 'R']

TRUE_VALUES = {
    1: -(2/3),
    2: -(1/3),
    3: 0.0,
    4: 1/3,
    5: 2/3
}

# ––– Generate episodes from state, reward and actions –––
def generate_episode(start_state=3):
    """
    Run one complete episode from start state
    Returns a list of (state, reward) tuples
    """
    episode = []
    s = start_state
    while True:
        # Random policy: go left or right with equal prob
        action = np.random.choice([-1, 1])
        s_next = s + action
        # Determine reward
        if s_next == LEFT_TERM:
            reward = -1
        elif s_next == RIGHT_TERM:
            reward = 1
        else:
            reward = 0
        episode.append((s, reward))
        s = s_next
        if s == LEFT_TERM or s == RIGHT_TERM:
            break
    return episode

def compute_returns(episode, gamma=1.0):
    """
    Compute actual return G_t for each step in episode.
    G_t = R_{t+1} + gamma*R_{t+2} + ... + gamma^{T-1}*R_T
    Works backwards from end of episode.
    """
    returns = []
    G = 0
    for state, reward in reversed(episode):
        G = reward + gamma * G
        returns.append((state, G))
    returns.reverse()
    return returns

# ─────────────────────────────────────────
# FIRST-VISIT MC PREDICTION
# ─────────────────────────────────────────
def first_visit_mc(n_episodes=1000, start_state=3):
    """
    First-Visit MC Prediction:
    For each episode, only use the FIRST visit to each state
    when averaging returns.
    """
    # Track sum of returns and visit count per state
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)
    V_history = {s: [] for s in range(1, 6)} # For convergence
    for ep in range(n_episodes):
        episode = generate_episode(start_state)
        returns = compute_returns(episode)
        # First visit: Track which states seen this episode
        visited = set()
        for state, G in returns:
            if state not in visited and state not in (LEFT_TERM, RIGHT_TERM):
                visited.add(state) # Mark as visited
                returns_sum[state] += G # Accumulate return
                returns_count[state] += 1
                V[state] = returns_sum[state]
        # Track V history every 10 episodes for convergence plot
        if (ep+1) % 10 == 0:
            for s in range(1, 6):
                V_history[s].append(V[s])
    return dict(V), V_history, returns_count

# ─────────────────────────────────────────
# EVERY-VISIT MC PREDICTION
# ─────────────────────────────────────────
def every_visit_mc(n_episodes=1000, start_state=3):
    """
    Every-Visit MC Prediction
    Use EVERY visit to a state within an episode
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)
    for _ in range(n_episodes):
        episode = generate_episode(start_state)
        returns = compute_returns(episode)
        # Every-visit: no "visited", use all occurrences
        for state, G in returns:
            if state not in (LEFT_TERM, RIGHT_TERM):
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state]/returns_count[state]
    return dict(V)

# RUN AND COMPARE
np.random.seed(42)
print("=" * 52)
print("  MC PREDICTION — 1D Random Walk")
print("  States: L-1-2-3-4-5-R  (start at 3)")
print("  Policy: random left/right equally")
print("=" * 52)

# Show a few sample episodes
print("\n  Sample Episodes (first 5):")
print(f"  {'─'*48}")
for i in range(5):
    episode = generate_episode(start_state=3)
    path = [STATE_NAMES[3]]
    for s,r in episode:
        next_s = s + (1 if r >= 0 else -1) if r != 0 else s
    path = "→".join([STATE_NAMES[s] for s, _ in episode])
    final_r = episode[-1][1]
    print(f"  Ep {i+1}: {path} → {'R' if final_r>0 else 'L'} | Return={final_r:+.0f}")

# ── Run First-Visit MC ──
print(f"\n{'─'*52}")
print("  FIRST-VISIT MC (1000 episodes)")
print(f"{'─'*52}")
V_fv, V_history, counts = first_visit_mc(n_episodes=1000)
print(f"\n  {'State':<8} {'True V':>10} {'MC Est':>10} {'Error':>10} {'Visits':>8}")
print(f"  {'─'*48}")
for s in range(1, 6):
    true_v = TRUE_VALUES[s]
    mc_v = V_fv.get(s, 0.0)
    error = abs(mc_v - true_v)
    print(f"  s={STATE_NAMES[s]:<5}  {true_v:>10.4f} {mc_v:>10.4f} {error:>10.4f} {counts[s]:>8}")

# ––– Run Every-visit MC ––
print(f"\n{'─'*52}")
print("  EVERY-VISIT MC (1000 episodes)")
print(f"{'─'*52}")
V_ev = every_visit_mc(n_episodes=1000)
print(f"\n  {'State':<8} {'True V':>10} {'MC Est':>10} {'Error':>10}")
print(f"  {'─'*40}")
for s in range(1, 6):
    true_v = TRUE_VALUES[s]
    mc_v = V_ev.get(s, 0.0)
    error = abs(mc_v - true_v)
    print(f"  s={STATE_NAMES[s]:<5}  {true_v:>10.4f} {mc_v:>10.4f} {error:>10.4f}")

# ––– Convergence: Show how estimates improve with more episodes –––
print(f"\n{'─'*52}")
print("  CONVERGENCE: V(state=3) estimate over episodes")
print("  True value = 0.0 (symmetric middle state)")
print(f"{'─'*52}")
print(f"  {'Episodes':>10} {'V(3) estimate':>16} {'Error':>10}")
print(f"  {'─'*38}")
np.random.seed(42)
V_run = defaultdict(float)
R_sum = defaultdict(float)
R_cnt = defaultdict(float)
for eps in range(1000):
    episode = generate_episode(3)
    returns = compute_returns(episode)
    visited = set()
    for state, G in returns:
        if state not in visited and state not in (LEFT_TERM, RIGHT_TERM):
            visited.add(state)
            R_sum[state] += G
            R_cnt[state] += 1
            V_run[state] = R_sum[state]/R_cnt[state]
        if (eps+1) in [1, 5, 10, 50, 100, 500, 1000]:
            v3 = V_run.get(3, 0.0)
            print(f"  {eps+1:>10} {v3:>16.4f} {abs(v3):>10.4f}")

# –– Key differences summary ––
print(f"""
{'='*52}
  FIRST-VISIT vs EVERY-VISIT
{'='*52}
  First-Visit MC:
  → Only counts first occurrence of s per episode
  → Unbiased estimator — better theoretical guarantees
  → Standard choice in most implementations

  Every-Visit MC:
  → Counts every occurrence of s per episode
  → Also converges (law of large numbers)
  → More data used but slightly biased for finite samples

  Both converge to true v_pi(s) as episodes → infinity!

  KEY vs DP:
  → DP:  V(s) = sum_a pi(a|s) * sum_{{s',r}} p(...)[r + gV(s')]
           Uses model p(s',r|s,a) — must KNOW environment
  → MC:  V(s) = average of actual observed returns G_t
           Uses experience — NO model needed!
{'='*52}
""")

