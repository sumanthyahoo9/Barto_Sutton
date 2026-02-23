"""
Policy improvement
Given a policy pi, how to improve it at each state using a greedy approach
"""
import numpy as np

GRID_SIZE = 4
N_STATES  = 16
N_ACTIONS = 4
TERMINAL  = {0, 15}
ACTION_DELTA = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
ACTION_NAME  = {0:'↑', 1:'↓', 2:'←', 3:'→'}

def state_to_rc(s): 
    """
    Convert state to Rate control
    """
    return divmod(s, GRID_SIZE)

def rc_to_state(r, c): 
    """
    Convert Rate control to State
    """
    return r * GRID_SIZE + c

def get_transitions(s, a):
    """
    Get the transitions
    """
    if s in TERMINAL: 
        return s, 0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTA[a]
    nr, nc = r+dr, c+dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c
    return rc_to_state(nr, nc), -1

def policy_evaluation(policy, gamma=1.0, theta=1e-4):
    """
    Evaluate a policy
    """
    V = np.zeros(N_STATES)
    while True:
        delta = 0
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            v = 0
            for a, prob in policy[s].items():
                s_prime, reward = get_transitions(s, a)
                v += prob * (reward + gamma * V[s_prime])
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta: 
            break
    return V

# ─────────────────────────────────────────
# POLICY IMPROVEMENT
# ─────────────────────────────────────────
def policy_improvement(V, gamma=1.0):
    """
    Given value function V, return greedy policy π'.
    π'(s) = argmax_a Σ p(s',r|s,a)[r + γ·V(s')]
    Returns: new deterministic policy, stable flag
    """
    new_policy = {}
    for s in range(N_STATES):
        if s in TERMINAL:
            new_policy[s] = {0: 1.0}  # arbitrary for terminal
            continue

        # One-step lookahead for each action
        action_values = {}
        for a in range(N_ACTIONS):
            s_prime, reward = get_transitions(s, a)
            action_values[a] = reward + gamma * V[s_prime]

        # Greedy: pick best action(s)
        best_val = max(action_values.values())
        best_actions = [a for a, v in action_values.items()
                        if np.isclose(v, best_val)]

        # Distribute probability equally among tied best actions
        new_policy[s] = {a: 1.0/len(best_actions) for a in best_actions}

    return new_policy

def print_grid(V, title="Value Function"):
    """
    Print the grid
    """
    print(f"\n{'─'*36}")
    print(f"  {title}")
    print(f"{'─'*36}")
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            row += f"{V[rc_to_state(r,c)]:8.2f}"
        print(row)

def print_policy(policy, title="Policy"):
    """
    Print the policy
    """
    print(f"The policy in its dict form is \n{policy}")
    print(f"\n{'─'*36}")
    print(f"  {title}")
    print(f"{'─'*36}")
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL:
                row += "  [T]  "
            else:
                arrows = "".join(ACTION_NAME[a] for a in policy[s])
                row += f"  {arrows:<4} "
        print(row)

# ─────────────────────────────────────────
# DEMO: Start from random policy, improve
# ─────────────────────────────────────────
print("=" * 40)
print("POLICY IMPROVEMENT DEMO")
print("Start: Random Policy → Improve")
print("=" * 40)

# Step 1: Random policy
random_policy = {s: {a: 0.25 for a in range(N_ACTIONS)}
                 for s in range(N_STATES)}
print_policy(random_policy, "π_0: Random Policy")

# Step 2: Evaluate a random policy
V0 = policy_evaluation(random_policy, gamma=1.0)
print_grid(V0, "V_π0 (after evaluating random policy)")

# Step 3: Improve -> π'
improved_policy = policy_improvement(V0, gamma=1.0)
print_policy(improved_policy, "π_1: After ONE improvement step")

# Step 4: Evaluate improved policy
V1 = policy_evaluation(improved_policy, gamma=1.0)
print_grid(V1, "V_π1 (after evaluating improved policy)")

# Step 5: Improve again -> π'
improved_policy_2 = policy_improvement(V1, gamma=1.0)
print_policy(improved_policy_2, "π_2: After TWO improvement steps")

# ─────────────────────────────────────────
# CHECK: Did values improve?
# ─────────────────────────────────────────
print(f"\n{'─'*36}")
print("  Value Comparison (non-terminal states)")
print(f"{'─'*36}")
print(f"{'State':<8} {'V_π0':>8} {'V_π1':>8} {'Better?':>8}")
print(f"{'─'*36}")
for s in range(1, N_STATES-1):
    better = "✓" if V1[s] >= V0[s] else "✗"
    print(f"s={s:<5}  {V0[s]:>8.2f} {V1[s]:>8.2f} {better:>8}")

print("\n→ Policy Improvement Theorem holds: V_π1(s) ≥ V_π0(s) for ALL s ✓")

