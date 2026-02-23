"""
Policy evaluation
# ─────────────────────────────────────────
# 4x4 GridWorld Setup
# ─────────────────────────────────────────
# States 0-15, terminals at 0 and 15
# Actions: 0=Up, 1=Down, 2=Left, 3=Right
# Reward: -1 on every non-terminal transition
"""
import numpy as np

# Global variables
GRID_SIZE = 4
N_STATES  = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
TERMINAL  = {0, 15}

ACTION_DELTA = {
    0: (-1, 0),  # Up
    1: ( 1, 0),  # Down
    2: ( 0,-1),  # Left
    3: ( 0, 1),  # Right
}
ACTION_NAME = {0:'↑', 1:'↓', 2:'←', 3:'→'}
def state_to_rc(s):
    """
    Convert State to Rate Control
    """
    return divmod(s, GRID_SIZE)

def rc_to_state(r, c): 
    """
    Rate Control to State
    """
    return r * GRID_SIZE + c

def get_transitions(s, a):
    """
    For a state s and action a, returns the tuple (next_state, reward)
    """
    if s in TERMINAL:
        return s, 0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTA[a]
    nr, nc = r + dr, c + dc
    # If out of bounds, stay in place
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c
    return rc_to_state(nr, nc), -1

def policy_evaluation(policy, gamma=1.0, theta=1e-4):
    """
    Iterative Policy Evaluation.
    policy[s] = dict {action: probability}
    Returns converged value function V.
    """
    V = np.zeros(N_STATES)
    sweep = 0
    while True:
        delta = 0
        sweep += 1
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            v = 0
            for a, prob in policy[s].items():
                s_prime, reward = get_transitions(s, a)
                v += prob * (reward + gamma * V[s_prime])
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        print(f"Sweep {sweep: 3d}| Max |Δv| = {delta:.6f}")
        if delta < theta:
            print(f"\n✓ Converged after {sweep} sweeps (θ={theta})\n")
            break
    return V

def print_grid(V, title="Value Function"):
    """
    Print the grid
    """
    print(f"\n{'─'*34}")
    print(f"  {title}")
    print(f"{'─'*34}")
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            row += f"{V[s]:7.2f} "
        print(row)
    print()

def print_policy(policy):
    """
    Print the current policy
    """
    print(f"\n{'─'*34}")
    print("  Policy")
    print(f"{'─'*34}")
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL:
                row += "  [T] "
            else:
                # Show action with highest prob
                best_a = max(policy[s], key=policy[s].get)
                row += f"  {ACTION_NAME[best_a]}   "
        print(row)
    print()

# ─────────────────────────────────────────
# Example 1: Random Policy (uniform 0.25 each action)
# ─────────────────────────────────────────
random_policy = {}
for s in range(N_STATES):
    random_policy[s] = {a:0.25 for a in range(N_ACTIONS)}
print("="*40)
print("Example 1: Random Policy (γ=1.0)")
print("="*40)
V_random = policy_evaluation(random_policy, gamma=1.0, theta=1e-4)
print_grid(V_random, "V_π (Random Policy, γ=1.0)")
print_policy(random_policy)

# ─────────────────────────────────────────
# Example 2: "Always Right" policy
# ─────────────────────────────────────────
right_policy = {}
for s in range(N_STATES):
    right_policy[s] = {3: 1.0} # Always right
print("=" * 40)
print("EXAMPLE 2: Always-Right Policy (γ=0.9)")
print("=" * 40)
V_right = policy_evaluation(right_policy, gamma=0.9, theta=1e-4)
print_grid(V_right, "V_π (Always Right, γ=0.9)")
print_policy(right_policy)

# ─────────────────────────────────────────
# Example 3: Alternative Down and Right policy
# ─────────────────────────────────────────
down_right_policy = {}
for s in range(N_STATES):
    if s%2 == 0:
        down_right_policy[s] = {3: 1.0} # Go Right
    else:
        down_right_policy[s] = {1: 1.0} # Go Down
print("=" * 40)
print("EXAMPLE 3: Down-Right Policy (γ=0.9)")
print("=" * 40)
V_down_right = policy_evaluation(down_right_policy, gamma=0.9, theta=1e-4)
print_grid(V_down_right, "V_π (Down and Right alternatively, γ=0.9)")
print_policy(down_right_policy)
