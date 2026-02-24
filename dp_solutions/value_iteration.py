"""
Value Iteration
"""
import numpy as np
# ─────────────────────────────────────────
# GridWorld setup (same throughout chapter)
# ─────────────────────────────────────────
GRID_SIZE = 4
N_STATES  = 16
N_ACTIONS = 4
TERMINAL  = {0, 15}
ACTION_DELTA = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}
ACTION_NAME  = {0:'↑', 1:'↓', 2:'←', 3:'→'}

def state_to_rc(s):
    """
    State to Rate Control
    """ 
    return divmod(s, GRID_SIZE)

def rc_to_state(r, c):
    """
    Rate Control to State
    """
    return r * GRID_SIZE + c

def get_transitions(s, a):
    """
    Get transitions
    """
    if s in TERMINAL: 
        return s, 0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTA[a]
    nr, nc = r+dr, c+dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c
    return rc_to_state(nr, nc), -1

def print_grid(V, title):
    """
    Print the grid
    """
    print(f"\n The title is {title}")
    print(f"{'─'*36}")
    for r in range(GRID_SIZE):
        print("  " + "".join(f"{V[rc_to_state(r,c)]:8.2f}" for c in range(GRID_SIZE)))

def print_policy(policy, title):
    """
    Print the current policy
    """
    print(f"\n The current policy is \n{policy}")
    print(f"\n  {title}")
    print(f"{'─'*36}")
    for r in range(GRID_SIZE):
        row = "  "
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if s in TERMINAL: 
                row += "  [T]  "
            else: row += f"  {''.join(ACTION_NAME[a] for a in policy[s]):<4} "
        print(row)

# ─────────────────────────────────────────
# VALUE ITERATION
# ─────────────────────────────────────────
# KEY DIFFERENCE from Policy Iteration:
#   Policy Iteration:  V(s) <- sum_a pi(a|s) [r + gamma*V(s')]  (average over policy)
#   Value Iteration:   V(s) <- MAX_a          [r + gamma*V(s')]  (take the best action)
#
# No policy π is needed or tracked during iteration!
# Policy is extracted ONCE at the end from converged V*.
# ─────────────────────────────────────────

def value_iteration(gamma=1.0, theta=1e-4, verbose=True):
    """
    Value iteration
    """
    V = np.zeros(N_STATES)
    sweep, history = 0, []
    if verbose:
        print("=" * 44)
        print("  VALUE ITERATION")
        print("  v(s) = max_a Σ p(s',r|s,a)[r + γv(s')]")
        print("  No policy tracked — just V!")
        print("=" * 44)
    while True:
        delta = 0
        sweep += 1
        V_new = np.zeros(N_STATES)
        for s in range(N_STATES):
            if s in TERMINAL:
                V_new[s] = 0
                continue
            # ── Core: MAX over actions (not sum over policy!) ──
            action_values = []
            for a in range(N_ACTIONS):
                s_prime, reward = get_transitions(s, a)
                action_values.append(reward + gamma*V[s_prime])
            V_new[s] = max(action_values) # Key line
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(delta)
        if verbose:
            print(f"  Sweep {sweep:3d} | Max |Δv| = {delta:.6f}")
        if delta < theta:
            print(f"Converged after sweep {sweep}")
            break
    policy = {}
    for s in range(N_STATES):
        if s in TERMINAL:
            policy[s] = {0: 1.0}
            continue
        action_values = {a: get_transitions(s,a)[1] + gamma * V[get_transitions(s,a)[0]]
                         for a in range(N_ACTIONS)}
        best_val  = max(action_values.values())
        best_acts = [a for a, v in action_values.items() if np.isclose(v, best_val)]
        policy[s] = {a: 1.0/len(best_acts) for a in best_acts}
    return V, policy, history

# ─────────────────────────────────────────
# RUN VALUE ITERATION
# ─────────────────────────────────────────
V_star, pi_star, history = value_iteration(gamma=1.0)
print_grid(V_star, "V* (Optimal Value Function)")
print_policy(pi_star, "π* (Optimal Policy — extracted ONCE at end)")

# ─────────────────────────────────────────
# SIDE-BY-SIDE: PI vs VI — what's different?
# ─────────────────────────────────────────
print(f"\n{'='*44}")
print("  POLICY ITERATION vs VALUE ITERATION")
print(f"{'='*44}")

print("""
  Policy Iteration:
  ─────────────────
  π_0 → [full eval ~173 sweeps] → V_π0
       → [improve] → π_1
       → [full eval ~89 sweeps]  → V_π1
       → [improve] → π_2  ...  3 outer iters

  Value Iteration:
  ─────────────────
  V_0 → [1 max-sweep] → V_1
       → [1 max-sweep] → V_2
       → [1 max-sweep] → V_3  ... 4 sweeps total
       → [extract π* once] → DONE
""")

print(f"{'─'*44}")
print(f"  {'Metric':<30} {'PI':>6} {'VI':>6}")
print(f"{'─'*44}")
print(f"  {'Total inner sweeps (approx)':<30} {'~260':>6} {'4':>6}")
print(f"  {'Outer iterations':<30} {'3':>6} {'4':>6}")
print(f"  {'Policy tracked during?':<30} {'Yes':>6} {'No':>6}")
print(f"  {'Policy extractions':<30} {'3':>6} {'1':>6}")
print(f"  {'Same V* result?':<30} {'Yes':>6} {'Yes':>6}")
print(f"  {'Better for large |S|?':<30} {'No':>6} {'Yes':>6}")
print(f"{'─'*44}")

# ─────────────────────────────────────────
# VERIFY: Bellman Optimality holds on V*
# ─────────────────────────────────────────
print(f"\n{'─'*44}")
print("  Bellman Optimality Verification")
print(f"{'─'*44}")
all_ok = True
for s in range(1, N_STATES-1):
    best = max(get_transitions(s,a)[1] + V_star[get_transitions(s,a)[0]]
               for a in range(N_ACTIONS))
    ok = np.isclose(best, V_star[s], atol=1e-2)
    if not ok: all_ok = False
print(f"  V*(s) == max_a[r + γV*(s')] for all s: {'✓' if all_ok else '✗'}")