"""
Generalized Policy Iteration, GPI
"""
import numpy as np
# ─────────────────────────────────────────
# GridWorld setup
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
    Get the state
    """
    return r * GRID_SIZE + c

def get_transitions(s, a):
    """
    Get transitions for the environment
    """
    if s in TERMINAL:
        return s, 0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTA[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c
    return rc_to_state(nr, nc), -1

def bellman_update_policy(s, V, policy, gamma):
    """
    One Bellman update under policy
    """
    return sum(prob * (get_transitions(s,a)[1] + gamma * V[get_transitions(s,a)[0]])
               for a, prob in policy[s].items())

def bellman_update_max(s, V, gamma):
    """
    Bellman update using max
    """
    return max(get_transitions(s,a)[1] + gamma * V[get_transitions(s,a)[0]]
               for a in range(N_ACTIONS))

def improve_policy(V, gamma):
    """
    Improve the current policy under the State-value function V
    """
    policy = {}
    for s in range(N_STATES):
        if s in TERMINAL:
            policy[s] = {0: 1.0}
            continue
        vals = {a: get_transitions(s, a)[1] + gamma*V[get_transitions(s, a)[0]]
                for a in range(N_ACTIONS)}
        best = max(vals.values())
        bests = [a for a, v in vals.items() if np.isclose(v, best)]
        policy[s] = {a: 1.0/len(bests) for a in bests}
    return policy

def print_grid(V, title):
    """
    Print the grid as is
    """
    print(f"\n  {title}")
    print(f"{'─'*36}")
    for r in range(GRID_SIZE):
        print("  " + "".join(f"{V[rc_to_state(r,c)]:8.2f}" for c in range(GRID_SIZE)))

# ─────────────────────────────────────────
# GENERALIZED POLICY ITERATION
# eval_sweeps: how many Bellman sweeps before improving
#   eval_sweeps = inf  → full Policy Iteration
#   eval_sweeps = 1    → Value Iteration
#   eval_sweeps = k    → truncated PI (GPI middle ground)
# ─────────────────────────────────────────
def gpi(eval_sweeps, gamma=1.0, theta=1e-4):
    """
    GPI with configurable evaluation depth.
    eval_sweeps=None means run eval to full convergence (= Policy Iteration).
    eval_sweeps=1 means one sweep then improve (= Value Iteration).
    eval_sweeps=k means truncated policy evaluation.
    """
    V = np.zeros(N_STATES)
    policy = {s: {a: 0.25 for a in range(N_ACTIONS)} for s in range(N_STATES)}
    total_sweeps, outer_iters = 0, 0
    while True:
        outer_iters += 1
        # ── EVALUATION PHASE: k sweeps ──
        sweeps_this_iter = 0
        while True:
            delta = 0
            V_new = np.zeros(N_STATES)
            for s in range(N_STATES):
                V_new[s] = bellman_update_policy(s, V, policy, gamma)
                delta = max(delta, abs(V_new[s]-V[s]))
            V = V_new
            total_sweeps += 1
            sweeps_this_iter += 1
            # Step eval based on eval_sweeps
            if eval_sweeps == 0:
                if delta < theta:
                    break
            else:
                if sweeps_this_iter >= eval_sweeps:
                    break
        # --- IMPROVEMENT PHASE ---
        new_policy = improve_policy(V, gamma)
        # --- STABILITY CHECK ---
        stable = all(set(new_policy[s].keys()) == set(policy[s].keys())
                     for s in range(N_STATES))
        policy = new_policy
        if stable:
            break
    return V, policy, total_sweeps, outer_iters

# ─────────────────────────────────────────
# RUN ALL VARIANTS — show they all converge
# to the SAME V* with different costs
# ─────────────────────────────────────────
print("=" * 52)
print("  GPI: Same V*, Different Evaluation Depths")
print("=" * 52)

configs = [
    (0,  "Policy Iteration     (eval_sweeps=full)"),
    (10,    "Truncated PI         (eval_sweeps=10)  "),
    (3,     "Truncated PI         (eval_sweeps=3)   "),
    (1,     "Value Iteration      (eval_sweeps=1)   "),
]

results = []
for s in range(N_STATES):
    for a in range(N_ACTIONS):
        s_prime, r = get_transitions(s, a)
        assert 0 <= s_prime < N_STATES, f"Bad transition: s={s}, a={a}, s'={s_prime}"
print("Transitions OK")
for k, label in configs:
    V, pi, total, outer = gpi(eval_sweeps=k, gamma=1.0)
    results.append((label, total, outer, V))
    print(f"\n  {label}")
    print(f"  → Outer iters: {outer:3d} | Total sweeps: {total:4d} | Converged: ✓")

# ─────────────────────────────────────────
# VERIFY: All produce identical V*
# ─────────────────────────────────────────
print(f"\n{'─'*52}")
print("  Verify: All variants produce same V*?")
print(f"{'─'*52}")
ref_V = results[0][3]
all_match = all(np.allclose(r[3], ref_V, atol=1e-2) for r in results[1:])
print(f"  V* identical across all GPI variants: {'✓' if all_match else '✗'}")

print_grid(ref_V, "V* (same for all variants)")
# ─────────────────────────────────────────
# THE GPI PICTURE — show the interplay
# ─────────────────────────────────────────
print(f"""
{'='*52}
  THE GPI PICTURE
{'='*52}

  Two processes interacting:

  V ──────────────────────────────────► more accurate V
  ▲   Evaluation: V closer to V_pi         │
  │                                         │
  │   Improvement: pi greedy w.r.t. V       ▼
  └─────────────────────────────────── better pi

  They COMPETE:
  → Improving pi makes V inaccurate for new pi
  → Updating V makes pi non-greedy

  They COOPERATE:
  → Both drive toward the SAME fixed point: (V*, pi*)
  → Stable only when pi is greedy w.r.t. its OWN V
  → That IS the Bellman Optimality Equation!

  PI  = full eval (many sweeps) + 1 improve
  VI  = 1 eval sweep            + 1 improve  (repeated)
  GPI = k eval sweeps           + 1 improve  (any k works!)
{'='*52}
""")

# ─────────────────────────────────────────
# SWEEP EFFICIENCY COMPARISON
# ─────────────────────────────────────────
print(f"  {'Variant':<40} {'Outer':>6} {'Total sweeps':>14}")
print(f"  {'─'*62}")
for label, total, outer, _ in results:
    print(f"  {label:<40} {outer:>6} {total:>14}")
print("\n  → All find same optimal policy. Tradeoff: outer iters vs sweep cost.")
