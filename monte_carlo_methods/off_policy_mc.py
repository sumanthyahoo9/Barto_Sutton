"""
Off-policy MC
Implements both:
  - Ordinary Importance Sampling  (unbiased, potentially infinite variance)
  - Weighted Importance Sampling  (biased but finite variance — preferred)

Environment: Blackjack (same as book Example 5.4)
  State : (player_sum, dealer_card, usable_ace)  →  200 states
  Target policy π : stick on 20 or 21, else hit (deterministic)
  Behavior policy b : hit or stick with equal probability (random/soft)
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────── Blackjack Environment ───────────────────────────
def draw_card():
    """
    Draw a card
    """
    return min(np.random.randint(1, 14), 10)

def hand_value(cards):
    """
    Compute the best hand value, usable ace is counted as 11
    """
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21
    return total + (10 if usable_ace else 0), usable_ace

def play_episode(policy_fn):
    """
    Run one blackjack episode under the policy
    Returns list of (state, action, reward) tuples
    """
    # Deal initial hands
    player = [draw_card(), draw_card()]
    dealer_card = draw_card()
    trajectory = []
    # Player's turn
    while True:
        total, usable_ace = hand_value(player)
        # Natural blackjack check
        if total > 21:
            # Bust - reward -1
            state = (min(total, 21), dealer_card, usable_ace)
            trajectory.append((state, 1, -1))
            return trajectory
        state = (total, dealer_card, usable_ace)
        action = policy_fn(state)
        trajectory.append((state, action, 0)) # Reward = 0 until episode ends
        if action == 0:
            break
        player.append(draw_card())
    # Dealer's turn
    dealer_hand = [dealer_card, draw_card()]
    while hand_value(dealer_hand)[0] < 17:
        dealer_hand.append(draw_card())
    dealer_total, _ = hand_value(dealer_hand)
    player_total, _ = hand_value(player)
    # Final reward goes on the last step
    if dealer_total > 21 or player_total > dealer_total:
        final_reward = 1
    elif player_total == dealer_total:
        final_reward = 0
    else:
        final_reward = -1
    trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward)
    return trajectory

# ─────────────────────────── Policies ────────────────────────────────────────
def target_policy(state):
    """
    π: stick (0) on 20 or 21, else hit (1). Deterministic.
    """
    player_sum, _, _ = state
    return 0 if player_sum >= 20 else 1

def behavior_policy(state):
    """
    b: hit ot stick with equal probs, soft/exploratory
    """
    return np.random.choice([0, 1])

def pi_prob(state, action):
    """
    P(action | state) under target policy π.
    """
    return 1.0 if target_policy(state) == action else 0.0

def b_prob():
    """
    P(action|state) under behaviour policy b
    """
    return 0.5

# ─────────────────────────── Off-Policy MC Prediction ────────────────────────
def offpolicy_mc(n_episodes=10000, gamma=1.0):
    """
    Off-policy MC prediction using WEIGHTED importance sampling.
    Estimates Q(s,a) ≈ q_π(s,a).

    Algorithm (from book p.110):
      G ← 0, W ← 1
      Loop steps backward t = T-1 … 0:
        G ← γG + R_{t+1}
        C(S_t, A_t) += W
        Q(S_t, A_t) += W/C(S_t,A_t) * [G - Q(S_t,A_t)]
        W *= π(A_t|S_t) / b(A_t|S_t)
        if W == 0: break
    """
    Q, C = defaultdict(float), defaultdict(float)
    for _ in range(n_episodes):
        episode = play_episode(behavior_policy)
        G, W = 0.0, 1.0
        # Iterate backwards through episode
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            # Weighted IS incremental update
            C[(state, action)] += W
            Q[(state, action)] += (W/C[(state, action)]) * (G - Q[(state, action)])
            # Update importance weight
            W *= pi_prob(state, action) / b_prob()
            if W == 0.0:
                break
    return Q

def offpolicy_mc_ordinary(n_episodes=10000, gamma=1.0):
    """
    Off-policy MC prediction using ORDINAY importance sampling
    Stores all weighted returns then, averages
    """
    returns = defaultdict(list)
    for _ in range(n_episodes):
        episode = play_episode(behavior_policy)
        G, W = 0.0, 1.0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            W *= pi_prob(state, action)/b_prob()
            if W == 0.0:
                break
            returns[(state, action)].append(W*G)
    Q = {k: np.mean(v) for k, v in returns.items()}
    return Q

# ─────────────────────────── Comparison Experiment ───────────────────────────
def run_comparison(target_state, true_value=-0.27726, runs=100, n_episodes=10000):
    """
    Fig 5.3 from the book
    Tracks the MSE of ordinary vs weighted over episodes for a specific state
    target_state: the (player_sum, dealer_card, usable_ace) state to track
    true_value  : known value of that state under π (~-0.27726 per book)
    """
    print(f"Comparing Ordinary vs Weighted IS on state {target_state}")
    print(f"True value ≈ {true_value}\n")

    ep_checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 500,
                      1000, 2000, 5000, 10000]
    ordinary_errors = np.zeros((runs, len(ep_checkpoints)))
    weighted_errors = np.zeros((runs, len(ep_checkpoints)))
    for run in range(runs):
        if (run+1)%20 == 0:
            print(f"  Run {run+1}/{runs}...")
            # Reset accumulators
            Q_w = defaultdict(float)
            C_w = defaultdict(float)
            ordinary_returns = defaultdict(list)
            ep_idx = 0
            for ep_num in range(1, n_episodes+1):
                episode = play_episode(behavior_policy)
                G, W = 0.0, 1.0
                for t in reversed(range(len(episode))):
                    state, action, reward = episode[t]
                    G = gamma_val *G + reward
                    # Weighted states
                    C_w[(state, action)] += W
                    Q_w[(state, action)] += (W / C_w[(state, action)]) * (G - Q_w[(state, action)])
                    # Ordinary accumulation
                    rho = pi_prob(state, action)/b_prob()
                    W *= rho
                    if W == 0.0:
                        break
                    ordinary_returns[(state, action)].append(W*G)
                if ep_idx < len(ep_checkpoints) and ep_num == ep_checkpoints[ep_idx]:
                    target_key_hit = (target_state, 1) # Hit action
                    target_key_stick = (target_state, 0) # stick action
                    # Use the action the target policy would take
                    target_action = target_policy(target_state)
                    target_key = (target_state, target_action)
                    w_val = Q_w.get(target_key, 0.0)
                    o_vals = ordinary_returns.get(target_key, [0.0])
                    o_val = np.mean(o_vals) if o_vals else 0.0
                    weighted_errors[run, ep_idx] = (w_val-true_value) ** 2
                    ordinary_errors[run, ep_idx] = (o_val-true_value) ** 2
                    ep_idx += 1
    return ep_checkpoints, ordinary_errors.mean(axis=0), weighted_errors.mean(axis=0)

# ─────────────────────────── Visualise Learned Q ─────────────────────────────
def visualise_policy(Q, title="Learned Policy from Off-Policy MC"):
    """Show the greedy policy derived from Q across player sums."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)

    for ace_idx, (usable, ax) in enumerate(zip([True, False], axes)):
        label = "Usable Ace" if usable else "No Usable Ace"
        ax.set_title(label)
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(12, 22))

        policy_grid = np.zeros((10, 10))   # player_sum × dealer_card

        for pi, player_sum in enumerate(range(12, 22)):
            for di, dealer_card in enumerate(range(1, 11)):
                state = (player_sum, dealer_card, usable)
                q_hit  = Q.get((state, 1), 0.0)
                q_stick = Q.get((state, 0), 0.0)
                policy_grid[pi, di] = 1 if q_hit > q_stick else 0  # 1=hit, 0=stick

        im = ax.imshow(policy_grid, cmap="RdYlGn", aspect="auto",
                       extent=[0.5, 10.5, 11.5, 21.5], origin="lower")
        plt.colorbar(im, ax=ax, label="1=Hit, 0=Stick")

    plt.tight_layout()
    plt.savefig("offpolicy_policy.png", dpi=120)
    print("Saved: offpolicy_policy.png")

gamma_val = 1.0
if __name__ == "__main__":
    print("=" * 60)
    print("OFF-POLICY MONTE CARLO — Chapter 5 Implementation")
    print("=" * 60)
    # 1. Train with weighted IS
    print("\n[1] Training with Weighted IS (10,000 episodes)...")
    Q_weighted = offpolicy_mc(n_episodes=10_000)
    print(f"    Q-values learned for {len(Q_weighted)} state-action pairs")

    # 2. Show a sample state
    sample_state = (13, 2, True)   # book's Example 5.4 state
    q_hit   = Q_weighted.get((sample_state, 1), "not visited")
    q_stick = Q_weighted.get((sample_state, 0), "not visited")
    print(f"\n[2] Sample state {sample_state}:")
    print(f"    Q(hit)   = {q_hit}")
    print(f"    Q(stick) = {q_stick}")
    print(f"    → Greedy action: {'HIT' if isinstance(q_hit, float) and isinstance(q_stick, float) and q_hit > q_stick else 'STICK'}")

    # 3. Comparison experiment (Fig 5.3)
    print("\n[3] Running Ordinary vs Weighted comparison (100 runs)...")
    eps, ord_err, wt_err = run_comparison(
        target_state=sample_state, true_value=-0.27726, runs=100, n_episodes=10_000
    )

    plt.figure(figsize=(9, 5))
    plt.plot(eps, ord_err, "g-o", label="Ordinary IS", linewidth=2)
    plt.plot(eps, wt_err,  "r-s", label="Weighted IS", linewidth=2)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Episodes (log scale)")
    plt.ylabel("Mean Squared Error (avg over 100 runs)")
    plt.title("Off-Policy MC: Ordinary vs Weighted Importance Sampling\n"
              f"State {sample_state}, True value ≈ -0.277")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("offpolicy_comparison.png", dpi=120)
    print("Saved: offpolicy_comparison.png")

    # 4. Visualize the learned greedy policy
    print("\n[4] Training longer (50,000 episodes) for policy visualisation...")
    Q_big = offpolicy_mc(n_episodes=50_000)
    visualise_policy(Q_big)
