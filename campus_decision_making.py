
import random
import numpy as np

# ==========================================
# 1. SIMULATED CAMPUS ENVIRONMENT (4x4 Grid)
# ==========================================
# States: 0 to 15 (0 is Dorm/Start, 15 is Target Department/Goal)
# Hazards: State 5 and State 10 are construction zones
# Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left

NUM_STATES = 16
NUM_ACTIONS = 4
START_STATE = 0
GOAL_STATE = 15
HAZARDS = (5, 10)
ALL_ACTIONS = (0, 1, 2, 3)


def get_next_state(state, action):
    """Moves the student on the 4x4 grid."""
    if state == GOAL_STATE:
        return GOAL_STATE

    row = state // 4
    col = state % 4

    if action == 0 and row > 0:      # Up
        row -= 1
    elif action == 1 and col < 3:    # Right
        col += 1
    elif action == 2 and row < 3:    # Down
        row += 1
    elif action == 3 and col > 0:    # Left
        col -= 1

    return row * 4 + col


def get_reward(state):
    """Returns the reward for landing in a state."""
    if state == GOAL_STATE:
        return 10.0   # Reached the target department
    elif state in HAZARDS:
        return -2.0   # Walked into campus construction
    else:
        return -0.1   # Normal walking cost


def take_step(state, action, slip_chance=0.2):
    """Simulates a step where the student might slip to a random direction."""
    if random.random() < slip_chance:
        actual_action = random.choice(ALL_ACTIONS)
    else:
        actual_action = action

    next_state = get_next_state(state, actual_action)
    reward = get_reward(next_state)
    return next_state, reward


# ==========================================
# 2. MDP: VALUE ITERATION & Q-LEARNING
# ==========================================
def run_value_iteration(gamma=0.9):
    """Model-based planning using simple Bellman updates."""
    V = np.zeros(NUM_STATES)

    for _ in range(50):  # Repeat 50 times until values stabilize
        new_V = np.copy(V)
        for s in range(NUM_STATES):
            if s == GOAL_STATE:
                continue

            action_scores = list()
            for a in range(NUM_ACTIONS):
                # 80% chance intended move, 20% chance random move (5% per direction)
                intended = get_next_state(s, a)
                score = 0.8 * (get_reward(intended) + gamma * V[intended])

                for random_a in range(NUM_ACTIONS):
                    slip_s = get_next_state(s, random_a)
                    score += 0.05 * (get_reward(slip_s) + gamma * V[slip_s])

                action_scores.append(score)

            new_V[s] = max(action_scores)
        V = new_V

    return V


def run_q_learning(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    """Model-free learning by trial and error."""
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))

    for _ in range(episodes):
        state = START_STATE
        steps = 0

        while state != GOAL_STATE and steps < 50:
            # Pick a random action sometimes, otherwise pick the best known action
            if random.random() < epsilon:
                action = random.choice(ALL_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward = take_step(state, action)

            # Q-learning update formula
            best_future_q = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * best_future_q - Q[state, action]
            )

            state = next_state
            steps += 1

    return Q


# ==========================================
# 3. POMDP: BELIEF UPDATE (PARTIAL OBSERVABILITY)
# ==========================================
def update_belief(belief, action, observation):
    """Updates probability distribution over campus locations using Bayes' rule."""
    new_belief = np.zeros(NUM_STATES)

    # Step A: Predict where we moved
    for s in range(NUM_STATES):
        if belief[s] > 0:
            intended = get_next_state(s, action)
            new_belief[intended] += belief[s] * 0.8
            for random_a in range(NUM_ACTIONS):
                slip_s = get_next_state(s, random_a)
                new_belief[slip_s] += belief[s] * 0.05

    # Step B: Adjust probabilities based on noisy campus landmark observation
    for s in range(NUM_STATES):
        if s == observation:
            new_belief[s] *= 0.85  # Sensor is 85% accurate
        else:
            new_belief[s] *= 0.01  # Small chance of false reading

    # Normalize so probabilities add up to 1
    total = np.sum(new_belief)
    if total > 0:
        new_belief = new_belief / total

    return new_belief


# ==========================================
# 4. BAMDP: DIRICHLET TRANSITION LEARNING
# ==========================================
def run_bamdp_learning():
    """Learns unknown transition probabilities by counting transitions (Dirichlet prior)."""
    # Start with prior count of 1.0 for every possible transition
    dirichlet_counts = np.ones((NUM_STATES, NUM_ACTIONS, NUM_STATES))

    # Simulate standing at State 0 and trying to move Right (Action 1) 40 times
    for _ in range(40):
        next_s, _ = take_step(START_STATE, 1)
        dirichlet_counts[START_STATE, 1, next_s] += 1.0

    # Calculate learned probabilities for State 0, Action 1
    counts_from_start = dirichlet_counts[START_STATE, 1]
    learned_probs = counts_from_start / np.sum(counts_from_start)
    return learned_probs


# ==========================================
# 5. I-POMDP: REASONING ABOUT ANOTHER STUDENT
# ==========================================
def run_ipomdp_peer_inference():
    """Guesses if another student is 'Competitive' or 'Cautious' based on their moves."""
    # Start 50/50 unsure about the peer's strategy
    prob_competitive = 0.5
    prob_cautious = 0.5

    # Suppose we observe the peer moving Right (1) and Down (2) toward the goal
    observed_peer_actions = (1, 2, 1, 2)
    goal_directed_actions = (1, 2)

    for action in observed_peer_actions:
        # Competitive peers move toward the goal (1 or 2) 80% of the time
        if action in goal_directed_actions:
            likelihood_comp = 0.40
            likelihood_caut = 0.25
        else:
            likelihood_comp = 0.10
            likelihood_caut = 0.25

        prob_competitive *= likelihood_comp
        prob_cautious *= likelihood_caut

        # Normalize probabilities
        total = prob_competitive + prob_cautious
        prob_competitive /= total
        prob_cautious /= total

    return prob_competitive, prob_cautious


# ==========================================
# RUN ALL DEMONSTRATIONS
# ==========================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    action_names = ("Up", "Right", "Down", "Left")

    print("=== Tabular Decision-Making Under Uncertainty ===\n")

    print("--- 1. MDP: Value Iteration & Q-Learning ---")
    V = run_value_iteration()
    Q = run_q_learning()
    best_q_action = int(np.argmax(Q[START_STATE]))
    print(f"Estimated value at Start (State 0): {V[START_STATE]:.2f}")
    print(f"Best action learned by Q-Learning at Start: {action_names[best_q_action]}\n")

    print("--- 2. POMDP: Belief Update ---")
    belief = np.zeros(NUM_STATES)
    belief[START_STATE] = 1.0  # We know we start at State 0
    belief = update_belief(belief, action=1, observation=1)
    most_likely_state = int(np.argmax(belief))
    print(
        f"Most likely state after moving Right: State {most_likely_state} "
        f"(Probability: {belief[most_likely_state]:.2f})\n"
    )

    print("--- 3. BAMDP: Dirichlet Transition Learning ---")
    probs = run_bamdp_learning()
    chance_state_1 = float(probs.item(1))
    print(f"Learned chance of reaching State 1 when moving Right from Start: {chance_state_1:.2f}\n")

    print("--- 4. I-POMDP: Peer Strategy Inference ---")
    comp, caut = run_ipomdp_peer_inference()
    print(f"Probability peer is Competitive: {comp:.2f} | Cautious: {caut:.2f}")
