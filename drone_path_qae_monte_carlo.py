import math
import random
import heapq
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import AmplitudeEstimation, EstimationProblem


# =====================================================================
# 1. DEFINITION OF A SIMPLE DRONE WORLD
# =====================================================================

# Grid width and height.
# The board is 5 x 5, so x and y coordinates belong to {0, 1, 2, 3, 4}.
W = 5
H = 5

# Drone starting point.
start = (0, 0)

# Drone target point.
goal = (4, 4)

# Static obstacles.
# They are known before path planning, and the A* algorithm avoids them.
# They can be interpreted as buildings, fixed no-fly zones, etc.
static_obstacles = {(1, 1), (1, 2), (3, 2), (3, 3)}

# Maximum allowed flight time.
# If the flight takes longer than T_MAX, the scenario is treated as failed.
T_MAX = 10


# =====================================================================
# 2. UNCERTAINTY SCENARIOS
# =====================================================================

# Each scenario describes additional uncertain flight conditions.
# dynamic: a set of dynamic obstacles that appear only in a given scenario.
# delay: an additional flight delay in a given scenario.
#
# set() means an empty set, so there are no dynamic obstacles.
scenarios = [
    {"dynamic": set(), "delay": 0},
    {"dynamic": {(2, 0)}, "delay": 0},
    {"dynamic": {(4, 2)}, "delay": 0},
    {"dynamic": {(2, 4)}, "delay": 1},
    {"dynamic": {(0, 3)}, "delay": 2},
    {"dynamic": {(2, 0), (4, 2)}, "delay": 0},
    {"dynamic": {(4, 1)}, "delay": 1},
    {"dynamic": {(3, 4)}, "delay": 0},
]

# We assume that all scenarios are equally likely.
# For 8 scenarios, each scenario has probability 1/8.
scenario_probabilities = np.ones(len(scenarios)) / len(scenarios)


# =====================================================================
# 3. CLASSICAL PART: A* ALGORITHM
# =====================================================================


def neighbors(cell):
    x, y = cell

    # Four possible moves on the grid.
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Valid neighboring cells will be stored here.
    out = []

    for dx, dy in moves:
        # New potential position.
        nx, ny = x + dx, y + dy

        # We check three conditions:
        # 1. nx is within the board range,
        # 2. ny is within the board range,
        # 3. the new cell is not a static obstacle.
        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in static_obstacles:
            out.append((nx, ny))

    return out



def heuristic(a, b):
    # Metric: Manhattan distance.
    # This fits grid movement without diagonal moves.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def astar():
    '''
    Finds a path from the start to the goal while avoiding static obstacles.

    Important distinction:
        - static obstacles are considered during path search,
        - dynamic obstacles from scenarios are NOT considered here;
          they are used later during path evaluation.

    A* does not initially know the true path length to the goal.
    For each cell, it computes only the priority:

        priority = known cost of reaching the cell + optimistic distance to the goal

    In the code:
        new_cost = number of moves made from the start to the cell nxt
        heuristic(nxt, goal) = Manhattan distance from nxt to the goal

    Manhattan distance does NOT take obstacles into account.
    It is only a fast estimate of how many moves would be needed without obstacles.
    Obstacles are handled separately in the neighbors() function, which prevents
    entering cells included in static_obstacles.

    Therefore:
        heuristic() helps choose a promising direction,
        neighbors() makes sure that the path does not enter a static obstacle.
    '''

    # q is the queue of cells to check.
    # Each element has the form:
    #     (priority, cell)
    #
    # priority says how promising a given cell is.
    # The lower the priority, the earlier A* checks that cell.
    q = []

    # At the beginning, only the start cell is known.
    # The start has priority 0 because the search begins there.
    # We add the start point with coordinates (0, 0) to the priority queue q.
    heapq.heappush(q, (0, start))

    # came stores the cell from which we reached a given cell.
    #
    # Example:
    #     came[(0, 2)] = (0, 1)
    #
    # This is needed at the end to reconstruct the final path.
    came = {start: None}

    # cost stores the real cost of reaching a given cell from the start.
    #
    # Example:
    #     cost[(0, 2)] = 2
    #
    # means:
    #     reaching (0, 2) from the start requires 2 moves.
    #
    # This is the true known cost, not a heuristic estimate.
    cost = {start: 0}

    # Continue as long as there are cells to check.
    while q:
        # Remove the cell with the lowest priority.
        #
        # If q contains, for example:
        #     (8, (0, 1))
        #     (10, (2, 0))
        #
        # then (0, 1) is selected because it has lower priority.
        _, current = heapq.heappop(q)

        # If the selected cell is the goal, stop the search.
        # At this point, we have the information needed to reconstruct the path.
        if current == goal:
            break

        # Check cells that can be reached from current in one move.
        #
        # The function neighbors(current) rejects:
        #     - cells outside the board,
        #     - cells that are static obstacles.
        #
        # This is where A* actually takes static_obstacles into account.
        for nxt in neighbors(current):
            # Moving from current to nxt requires one additional move.
            # The cost of reaching nxt increases by 1 relative to current.
            new_cost = cost[current] + 1

            # Update information about nxt only if:
            #
            # 1. nxt has not been found before,
            # or
            # 2. we have found a shorter path to nxt than before.
            if nxt not in cost or new_cost < cost[nxt]:
                # Store the best currently known cost of reaching nxt.
                cost[nxt] = new_cost

                # Compute the priority of cell nxt.
                #
                # new_cost:
                #     true number of moves from the start to nxt.
                #
                # heuristic(nxt, goal):
                #     optimistic estimate of the number of moves from nxt to the goal.
                #     It does not include obstacles.
                #
                # Example:
                #     nxt = (1, 0)
                #     goal = (4, 4)
                #     heuristic = |1 - 4| + |0 - 4| = 7
                #
                # This 7 does not mean that the goal can certainly be reached in 7 moves.
                # It only means that without obstacles at least 7 moves would be needed.
                priority = new_cost + heuristic(nxt, goal)

                # Add nxt to the queue of cells to check.
                #
                # If two cells have the same priority, Python may break the tie
                # by comparing the cell coordinates, e.g. (0, 1) before (1, 0).
                # That is why a border path may appear in this example.
                heapq.heappush(q, (priority, nxt))

                # Store that the best currently known path to nxt goes through current.
                came[nxt] = current

    # When A* reaches the goal, the predecessors of cells are stored in came.
    #
    # Example:
    #     came[(4, 4)] = (3, 4)
    #     came[(3, 4)] = (2, 4)
    #     came[(2, 4)] = (1, 4)
    #
    # This means we can move backward from the goal to the start.
    path = []
    current = goal

    # Reconstruct the path from the goal to the start.
    while current is not None:
        path.append(current)
        current = came[current]

    # At this point, path is reversed, e.g.:
    #     goal -> ... -> start
    #
    # Reverse the list to get:
    #     start -> ... -> goal
    path.reverse()

    return path


# =====================================================================
# 4. CLASSICAL PART: PATH EVALUATION
# =====================================================================


def path_safe(path, scenario):
    '''
    Checks whether a given path is safe in a given scenario.

    Returns:
        1 if the path is safe,
        0 if the path hits an obstacle or exceeds the time limit.
    '''

    # Combine static and dynamic obstacles.
    # The | operator for sets means set union.
    blocked = static_obstacles | scenario["dynamic"]

    # If any cell on the path is blocked, there is a collision.
    for cell in path:
        if cell in blocked:
            return 0

    # The number of moves is the number of cells minus 1.
    # Add the delay from the given scenario.
    flight_time = len(path) - 1 + scenario["delay"]

    # If the flight is too long, the scenario is treated as failed.
    if flight_time > T_MAX:
        return 0

    # No collision and the time limit is satisfied.
    return 1



def path_cost_score(path, scenario):
    '''
    Computes a simple quality score for a path in a given scenario.

    This is not only a 0/1 value.
    If there is a collision, the score is 0.0.
    If there is no collision, the score depends on the flight time:
        score = 1 - flight_time / T_MAX

    The shorter the flight, the higher the score.
    '''

    # Combine static and dynamic obstacles in the same way as before.
    blocked = static_obstacles | scenario["dynamic"]

    # A collision gives quality score 0.0.
    for cell in path:
        if cell in blocked:
            return 0.0

    # Flight time including the scenario delay.
    flight_time = len(path) - 1 + scenario["delay"]

    # Simple quality score: the shorter the flight relative to the limit, the better.
    score = 1.0 - flight_time / T_MAX

    # Clamp the score to the interval [0, 1].
    return max(0.0, min(1.0, score))



def exact_probability(path):
    '''
    Computes the exact safety probability.

    Since there are only 8 scenarios, all scenarios can be checked directly.
    '''

    # values may look like [1, 1, 1, 0, 0, 1, 1, 0].
    values = np.array([path_safe(path, s) for s in scenarios], dtype=float)

    # The dot product of probabilities and 0/1 values gives the weighted average.
    return float(np.dot(scenario_probabilities, values))



def exact_expected_score(path):
    '''
    Computes the exact average path quality across all scenarios.
    '''

    # values is a list of quality scores, for example [0.2, 0.2, 0.0, ...].
    values = np.array([path_cost_score(path, s) for s in scenarios], dtype=float)

    # Weighted average of quality scores.
    return float(np.dot(scenario_probabilities, values))



def monte_carlo_probability(path, shots):
    '''
    Estimates the safety probability using the Monte Carlo method.

    Instead of checking all scenarios exactly, scenarios are sampled
    according to their probabilities, and the average of 0/1 outcomes is computed.
    '''

    # Sample scenario indices, for example [0, 7, 2, 2, 5, ...].
    idx = np.random.choice(len(scenarios), size=shots, p=scenario_probabilities)

    # Check path safety for each sampled scenario.
    values = [path_safe(path, scenarios[i]) for i in idx]

    # The mean of 0/1 values is the estimated success probability.
    return sum(values) / shots



def monte_carlo_score(path, shots):
    '''
    Estimates the average path quality using the Monte Carlo method.
    '''

    # Sample scenarios.
    idx = np.random.choice(len(scenarios), size=shots, p=scenario_probabilities)

    # Compute the score for each sampled scenario.
    values = [path_cost_score(path, scenarios[i]) for i in idx]

    # Mean of the quality scores.
    return sum(values) / shots


# =====================================================================
# 5. QUANTUM PART IN QISKIT: CIRCUIT CONSTRUCTION
# =====================================================================


def mark_good_scenario(qc, scenario_index, scenario_qubits, target_qubit):
    '''
    Marks one specific good scenario in the quantum circuit.

    Assumption:
        scenario_qubits store the scenario number in binary form.
        target_qubit should be set to 1 if the scenario is good.

    Example:
        scenario_index = 5
        5 in binary is 101

    We want to implement:
        |101>|0> -> |101>|1>

    We use the mcx gate, i.e. multi-controlled X.
    mcx acts when all control qubits are equal to 1.

    If the desired pattern contains 0, that qubit is temporarily flipped with X,
    then mcx is applied, and finally the flip is undone.
    '''

    # Step 1: pattern matching.
    # If a given scenario bit should be 0, apply X to change it to 1.
    for q in scenario_qubits:
        bit = (scenario_index >> q) & 1
        if bit == 0:
            qc.x(q)

    # Step 2: if all scenario qubits are now equal to 1,
    # flip target_qubit from 0 to 1.
    qc.mcx(scenario_qubits, target_qubit)

    # Step 3: undo the X gates from step 1 so the scenario register is restored.
    for q in scenario_qubits:
        bit = (scenario_index >> q) & 1
        if bit == 0:
            qc.x(q)



def build_state_preparation_circuit(path):
    '''
    Builds circuit A, i.e. state_preparation for QAE.

    Circuit A prepares a state in which the amplitude of the state with target=1
    corresponds to the safety probability of the path.

    For 8 = 2**3 scenarios, 3 scenario qubits are needed:
        |000>, |001>, ..., |111>

    One target qubit is added:
        target = 1 means a good scenario,
        target = 0 means a bad scenario.
    '''

    # Determine classically which scenarios are good.
    # In a realistic algorithm, this logic would be a reversible quantum oracle.
    good_indices = {i for i, s in enumerate(scenarios) if path_safe(path, s) == 1}

    # Number of qubits required to encode scenarios.
    # For 8 scenarios: log2(8) = 3.
    scenario_bits = int(math.log2(len(scenarios)))

    # Scenario qubits have indices 0, 1, 2.
    scenario_qubits = list(range(scenario_bits))

    # The target qubit is the last one, so for 3 scenario qubits it has index 3.
    target_qubit = scenario_bits

    # Create a quantum circuit with 3 scenario qubits and 1 target qubit.
    qc = QuantumCircuit(scenario_bits + 1, name="A")

    # Hadamard gates create a uniform superposition of all scenarios.
    # After these gates, the scenario register represents 000, 001, ..., 111 simultaneously.
    for q in scenario_qubits:
        qc.h(q)

    # For each good scenario, set target_qubit to 1.
    for scenario_index in good_indices:
        mark_good_scenario(qc, scenario_index, scenario_qubits, target_qubit)

    return qc, scenario_qubits, target_qubit, good_indices



def quantum_probability_from_statevector(path):
    '''
    Checks what target=1 probability is encoded by the quantum circuit.

    This is not QAE yet.
    It is a correctness check for the state_preparation circuit.

    If the path is good in 5 out of 8 scenarios, this function should return 0.625.
    Recall that each scenario has equal probability 0.125.
    '''

    # Build the state-preparation circuit.
    qc, scenario_qubits, target_qubit, good_indices = build_state_preparation_circuit(path)

    # Simulate the final state exactly as a statevector.
    state = Statevector.from_instruction(qc)

    # Sum probabilities of all states where target_qubit = 1.
    probability = 0.0

    for basis_index, amplitude in enumerate(state.data):
        # Qiskit uses little-endian convention:
        # shift the binary representation of basis_index right by target_qubit positions.
        target_value = (basis_index >> target_qubit) & 1

        if target_value == 1:
            probability += abs(amplitude) ** 2

    return probability, qc, good_indices



def quantum_probability_qae(path, eval_qubits=4):
    '''
    Runs Qiskit's amplitude estimation.

    QAE estimates the probability that after state preparation by circuit A,
    the target qubit is equal to 1.

    The eval_qubits parameter specifies how many additional evaluation qubits QAE uses.
    More qubits mean a more accurate grid of possible results, but also a larger circuit.
    '''

    # Build circuit A, which encodes good scenarios as target=1.
    state_preparation, scenario_qubits, target_qubit, good_indices = build_state_preparation_circuit(path)

    # EstimationProblem describes the problem for QAE:
    # - state_preparation: how to prepare the state,
    # - objective_qubits: which qubits represent success.
    # Here, success means target_qubit = 1.
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[target_qubit],
    )

    # StatevectorSampler performs a local statevector simulation.
    # Therefore, no real quantum hardware is needed.
    sampler = StatevectorSampler(seed=7)

    # Create the AmplitudeEstimation algorithm.
    # This is the canonical QAE version based on quantum phase estimation.
    ae = AmplitudeEstimation(
        num_eval_qubits=eval_qubits,
        sampler=sampler,
    )

    # Run amplitude estimation.
    result = ae.estimate(problem)

    # Additionally construct the QAE circuit so it can be inspected.
    # measurement=True adds measurements of the evaluation register.
    qae_circuit = ae.construct_circuit(problem, measurement=True)

    return result, state_preparation, qae_circuit, good_indices


# =====================================================================
# 6. HELPER FUNCTIONS FOR DISPLAY
# =====================================================================


def print_grid_with_path(path):
    '''
    Prints the 5 x 5 board with the path marked.

    S = start
    G = goal
    # = static obstacle
    P = path cell
    . = free cell
    '''

    path_set = set(path)

    for y in range(H):
        row = []
        for x in range(W):
            cell = (x, y)
            if cell == start:
                row.append("S")
            elif cell == goal:
                row.append("G")
            elif cell in static_obstacles:
                row.append("#")
            elif cell in path_set:
                row.append("P")
            else:
                row.append(".")
        print(" ".join(row))



def print_scenario_table(path):
    '''
    Prints a table showing the path result in each scenario.
    '''

    print("scenario | dynamic obstacles      | delay | safe | score")
    print("---------|------------------------|-------|------|------")

    for i, scenario in enumerate(scenarios):
        safe = path_safe(path, scenario)
        score = path_cost_score(path, scenario)
        print(f"{i:8d} | {str(scenario['dynamic']):22s} | {scenario['delay']:5d} | {safe:4d} | {score:.4f}")


# =====================================================================
# 7. EXAMPLE RUN
# =====================================================================

if __name__ == "__main__":
    # Set seeds so Monte Carlo results are reproducible.
    np.random.seed(7)
    random.seed(7)

    # First path: generated automatically by A*.
    path_a_star = astar()

    # Second path: written manually to compare it with the A* path.
    # This path goes along the top edge and then down the right column.
    path_manual = [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]

    # Compare both paths.
    paths = [
        ("A*", path_a_star),
        ("manual", path_manual),
    ]

    for path_name, path in paths:
        print("=" * 80)
        print(f"PATH: {path_name}")
        print("=" * 80)

        print("\nPath cell list:")
        print(path)

        print("\nGrid with path marked (P means path):")
        print_grid_with_path(path)

        print("\nScenario table:")
        print_scenario_table(path)

        print("\nClassical results:")
        print("exact safety probability:", exact_probability(path))
        print("exact expected score:", exact_expected_score(path))
        print("monte carlo safety probability:", monte_carlo_probability(path, shots=1000))
        print("monte carlo expected score:", monte_carlo_score(path, shots=1000))

        print("\nQuantum part: Statevector circuit check")
        sv_probability, state_prep_circuit, good_indices_sv = quantum_probability_from_statevector(path)
        print("good scenario indices:", sorted(good_indices_sv))
        print("statevector probability target=1:", sv_probability)

        # print("\nState preparation circuit A:")
        # print(state_prep_circuit.draw(output="text"))

        print("\nQuantum part: QAE with Qiskit")
        qae_result, state_prep, qae_circuit, good_indices_qae = quantum_probability_qae(path, eval_qubits=4)

        # qae estimation is the basic raw amplitude estimate returned by QAE.
        # With eval_qubits=4, the evaluation register has only 16 possible outcomes: 0, 1, ..., 15.
        # Therefore, this value lies on a relatively coarse grid of possible estimates.
        # It may be less accurate than qae_result.mle.
        print("qae estimation:", qae_result.estimation)

        # qae mle is the maximum likelihood estimation result.
        # It is computed from the full QAE measurement distribution, not only from one most frequent result.
        # In practice, this is usually a better approximation of the true success probability.
        # For this problem, it should be close to exact safety probability.
        print("qae mle:", qae_result.mle)

        # print("\nQAE circuit:")
        # print(qae_circuit.draw(output="text"))
        print()
