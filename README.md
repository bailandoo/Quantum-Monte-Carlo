# Monte Carlo and Quantum Amplitude Estimation for Drone Path Evaluation

This project demonstrates a simple hybrid classical-quantum workflow for evaluating drone paths under uncertainty. The drone moves on a 5x5 grid from a fixed start point to a target point while avoiding static obstacles. Additional uncertainty is introduced through several possible scenarios containing dynamic obstacles and flight delays.

The classical part of the project uses the A* algorithm to generate a path and evaluates its reliability using exact scenario enumeration and Monte Carlo sampling. Each path is tested against all uncertainty scenarios to estimate its safety probability and expected quality score.

The quantum part uses Qiskit to encode the uncertainty scenarios into a quantum circuit. Good scenarios are marked with an auxiliary target qubit, and the probability of measuring this qubit in state `1` represents the estimated path success probability. This probability is first verified using statevector simulation and then estimated with Quantum Amplitude Estimation.

The goal of the project is to compare classical Monte Carlo-based probability estimation with a quantum amplitude estimation approach in a small, interpretable path-planning example.

## Main components

- A* path planning on a 2D grid
- Static and dynamic obstacle modeling
- Exact probability and expected score calculation
- Classical Monte Carlo estimation
- Qiskit state-preparation circuit for scenario encoding
- Statevector validation of encoded probabilities
- Quantum Amplitude Estimation using Qiskit
