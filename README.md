# Santa's Workshop Tour 2019 - Optimization Solution

This project implements a Particle Swarm Optimization (PSO) algorithm to solve the Santa's Workshop Tour 2019 optimization problem from Kaggle. The goal is to assign families to workshop days while minimizing accounting penalties and satisfying daily occupancy constraints.

## Problem Description

The Santa's Workshop Tour 2019 problem involves:
- 5000 families to be assigned to 100 workshop days
- Each family has 10 preferred days (choices 0-9)
- Daily occupancy must be between 125 and 300 people
- Accounting penalties are incurred based on:
  - Choice satisfaction (lower choice number = lower penalty)
  - Family size
  - Day preferences

## Project Structure

```
IU-NIC-Assignment/
├── data/
│   └── family_data.csv     # Input data containing family preferences
├── experiments/            # Folder with generated plots for section `Experiments and Evaluation`
├── main.ipynb              # Main file
└── README.md               # This file
```

## Dependencies

The project requires the following Python packages:
- NumPy: For numerical computations and array operations
- Pandas: For data manipulation and analysis
- Matplotlib: For data visualization
- Seaborn: For enhanced statistical visualizations
- Numba: For JIT compilation and performance optimization
- Collections: For specialized container datatypes


## Data Exploration

The `data_exploration` section in main.ipynb performs comprehensive analysis of the dataset:

1. **Basic Dataset Overview**
   - Number of families and features
   - Sample data preview

2. **Family Size Analysis**
   - Distribution of family sizes
   - Statistical summary

3. **Preference Analysis**
   - Distribution of family preferences
   - Preference uniqueness
   - Common preference patterns

4. **Constraint Analysis**
   - Daily occupancy analysis
   - Constraint violations
   - First-choice scenario analysis


## Brute Force Approach

The project includes a brute-force implementation (`brute_force_schedule` function) that:
1. Generates all possible combinations of day assignments
2. Validates each combination against constraints
3. Calculates total cost (preference + accounting)
4. Keeps track of the best solution found

Key features:
- Demonstrates the problem's complexity (O(N^D) where N is days and D is families)
- Serves as a baseline for comparing PSO performance
- Limited to small problem instances (4-5 families) due to computational complexity
- Includes progress tracking and cost calculation

## PSO Implementation

The PSO algorithm is implemented with the following features:

1. **Particle Representation**
   - Each particle represents a complete assignment of families to days
   - Position updates maintain integer values (day assignments)

2. **Initialization Strategies**
   - First choice prioritization
   - Weighted random choices
   - Load balancing

3. **Fitness Function**
   - Calculates accounting penalties
   - Penalizes constraint violations
   - Considers family preferences

4. **Velocity and Position Updates**
   - Maintains integer values
   - Includes repair mechanism for constraint violations
   - Implements local search for refinement

Results of PSO algorithm are saved in `best_schedule.npy`

## Experiments and Evaluation

The `experiments` section evaluates the PSO algorithm through:

1. **Parameter Sensitivity Analysis**
   - Tests different particle counts
   - Varies iteration numbers
   - Analyzes convergence patterns

2. **Initialization Strategy Comparison**
   - Compares different initialization methods
   - Measures initial solution quality
   - Evaluates convergence speed

3. **Constraint Satisfaction Analysis**
   - Tracks constraint violations
   - Measures repair effectiveness
   - Analyzes solution feasibility

The section generates:
- `experiments/parameter_sensitivity_results.json`
- `experiments/parameter_sensitivity.png`
- `experiments/initialization_strategy_results.json`
- `experiments/initialization_strategy_comparison.png`
- `experiments/constraint_violations.json`
- `experiments/constraint_violations.png`

## Results

The PSO algorithm's performance is evaluated based on:
1. Solution quality (accounting penalties)
2. Constraint satisfaction
3. Computational efficiency
4. Convergence behavior