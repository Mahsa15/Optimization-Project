# API Documentation

This document provides comprehensive documentation for all public APIs, functions, and components in the ILP Optimization Project.

## Table of Contents

- [Core Functions](#core-functions)
- [Utility Functions](#utility-functions)
- [Constraint Functions](#constraint-functions)
- [Optimization Functions](#optimization-functions)
- [Main Entry Point](#main-entry-point)
- [Usage Examples](#usage-examples)

## Core Functions

### `setup_problem(file_path)`

**Description**: Initializes and loads an ILP model from an LP file.

**Parameters**:
- `file_path` (str): Path to the LP file containing the optimization problem

**Returns**:
- `Model`: PySCIPOpt Model object loaded with the problem data

**Example**:
```python
from Algorithm_1 import setup_problem

# Load a problem from an LP file
model = setup_problem("example.lp")
print(f"Model has {len(model.getVars())} variables")
```

**Error Handling**:
- Logs errors if file cannot be read
- Returns model object even on failure (for graceful handling)

---

### `vu_values(n)`

**Description**: Generates discrete Fourier transform coefficient arrays for n variables.

**Parameters**:
- `n` (int): Number of variables in the problem

**Returns**:
- `tuple`: (v, u) where:
  - `v` (list): Cosine coefficients for DFT projections
  - `u` (list): Sine coefficients for DFT projections

**Mathematical Foundation**:
- `v[j][i] = cos(2π * i * j / n)`
- `u[j][i] = sin(2π * i * j / n)`

**Example**:
```python
from Algorithm_1 import vu_values

# Generate DFT coefficients for 4 variables
v, u = vu_values(4)
print(f"v coefficients: {v}")
print(f"u coefficients: {u}")

# Output:
# v coefficients: [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, -1.0, 0.0]]
# u coefficients: [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]]
```

---

### `create_pv_pu(model, v, u, x, j)`

**Description**: Creates projection expressions for the j-th frequency component.

**Parameters**:
- `model` (Model): PySCIPOpt model object
- `v` (list): Cosine coefficients from `vu_values()`
- `u` (list): Sine coefficients from `vu_values()`
- `x` (list): Decision variables
- `j` (int): Frequency index

**Returns**:
- `tuple`: (p_v, p_u) linear expressions for real and imaginary parts

**Example**:
```python
from Algorithm_1 import setup_problem, vu_values, create_pv_pu

model = setup_problem("problem.lp")
x = model.getVars()
v, u = vu_values(len(x))

# Create projections for frequency j=1
p_v, p_u = create_pv_pu(model, v, u, x, 1)
```

## Constraint Functions

### `creating_P0(model, v, u)`

**Description**: Creates P0 relaxation constraints using DFT-based projections and solves the relaxed problem.

**Parameters**:
- `model` (Model): PySCIPOpt model object
- `v` (list): Cosine coefficients
- `u` (list): Sine coefficients

**Returns**:
- `float`: Objective value of P0 relaxation, or -∞ if infeasible

**Algorithm**:
1. Creates binary variables r_j for each frequency component
2. Adds constraints based on frequency domain projections
3. Limits the number of active frequency components
4. Solves the resulting optimization problem

**Example**:
```python
from Algorithm_1 import setup_problem, vu_values, creating_P0

model = setup_problem("problem.lp")
v, u = vu_values(len(model.getVars()))

# Solve P0 relaxation
p0_value = creating_P0(model, v, u)
print(f"P0 objective value: {p0_value}")
```

**Output Files**:
- Creates `1-P_0-const.lp` with the P0 constraints

---

### `compute_T_k(n, x, k, model)`

**Description**: Computes T_k coefficients for constraint generation.

**Parameters**:
- `n` (int): Number of variables
- `x` (list): Decision variables
- `k` (int): Shift parameter
- `model` (Model): PySCIPOpt model object

**Returns**:
- `Expression`: T_k value as a PySCIPOpt expression

**Mathematical Foundation**:
- Computes shifted DFT projections
- Uses circular shift operations on trigonometric basis

**Example**:
```python
from Algorithm_1 import setup_problem, compute_T_k

model = setup_problem("problem.lp")
x = model.getVars()
n = len(x)

# Compute T_k for k=0
t_0 = compute_T_k(n, x, 0, model)
```

## Utility Functions

### `solve_lp_relaxation(model)`

**Description**: Solves the linear programming relaxation of the ILP problem.

**Parameters**:
- `model` (Model): PySCIPOpt model object

**Returns**:
- `float`: Optimal objective value of LP relaxation, or -∞ if infeasible

**Process**:
1. Converts all integer/binary variables to continuous
2. Solves the relaxed problem
3. Returns optimal value

**Example**:
```python
from Algorithm_1 import setup_problem, solve_lp_relaxation

model = setup_problem("problem.lp")
lp_value = solve_lp_relaxation(model)
print(f"LP relaxation value: {lp_value}")
```

---

### `create_essential_and_projected_sets(k, n)`

**Description**: Generates essential and projected point sets for layer k.

**Parameters**:
- `k` (int): Layer value (sum of variables)
- `n` (int): Number of variables

**Returns**:
- `tuple`: (essential_set, projected_set) where:
  - `essential_set` (list): Points with sum equal to k
  - `projected_set` (list): Projected points for constraint generation

**Algorithm**:
1. Creates uniform distribution base point
2. Distributes remainder randomly
3. Projects points for constraint generation

**Example**:
```python
from Algorithm_1 import create_essential_and_projected_sets

# Create sets for layer 10 with 4 variables
essential, projected = create_essential_and_projected_sets(10, 4)
print(f"Essential set: {essential}")
print(f"Projected set: {projected}")

# Example output:
# Essential set: [[2, 3, 2, 3]]
# Projected set: [[0, 1, 0, 1]]
```

---

### `check_feasibility_of_point(model, point)`

**Description**: Checks if a specific point is feasible for the original problem.

**Parameters**:
- `model` (Model): PySCIPOpt model object
- `point` (list): Point to check (values for each variable)

**Returns**:
- `bool`: True if point is feasible, False otherwise

**Process**:
1. Fixes variables to point values
2. Solves the constrained problem
3. Verifies feasibility and objective value

**Example**:
```python
from Algorithm_1 import setup_problem, check_feasibility_of_point

model = setup_problem("problem.lp")
point = [1, 2, 1, 3]  # Example point

is_feasible = check_feasibility_of_point(model, point)
print(f"Point {point} is feasible: {is_feasible}")
```

**Output Files**:
- Creates `1-feasibility_check.lp` for debugging

## Optimization Functions

### `feasibility_check_by_layer(model, n, l, l_prime)`

**Description**: Systematically checks feasibility across multiple layers.

**Parameters**:
- `model` (Model): PySCIPOpt model object
- `n` (int): Number of variables
- `l` (int): Upper layer bound
- `l_prime` (int): Lower layer bound

**Returns**:
- `bool`: True if feasible solution found, False otherwise

**Algorithm**:
- Iterates from layer l down to l_prime
- Checks essential set points for each layer
- Returns on first feasible solution found

**Example**:
```python
from Algorithm_1 import setup_problem, feasibility_check_by_layer

model = setup_problem("problem.lp")
n = len(model.getVars())

# Check layers from 20 down to 16
found = feasibility_check_by_layer(model, n, 20, 16)
print(f"Feasible solution found: {found}")
```

---

### `add_constraints_and_optimize(model, x, n, projected_set, layer)`

**Description**: Adds cutting plane constraints based on projected set and optimizes.

**Parameters**:
- `model` (Model): PySCIPOpt model object
- `x` (list): Decision variables
- `n` (int): Number of variables
- `projected_set` (list): Projected points for constraint generation
- `layer` (int): Current layer being processed

**Returns**:
- `bool`: True if optimal solution found, False otherwise

**Process**:
1. Generates T_k based constraints for each projected point
2. Adds constraints to model
3. Optimizes and returns results

**Example**:
```python
from Algorithm_1 import setup_problem, create_essential_and_projected_sets, add_constraints_and_optimize

model = setup_problem("problem.lp")
x = model.getVars()
n = len(x)
_, projected_set = create_essential_and_projected_sets(15, n)

# Add constraints and optimize
success = add_constraints_and_optimize(model, x, n, projected_set, 15)
print(f"Optimization successful: {success}")
```

**Output Files**:
- Creates `model_new_constraints_layer_{layer}.cip` for debugging

## Main Entry Point

### `main()`

**Description**: Main algorithm entry point that orchestrates the entire solution process.

**Parameters**: None (reads from command line arguments)

**Returns**:
- `float`: Final optimal objective value

**Algorithm Flow**:
1. **Setup**: Load problem from command line argument
2. **P0 Phase**: Solve P0 relaxation with DFT constraints
3. **LP Phase**: Solve standard LP relaxation
4. **Layer Processing**: Iterate through layers checking feasibility
5. **Constraint Addition**: Add cutting planes when needed
6. **Result**: Return best solution found

**Command Line Usage**:
```bash
python Algorithm_1.py problem.lp
```

**Example Output**:
```
Objective value from P0: 15.5
LP relaxation value: 18.7
Checking feasibility for layer 18
Feasible solution found at layer 18 with point [4, 5, 4, 5].
Final result: 18.0
```

## Usage Examples

### Complete Workflow Example

```python
#!/usr/bin/env python3
"""
Example script demonstrating the complete ILP algorithm workflow.
"""

import sys
from Algorithm_1 import *

def solve_ilp_problem(lp_file_path):
    """
    Solve an ILP problem using the complete algorithm.
    
    Args:
        lp_file_path (str): Path to LP file
    
    Returns:
        float: Optimal objective value
    """
    
    print(f"Solving ILP problem: {lp_file_path}")
    
    # Phase 1: P0 Relaxation
    print("\n=== Phase 1: P0 Relaxation ===")
    model_p0 = setup_problem(lp_file_path)
    n = len(model_p0.getVars())
    v, u = vu_values(n)
    p0_value = creating_P0(model_p0, v, u)
    print(f"P0 objective value: {p0_value}")
    
    # Phase 2: LP Relaxation
    print("\n=== Phase 2: LP Relaxation ===")
    model_lp = setup_problem(lp_file_path)
    lp_value = solve_lp_relaxation(model_lp)
    print(f"LP relaxation value: {lp_value}")
    
    if lp_value == -float('inf'):
        print("Problem is infeasible")
        return max(p0_value, lp_value)
    
    # Phase 3: Layer Processing
    print("\n=== Phase 3: Layer Processing ===")
    l_prime = (int(lp_value) // n) * n
    
    for layer in range(int(lp_value), l_prime - 1, -1):
        print(f"\nProcessing layer {layer}")
        
        # Check feasibility
        model_feasibility = setup_problem(lp_file_path)
        essential_set, projected_set = create_essential_and_projected_sets(layer, n)
        
        # Try feasibility check first
        feasible_found = False
        for point in essential_set:
            if check_feasibility_of_point(model_feasibility, point):
                print(f"Feasible solution found: {point}")
                return max(p0_value, model_feasibility.getObjVal())
        
        # If no direct feasible solution, add constraints
        model_constraints = setup_problem(lp_file_path)
        x = model_constraints.getVars()
        
        if add_constraints_and_optimize(model_constraints, x, n, projected_set, layer):
            print(f"Solution found with constraints at layer {layer}")
            return max(p0_value, model_constraints.getObjVal())
    
    print("No feasible solution found")
    return max(p0_value, -float('inf'))

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example.py problem.lp")
        sys.exit(1)
    
    result = solve_ilp_problem(sys.argv[1])
    print(f"\nFinal optimal value: {result}")
```

### Custom Constraint Example

```python
from Algorithm_1 import setup_problem, vu_values, creating_P0

def solve_with_custom_constraints(lp_file):
    """
    Example showing how to add custom constraints before solving.
    """
    model = setup_problem(lp_file)
    x = model.getVars()
    
    # Add custom constraint: sum of first half <= sum of second half
    n = len(x)
    mid = n // 2
    model.addCons(sum(x[:mid]) <= sum(x[mid:]))
    
    # Solve P0 with custom constraints
    v, u = vu_values(n)
    result = creating_P0(model, v, u)
    
    return result
```

### Debugging and Analysis Example

```python
import logging
from Algorithm_1 import *

def analyze_problem_structure(lp_file):
    """
    Analyze the structure of an ILP problem.
    """
    model = setup_problem(lp_file)
    x = model.getVars()
    n = len(x)
    
    print(f"Problem Analysis for {lp_file}")
    print(f"Number of variables: {n}")
    print(f"Number of constraints: {len(model.getConss())}")
    
    # Analyze DFT structure
    v, u = vu_values(n)
    print(f"DFT frequencies used: {len(v)}")
    
    # Check LP bounds
    lp_value = solve_lp_relaxation(setup_problem(lp_file))
    print(f"LP relaxation bound: {lp_value}")
    
    # Estimate layer range
    if lp_value != -float('inf'):
        l_prime = (int(lp_value) // n) * n
        print(f"Layer processing range: {int(lp_value)} down to {l_prime}")
        print(f"Estimated layers to process: {int(lp_value) - l_prime + 1}")

# Example usage
analyze_problem_structure("example.lp")
```

## Error Handling and Debugging

### Common Issues and Solutions

1. **File Not Found**:
   ```python
   try:
       model = setup_problem("nonexistent.lp")
   except Exception as e:
       print(f"Failed to load problem: {e}")
   ```

2. **Infeasible Problems**:
   ```python
   lp_value = solve_lp_relaxation(model)
   if lp_value == -float('inf'):
       print("Problem is infeasible or unbounded")
   ```

3. **Memory Issues with Large Problems**:
   - Monitor log files for memory usage
   - Consider problem preprocessing
   - Use smaller layer ranges

### Log Analysis

The algorithm produces detailed logs in `ilp_process.log`. Key log patterns:

- `Model loaded successfully`: Problem setup OK
- `P0 solved optimally`: P0 phase completed
- `LP relaxation solved`: LP bound computed
- `Feasible solution found`: Solution discovered
- `No optimal solution found`: Layer failed

## Performance Notes

- **Problem Size**: Algorithm scales with O(n²) for DFT computations
- **Layer Range**: Performance depends on LP-P0 gap
- **Memory Usage**: Large problems may need disk-based intermediate files
- **Convergence**: Algorithm terminates when feasible solution found or all layers exhausted

## Integration Examples

### With Other Optimization Libraries

```python
import gurobipy as gp
from Algorithm_1 import vu_values, create_essential_and_projected_sets

def convert_to_gurobi(scip_model):
    """Convert SCIP model to Gurobi for comparison."""
    # Implementation depends on specific needs
    pass

def hybrid_solver(lp_file):
    """Use our algorithm with Gurobi for verification."""
    # Solve with our algorithm
    result_ours = main()
    
    # Verify with Gurobi
    # ... verification code ...
    
    return result_ours
```

### With Machine Learning

```python
import numpy as np
from sklearn.cluster import KMeans
from Algorithm_1 import create_essential_and_projected_sets

def ml_guided_layer_selection(n, max_layer):
    """Use ML to prioritize promising layers."""
    layers = list(range(max_layer, 0, -1))
    
    # Generate features for each layer
    features = []
    for layer in layers:
        essential, projected = create_essential_and_projected_sets(layer, n)
        # Extract features (variance, entropy, etc.)
        variance = np.var([sum(p) for p in essential])
        features.append([layer, variance])
    
    # Cluster and prioritize
    features = np.array(features)
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(features)
    
    # Return prioritized layer order
    return sorted(layers, key=lambda l: clusters[layers.index(l)])
```

This comprehensive API documentation covers all public functions, their usage, parameters, return values, and practical examples. Each function is documented with mathematical foundations where relevant, error handling considerations, and integration possibilities.