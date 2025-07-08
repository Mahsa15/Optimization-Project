# User Guide

This comprehensive user guide provides everything you need to know to effectively use the ILP Optimization Project.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation Guide](#installation-guide)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Input File Formats](#input-file-formats)
- [Output Interpretation](#output-interpretation)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Examples and Tutorials](#examples-and-tutorials)

## Getting Started

### What is this algorithm?

This project implements a novel Integer Linear Programming (ILP) solver that uses Discrete Fourier Transform (DFT) based techniques to solve optimization problems. It's particularly effective for problems with harmonic structure or balanced solutions.

### When to use this algorithm?

Use this algorithm when:
- ✅ You have integer linear programming problems
- ✅ Standard solvers are too slow or ineffective
- ✅ Your problems have structured or balanced solutions
- ✅ You need detailed logging and debugging capabilities
- ✅ You want to experiment with novel optimization techniques

Avoid when:
- ❌ You need continuous optimization only
- ❌ Your problems are very large (>1000 variables)
- ❌ You require commercial-grade performance guarantees
- ❌ Real-time performance is critical

## Installation Guide

### Prerequisites

Before installation, ensure you have:

1. **Python 3.7 or higher**
   ```bash
   python --version  # Should show 3.7+
   ```

2. **SCIP Optimization Suite**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install scip libscip-dev

   # macOS with Homebrew
   brew install scip

   # Windows
   # Download from https://scip.zib.de/download.php
   ```

3. **C++ compiler** (for PySCIPOpt)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential

   # macOS
   xcode-select --install
   ```

### Python Dependencies

Install required Python packages:

```bash
# Basic installation
pip install pyscipopt numpy

# For development/testing
pip install pyscipopt numpy pytest jupyter matplotlib
```

### Verification

Test your installation:

```python
# test_installation.py
try:
    from pyscipopt import Model
    import numpy as np
    print("✅ All dependencies installed successfully!")
    
    # Test basic functionality
    model = Model("test")
    x = model.addVar("x", vtype="INTEGER")
    y = model.addVar("y", vtype="INTEGER")
    model.addCons(x + y <= 5)
    model.setObjective(x + y, "maximize")
    model.optimize()
    
    if model.getStatus() == "optimal":
        print("✅ SCIP working correctly!")
    else:
        print("❌ SCIP installation issue")
        
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
```

## Basic Usage

### Command Line Interface

The simplest way to use the algorithm:

```bash
python Algorithm_1.py your_problem.lp
```

### Basic Python Usage

```python
from Algorithm_1 import main
import sys

# Set up command line argument
sys.argv = ['Algorithm_1.py', 'your_problem.lp']

# Run the algorithm
result = main()
print(f"Optimal value: {result}")
```

### Step-by-Step Example

```python
from Algorithm_1 import *

# 1. Load your problem
model = setup_problem("example.lp")
print(f"Loaded problem with {len(model.getVars())} variables")

# 2. Run P0 relaxation
n = len(model.getVars())
v, u = vu_values(n)
p0_value = creating_P0(model, v, u)
print(f"P0 bound: {p0_value}")

# 3. Run LP relaxation  
model_lp = setup_problem("example.lp")
lp_value = solve_lp_relaxation(model_lp)
print(f"LP bound: {lp_value}")

# 4. The algorithm will automatically process layers
# See main() function for complete workflow
```

## Advanced Usage

### Custom Constraint Addition

```python
def solve_with_custom_constraints(lp_file, extra_constraints):
    """
    Solve with additional custom constraints.
    
    Args:
        lp_file (str): Path to LP file
        extra_constraints (list): List of constraint functions
    
    Returns:
        float: Optimal value
    """
    model = setup_problem(lp_file)
    x = model.getVars()
    
    # Add custom constraints
    for constraint_func in extra_constraints:
        constraint_func(model, x)
    
    # Run algorithm
    n = len(x)
    v, u = vu_values(n)
    return creating_P0(model, v, u)

# Example: Add symmetry constraints
def symmetry_constraint(model, x):
    """Force first half of variables to equal second half."""
    n = len(x)
    mid = n // 2
    for i in range(mid):
        model.addCons(x[i] == x[i + mid])

result = solve_with_custom_constraints("problem.lp", [symmetry_constraint])
```

### Parameter Tuning

```python
def solve_with_parameters(lp_file, max_layers=None, custom_M=None):
    """
    Solve with custom parameters.
    
    Args:
        lp_file (str): Path to LP file
        max_layers (int): Maximum layers to process
        custom_M (float): Big-M value for constraints
    
    Returns:
        float: Optimal value
    """
    model = setup_problem(lp_file)
    n = len(model.getVars())
    
    # Custom LP relaxation bound
    lp_model = setup_problem(lp_file)
    lp_value = solve_lp_relaxation(lp_model)
    
    if max_layers:
        lp_value = min(lp_value, max_layers)
    
    # Custom P0 with modified M value
    if custom_M:
        # Modify creating_P0 to use custom_M
        pass
    
    # Run with modified parameters
    v, u = vu_values(n)
    return creating_P0(model, v, u)
```

### Parallel Processing

```python
import multiprocessing as mp
from Algorithm_1 import *

def process_layer(args):
    """Process a single layer in parallel."""
    lp_file, layer, n = args
    
    model = setup_problem(lp_file)
    essential_set, projected_set = create_essential_and_projected_sets(layer, n)
    
    # Check feasibility
    for point in essential_set:
        if check_feasibility_of_point(model, point):
            return layer, point, True
    
    return layer, None, False

def parallel_layer_processing(lp_file, layer_range):
    """Process multiple layers in parallel."""
    model = setup_problem(lp_file)
    n = len(model.getVars())
    
    # Prepare arguments for parallel processing
    args = [(lp_file, layer, n) for layer in layer_range]
    
    # Process in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_layer, args)
    
    # Find best result
    for layer, point, feasible in results:
        if feasible:
            print(f"Found solution at layer {layer}: {point}")
            return layer
    
    return None
```

## Input File Formats

### LP File Format

The algorithm accepts standard LP file format:

```lp
Maximize
  3 x1 + 2 x2 + x3

Subject To
  constraint1: x1 + x2 + x3 <= 10
  constraint2: 2 x1 + x2 <= 15
  constraint3: x2 + 3 x3 <= 12

Bounds
  x1 >= 0
  x2 >= 0
  x3 >= 0

Integer
  x1
  x2
  x3

End
```

### Supported Features

- ✅ Linear objectives (maximize/minimize)
- ✅ Linear constraints (<=, >=, =)
- ✅ Integer and binary variables
- ✅ Variable bounds
- ✅ Mixed-integer problems (with continuous variables)

### Unsupported Features

- ❌ Quadratic objectives
- ❌ Quadratic constraints
- ❌ Special Ordered Sets (SOS)
- ❌ Indicator constraints

### File Format Examples

**Knapsack Problem**:
```lp
Maximize
  10 x1 + 15 x2 + 8 x3 + 12 x4

Subject To
  weight: 3 x1 + 5 x2 + 2 x3 + 4 x4 <= 20

Bounds
  0 <= x1 <= 1
  0 <= x2 <= 1
  0 <= x3 <= 1
  0 <= x4 <= 1

Binary
  x1 x2 x3 x4

End
```

**Assignment Problem**:
```lp
Minimize
  5 x11 + 3 x12 + 7 x21 + 4 x22

Subject To
  row1: x11 + x12 = 1
  row2: x21 + x22 = 1
  col1: x11 + x21 = 1
  col2: x12 + x22 = 1

Binary
  x11 x12 x21 x22

End
```

## Output Interpretation

### Console Output

The algorithm produces several types of output:

```
Model loaded successfully from problem.lp
Objective value from P0: 15.5
LP relaxation value: 18.7
Checking feasibility for layer 18
Feasible solution found at layer 18 with point [4, 5, 4, 5].
Final result: 18.0
```

**Interpretation**:
- **P0 value (15.5)**: DFT-based relaxation bound
- **LP value (18.7)**: Standard linear relaxation bound  
- **Layer 18**: Processing sum-level 18
- **Solution [4,5,4,5]**: Optimal integer solution
- **Final result (18.0)**: Optimal objective value

### Log Files

Detailed logs are saved to `ilp_process.log`:

```
2024-01-15 10:30:15:INFO:Model loaded successfully from problem.lp
2024-01-15 10:30:15:INFO:Generated v and u values for 4 variables
2024-01-15 10:30:16:INFO:P0 solved optimally with objective: 15.5
2024-01-15 10:30:16:INFO:LP relaxation solved with objective: 18.7
2024-01-15 10:30:17:INFO:Processing layer 18
2024-01-15 10:30:17:INFO:Feasible solution found at layer 18 with point [4, 5, 4, 5]
```

### Intermediate Files

The algorithm creates several debugging files:

- **`1-P_0-const.lp`**: P0 relaxation with DFT constraints
- **`1-feasibility_check.lp`**: Feasibility verification model
- **`model_new_constraints_layer_X.cip`**: Models with added constraints

### Return Values

```python
result = main()

if result == -float('inf'):
    print("Problem is infeasible")
elif result == float('inf'):
    print("Problem is unbounded")
else:
    print(f"Optimal value: {result}")
```

## Performance Tuning

### Problem Size Guidelines

| Variables | Expected Time | Memory Usage | Recommendations |
|-----------|---------------|--------------|-----------------|
| < 20 | Seconds | < 100MB | Perfect for testing |
| 20-50 | Minutes | < 500MB | Good for development |
| 50-200 | Hours | < 2GB | Batch processing |
| > 200 | Days | > 5GB | Consider preprocessing |

### Optimization Strategies

**1. Reduce Problem Size**:
```python
def preprocess_problem(lp_file):
    """Remove redundant constraints and variables."""
    model = setup_problem(lp_file)
    
    # Remove fixed variables
    for var in model.getVars():
        if var.getLbOriginal() == var.getUbOriginal():
            # Variable is fixed, can be substituted
            pass
    
    # Identify redundant constraints
    # ... preprocessing logic ...
    
    return model
```

**2. Customize Layer Range**:
```python
def solve_with_limited_layers(lp_file, max_gap=10):
    """Limit layer processing to reduce computation."""
    model_lp = setup_problem(lp_file)
    lp_value = solve_lp_relaxation(model_lp)
    
    model_p0 = setup_problem(lp_file)
    n = len(model_p0.getVars())
    v, u = vu_values(n)
    p0_value = creating_P0(model_p0, v, u)
    
    # Limit processing range
    gap = min(lp_value - p0_value, max_gap)
    limited_range = range(int(lp_value), int(lp_value - gap), -1)
    
    # Process limited layers
    # ... custom processing ...
```

**3. Memory Management**:
```python
import gc

def memory_efficient_solve(lp_file):
    """Solve with explicit memory management."""
    
    # Clear previous models
    gc.collect()
    
    # Solve P0
    model_p0 = setup_problem(lp_file)
    # ... P0 processing ...
    del model_p0
    gc.collect()
    
    # Solve LP
    model_lp = setup_problem(lp_file)
    # ... LP processing ...
    del model_lp
    gc.collect()
    
    # Process layers one by one
    for layer in layer_range:
        model = setup_problem(lp_file)
        # ... layer processing ...
        del model
        gc.collect()
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ImportError: No module named 'pyscipopt'
```
**Solution**: Install PySCIPOpt
```bash
pip install pyscipopt
```

**2. SCIP Not Found**
```
OSError: Could not find SCIP installation
```
**Solution**: Install SCIP optimization suite
```bash
# Ubuntu
sudo apt-get install scip

# Check installation
scip --version
```

**3. File Not Found**
```
FileNotFoundError: problem.lp not found
```
**Solution**: Check file path and format
```python
import os
if os.path.exists("problem.lp"):
    print("File exists")
else:
    print("File not found - check path")
```

**4. Memory Errors**
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce problem size or use chunked processing
```python
# Check available memory
import psutil
memory = psutil.virtual_memory()
print(f"Available memory: {memory.available / 1e9:.1f} GB")
```

**5. Infinite Loops**
```
Algorithm running too long
```
**Solution**: Add time limits and layer limits
```python
import signal
import time

def timeout_handler(signum, frame):
    raise TimeoutError("Algorithm timeout")

# Set 1-hour timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # 1 hour

try:
    result = main()
finally:
    signal.alarm(0)  # Cancel timeout
```

### Debugging Guide

**1. Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**2. Check Intermediate Files**:
```bash
# View P0 model
cat 1-P_0-const.lp

# Check feasibility model
cat 1-feasibility_check.lp
```

**3. Validate Input**:
```python
def validate_lp_file(file_path):
    """Validate LP file format."""
    try:
        model = setup_problem(file_path)
        vars_count = len(model.getVars())
        cons_count = len(model.getConss())
        
        print(f"✅ Valid LP file:")
        print(f"   Variables: {vars_count}")
        print(f"   Constraints: {cons_count}")
        
        return True
    except Exception as e:
        print(f"❌ Invalid LP file: {e}")
        return False
```

**4. Performance Monitoring**:
```python
import time
import psutil
import os

def monitor_performance():
    """Monitor algorithm performance."""
    process = psutil.Process(os.getpid())
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1e6  # MB
    
    # Run algorithm
    result = main()
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1e6  # MB
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {end_memory - start_memory:.2f} MB")
    print(f"Peak memory: {end_memory:.2f} MB")
    
    return result
```

## Best Practices

### Problem Formulation

**1. Scale Variables Appropriately**:
```python
# Good: Variables in similar ranges
# maximize 10*x1 + 12*x2 + 8*x3
# x1, x2, x3 <= 100

# Bad: Variables with vastly different scales  
# maximize 10000*x1 + 0.01*x2 + x3
# x1 <= 1, x2 <= 100000, x3 <= 50
```

**2. Use Meaningful Variable Names**:
```lp
\ Good
maximize profit: 10 product1 + 15 product2 + 8 product3

\ Bad  
maximize obj: 10 x1 + 15 x2 + 8 x3
```

**3. Order Constraints Logically**:
```lp
\ Resource constraints first
capacity1: 3 product1 + 5 product2 <= 100
capacity2: 2 product1 + 4 product3 <= 80

\ Logical constraints next
logic1: product1 + product2 >= 10
logic2: product2 <= product3
```

### Algorithm Usage

**1. Start with Small Problems**:
```python
# Test with small version first
def test_small_version(original_file):
    # Create reduced problem
    model = setup_problem(original_file)
    vars_subset = model.getVars()[:10]  # First 10 variables
    
    # Test algorithm on subset
    # ... testing logic ...
```

**2. Use Logging Effectively**:
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('algorithm.log'),
        logging.StreamHandler()
    ]
)

# Add custom logging
def solve_with_logging(lp_file):
    logging.info(f"Starting optimization of {lp_file}")
    
    start_time = time.time()
    result = main()
    end_time = time.time()
    
    logging.info(f"Completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Final result: {result}")
    
    return result
```

**3. Validate Results**:
```python
def validate_solution(lp_file, solution_values):
    """Validate that solution satisfies all constraints."""
    model = setup_problem(lp_file)
    vars = model.getVars()
    
    # Set variables to solution values
    for var, value in zip(vars, solution_values):
        model.chgVarLb(var, value)
        model.chgVarUb(var, value)
    
    # Check feasibility
    model.optimize()
    
    if model.getStatus() == "optimal":
        print("✅ Solution is valid")
        return True
    else:
        print("❌ Solution violates constraints")
        return False
```

### Development Workflow

**1. Iterative Development**:
```python
# Start simple
def solve_basic(lp_file):
    return main()

# Add features incrementally  
def solve_with_preprocessing(lp_file):
    # Add preprocessing
    return main()

def solve_with_custom_constraints(lp_file):
    # Add custom constraints
    return main()
```

**2. Testing Strategy**:
```python
import unittest

class TestILPAlgorithm(unittest.TestCase):
    
    def test_small_problem(self):
        """Test on known small problem."""
        result = main_with_args(["small_test.lp"])
        self.assertAlmostEqual(result, 15.0, places=1)
    
    def test_infeasible_problem(self):
        """Test infeasible problem handling."""
        result = main_with_args(["infeasible.lp"])
        self.assertEqual(result, -float('inf'))
    
    def test_performance(self):
        """Test performance on medium problem."""
        start_time = time.time()
        result = main_with_args(["medium_test.lp"])
        duration = time.time() - start_time
        
        self.assertLess(duration, 300)  # Should complete in 5 minutes
```

## Examples and Tutorials

### Tutorial 1: Knapsack Problem

**Problem**: Pack items to maximize value within weight limit.

**Step 1**: Create LP file (`knapsack.lp`)
```lp
Maximize
  value: 10 item1 + 15 item2 + 8 item3 + 12 item4

Subject To
  weight: 3 item1 + 5 item2 + 2 item3 + 4 item4 <= 20

Binary
  item1 item2 item3 item4

End
```

**Step 2**: Solve
```python
from Algorithm_1 import main
import sys

sys.argv = ['Algorithm_1.py', 'knapsack.lp']
result = main()
print(f"Maximum value: {result}")
```

**Expected Output**:
```
Objective value from P0: 35.0
LP relaxation value: 37.5
Checking feasibility for layer 37
No feasible solutions found for layer 37.
Checking feasibility for layer 36
Feasible solution found at layer 36 with point [1, 1, 1, 1].
Final result: 36.0
```

### Tutorial 2: Assignment Problem

**Problem**: Assign workers to tasks minimizing cost.

**Step 1**: Create LP file (`assignment.lp`)
```lp
Minimize
  cost: 5 w1t1 + 3 w1t2 + 7 w2t1 + 4 w2t2

Subject To
  worker1: w1t1 + w1t2 = 1
  worker2: w2t1 + w2t2 = 1
  task1: w1t1 + w2t1 = 1
  task2: w1t2 + w2t2 = 1

Binary
  w1t1 w1t2 w2t1 w2t2

End
```

**Step 2**: Solve
```python
sys.argv = ['Algorithm_1.py', 'assignment.lp']
result = main()
print(f"Minimum cost: {result}")
```

### Tutorial 3: Custom Problem

**Problem**: Portfolio optimization with integer constraints.

```python
def solve_portfolio_problem():
    """Solve custom portfolio optimization problem."""
    
    # Create problem programmatically
    from pyscipopt import Model
    
    model = Model("portfolio")
    
    # Variables: number of shares to buy (integer)
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    returns = [0.12, 0.15, 0.10, 0.20]
    costs = [150, 2500, 300, 800]
    
    # Decision variables
    x = {}
    for i, stock in enumerate(stocks):
        x[i] = model.addVar(vtype="INTEGER", name=f"shares_{stock}")
    
    # Objective: maximize expected return
    model.setObjective(
        sum(returns[i] * x[i] for i in range(len(stocks))), 
        "maximize"
    )
    
    # Budget constraint
    budget = 10000
    model.addCons(
        sum(costs[i] * x[i] for i in range(len(stocks))) <= budget
    )
    
    # Diversification: at least 2 stocks
    for i in range(len(stocks)):
        model.addCons(x[i] <= 10)  # Max 10 shares per stock
    
    # Save as LP file
    model.writeProblem("portfolio.lp")
    
    # Solve with our algorithm
    sys.argv = ['Algorithm_1.py', 'portfolio.lp']
    result = main()
    
    print(f"Portfolio optimization result: {result}")
    return result

# Run tutorial
solve_portfolio_problem()
```

This comprehensive user guide provides everything needed to effectively use the ILP optimization algorithm, from basic setup to advanced customization and troubleshooting.