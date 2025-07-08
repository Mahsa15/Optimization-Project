# ILP Optimization Project

A sophisticated Integer Linear Programming (ILP) solver that uses Discrete Fourier Transform (DFT) based methods to solve optimization problems. This implementation combines theoretical foundations from harmonic analysis with practical optimization techniques.

## Overview

This project implements an advanced ILP algorithm that:
- Uses DFT-based projections to create cutting planes
- Implements layer-by-layer feasibility checking
- Combines P0 relaxation with LP relaxation techniques
- Provides comprehensive logging and debugging capabilities

## Features

- **DFT-Based Optimization**: Uses trigonometric projections for constraint generation
- **Multi-Layer Processing**: Systematic layer-by-layer solution approach
- **Robust Feasibility Checking**: Multiple feasibility verification methods
- **Comprehensive Logging**: Detailed execution logs for debugging and analysis
- **LP File Integration**: Direct support for standard LP file formats

## Requirements

- Python 3.7+
- PySCIPOpt
- NumPy
- SCIP Optimization Suite

## Installation

1. Install SCIP Optimization Suite:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install scip

   # On macOS with Homebrew
   brew install scip
   ```

2. Install Python dependencies:
   ```bash
   pip install pyscipopt numpy
   ```

## Quick Start

### Basic Usage

```python
# Run the algorithm on an LP file
python Algorithm_1.py path/to/your/problem.lp
```

### Example

```bash
# Example with a sample LP file
python Algorithm_1.py example_problem.lp
```

The algorithm will:
1. Load the LP problem
2. Solve P0 relaxation with DFT constraints
3. Solve standard LP relaxation
4. Process layers systematically
5. Output the optimal solution

## Output

The algorithm produces:
- **Console Output**: Real-time progress and results
- **Log Files**: Detailed execution logs in `ilp_process.log`
- **Intermediate Files**: LP files for debugging (`1-P_0-const.lp`, etc.)

## Algorithm Overview

1. **P0 Phase**: Creates DFT-based constraints using trigonometric projections
2. **LP Relaxation**: Solves continuous relaxation for bounds
3. **Layer Processing**: Systematically checks feasibility across discrete layers
4. **Constraint Addition**: Dynamically adds cutting plane constraints
5. **Optimization**: Returns the best feasible solution found

## Project Structure

```
.
├── Algorithm_1.py          # Main algorithm implementation
├── README.md              # This file
├── API_DOCUMENTATION.md   # Detailed API documentation
└── ilp_process.log       # Generated log file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of an optimization research initiative. Please refer to the institutional guidelines for usage and distribution.

## Support

For questions and support, please check the API documentation and log files for debugging information.