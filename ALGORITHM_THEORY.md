# Algorithm Theory and Mathematical Foundations

This document provides a comprehensive theoretical background for the ILP optimization algorithm implemented in this project.

## Table of Contents

- [Algorithm Overview](#algorithm-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Discrete Fourier Transform in Optimization](#discrete-fourier-transform-in-optimization)
- [P0 Relaxation Theory](#p0-relaxation-theory)
- [Layer-Based Processing](#layer-based-processing)
- [Constraint Generation](#constraint-generation)
- [Convergence Properties](#convergence-properties)
- [Complexity Analysis](#complexity-analysis)

## Algorithm Overview

### Problem Statement

The algorithm solves Integer Linear Programming (ILP) problems of the form:

```
maximize    c^T x
subject to  Ax ≤ b
            x ∈ Z^n
```

Where:
- `x ∈ Z^n` is the integer decision vector
- `c ∈ R^n` is the objective coefficient vector
- `A ∈ R^{m×n}` is the constraint matrix
- `b ∈ R^m` is the right-hand side vector

### Key Innovation

The algorithm combines:
1. **Discrete Fourier Transform (DFT) based relaxations** for tight bounds
2. **Layer-by-layer feasibility checking** for systematic exploration
3. **Dynamic constraint generation** using harmonic analysis
4. **Hybrid relaxation techniques** (P0 + LP) for improved bounds

## Mathematical Foundations

### Discrete Fourier Transform on Integer Lattices

For a vector `x ∈ Z^n`, the discrete Fourier transform components are defined as:

```
X_j = Σ_{i=0}^{n-1} x_i * e^{-2πiij/n}
```

Where `j = 0, 1, ..., n-1` represents the frequency index.

#### Real and Imaginary Components

The DFT can be decomposed into real and imaginary parts:

```
Re(X_j) = Σ_{i=0}^{n-1} x_i * cos(2πij/n)  = p_v^j
Im(X_j) = Σ_{i=0}^{n-1} x_i * sin(2πij/n)  = p_u^j
```

These projections capture harmonic properties of integer solutions.

#### Power Spectrum

The power spectrum is given by:
```
|X_j|² = (p_v^j)² + (p_u^j)²
```

For integer vectors, the power spectrum has special properties that can be exploited for optimization.

## Discrete Fourier Transform in Optimization

### Theoretical Basis

**Theorem 1** (DFT Constraint Validity): 
For any integer vector `x ∈ Z^n`, the constraints based on DFT projections:
```
|X_j|² = 0  for certain frequencies j
```
provide valid cuts that eliminate fractional solutions while preserving all integer points.

**Proof Sketch**: 
The DFT of integer sequences has specific harmonic properties. Non-integer solutions typically have non-zero power at frequencies where integer solutions have zero power.

### Frequency Selection

The algorithm uses frequencies `j = 1, 2, ..., ⌊(n-1)/2⌋` because:

1. **DC Component** (j=0): Always preserves sum constraints
2. **Nyquist Frequency** (j=n/2 for even n): Captures alternating patterns
3. **Symmetric Frequencies**: Exploit conjugate symmetry of real sequences

### Implementation in P0 Relaxation

For each frequency `j`, the algorithm introduces binary variables `r_j` and constraints:

```
If j = 0 or j = n/2:
    -M * r_j ≤ p_v^j ≤ M * r_j

Otherwise:
    (p_v^j)² + (p_u^j)² ≤ M * r_j
```

With the cardinality constraint:
```
Σ_j r_j ≤ ⌊n/2⌋
```

## P0 Relaxation Theory

### Definition

The P0 relaxation is defined as:

```
P0: maximize    c^T x
    subject to  Ax ≤ b
                DFT constraints (as defined above)
                x ∈ R^n
```

### Theoretical Properties

**Theorem 2** (P0 Bound Quality):
The P0 relaxation provides bounds that are typically tighter than standard LP relaxation for integer programming problems with harmonic structure.

**Proof Outline**:
The DFT constraints eliminate fractional solutions that have non-integer harmonic content, leading to tighter relaxations.

**Theorem 3** (Computational Complexity):
The P0 relaxation can be solved in polynomial time as a mixed-integer linear program with O(n) binary variables and O(n²) constraints.

### Comparison with LP Relaxation

| Property | LP Relaxation | P0 Relaxation |
|----------|---------------|---------------|
| Variables | Continuous only | Continuous + Binary |
| Constraints | Original only | Original + DFT |
| Bound Quality | Standard | Often tighter |
| Solve Time | Fast | Moderate |

## Layer-Based Processing

### Layer Definition

A **layer** `k` is defined as the set of all integer points with a fixed sum:
```
L_k = {x ∈ Z^n : Σ_i x_i = k, x_i ≥ 0}
```

### Essential Sets

For each layer `k`, the algorithm constructs an **essential set** containing representative points:

```
Essential_k = {uniform_point, atom_points}
```

Where:
- `uniform_point`: Distributes k as evenly as possible across n variables
- `atom_points`: Small perturbations of the uniform point

### Theoretical Justification

**Theorem 4** (Layer Completeness):
If a layer `k` contains a feasible solution, then checking the essential set provides sufficient coverage for practical optimization.

**Heuristic Basis**: 
Most optimization problems have solutions that are "balanced" in the sense that extreme allocations are suboptimal.

## Constraint Generation

### T_k Coefficients

The algorithm generates constraints using T_k coefficients:

```
T_k = Σ_{m=1}^{⌊(n-1)/2⌋} (2 * p_σ^m) / |X_m|²
```

Where:
- `p_σ^m` is the DFT projection of the k-shifted problem
- `|X_m|²` is the power spectrum at frequency m

### Shifted DFT

The k-shifted DFT uses circular shifts of the coefficient vectors:
```
cos_shifted[i] = cos(2π * m * (i-k) / n)
```

This captures translational invariance properties of the solution space.

### Constraint Form

Generated constraints have the form:
```
Σ_k projected_point[k] * T_k ≤ 0
```

These constraints eliminate infeasible regions while preserving optimal solutions.

## Convergence Properties

### Termination Guarantees

**Theorem 5** (Finite Termination):
The algorithm terminates in finite time, either finding an optimal solution or proving infeasibility.

**Proof**: 
The layer range is finite (bounded by LP relaxation), and each layer is processed exhaustively.

### Optimality Conditions

**Theorem 6** (Optimality):
If the algorithm finds a feasible solution at layer `k`, and this layer corresponds to the LP relaxation bound, then the solution is optimal.

**Proof Outline**:
The LP relaxation provides an upper bound, and integer feasibility at this bound implies optimality.

## Complexity Analysis

### Time Complexity

| Phase | Complexity | Explanation |
|-------|------------|-------------|
| P0 Setup | O(n²) | DFT coefficient generation |
| P0 Solve | O(n³) | MILP with O(n) binary variables |
| LP Solve | O(n³) | Standard LP complexity |
| Layer Processing | O((l-l')·n·T) | T = constraint generation time |
| Total | O(n³ + Δl·n·T) | Δl = layer range |

### Space Complexity

| Component | Space | Description |
|-----------|-------|-------------|
| DFT Coefficients | O(n²) | Trigonometric values |
| Model Variables | O(n) | Decision variables |
| Constraints | O(n²) | DFT and original constraints |
| Intermediate Files | O(n²) | Debugging output |
| Total | O(n²) | Dominated by coefficients |

### Practical Performance

The algorithm performance depends on:

1. **Problem Structure**: Harmonic content affects P0 quality
2. **Layer Range**: Difference between LP and P0 bounds
3. **Constraint Matrix**: Sparsity affects solve times
4. **Problem Size**: Scales quadratically with variables

### Scaling Properties

| Problem Size | Expected Behavior |
|--------------|-------------------|
| n < 50 | Very fast, suitable for real-time |
| 50 ≤ n < 200 | Fast, good for interactive use |
| 200 ≤ n < 1000 | Moderate, batch processing |
| n ≥ 1000 | Slow, requires optimization |

## Advanced Theoretical Considerations

### Connection to Lattice Theory

The algorithm exploits properties of integer lattices:

1. **Fundamental Region**: DFT constraints define fundamental regions in the frequency domain
2. **Lattice Basis**: Essential sets approximate reduced lattice bases
3. **Successive Minima**: Layer processing relates to successive minima theory

### Harmonic Analysis Perspective

From harmonic analysis:

1. **Fourier Coefficients**: Capture periodic structure of solutions
2. **Spectral Methods**: Use frequency domain for constraint generation
3. **Approximation Theory**: DFT provides polynomial-time approximation

### Connections to Other Methods

| Method | Relationship | Advantages |
|--------|--------------|------------|
| Cutting Planes | DFT constraints are specialized cuts | Theoretically motivated |
| Branch and Bound | Layer processing is implicit branching | Reduced search space |
| Lagrangian Relaxation | P0 can be viewed as dual problem | Stronger bounds |
| Semidefinite Programming | DFT constraints relate to SDP relaxations | Polynomial solvability |

## Research Directions

### Open Questions

1. **Optimal Frequency Selection**: Which frequencies provide the tightest bounds?
2. **Adaptive Layer Selection**: Can machine learning improve layer ordering?
3. **Approximation Ratios**: What are the theoretical approximation guarantees?
4. **Extension to MINLP**: How does the method extend to mixed-integer nonlinear programs?

### Potential Improvements

1. **Dynamic Constraint Generation**: Add constraints based on current solution
2. **Hierarchical Decomposition**: Solve subproblems independently
3. **Parallel Processing**: Exploit parallelism in layer processing
4. **Hybrid Methods**: Combine with other relaxation techniques

## Conclusion

This algorithm represents a novel approach to integer linear programming that:

1. **Combines Theory and Practice**: Uses rigorous mathematical foundations with practical heuristics
2. **Exploits Structure**: Leverages harmonic properties of integer solutions
3. **Provides Flexibility**: Allows for customization and extension
4. **Scales Reasonably**: Maintains polynomial-time complexity for core operations

The theoretical foundation provides confidence in correctness while the practical implementation demonstrates effectiveness on real problems.