# 🧮 BlendedSolvers

**BlendedSolvers** is a compact research library introducing an **experimental type of implicit ODE integrators** based on local stiffness rather than changing the timestep.

Each solver blends two fully implicit integrators (e.g., BDF, Gauss–Legendre, Radau, Lobatto, ESDIRK) using a smooth stiffness-dependent weight function.  
All codes are self-contained, transparent, and written for research clarity.

---

## 🌟 Concept

Classical ODE solvers adapt **step size** \(h\) to handle stiffness.  
**BlendedSolvers** instead adapt **method form**:

\[
a(\sigma) = \frac{\sigma^p}{1+\sigma^p}, \qquad
y_{n+1} = (1-a)\,\Phi_A(y_n,h) + a\,\Phi_B(y_n,h)
\]

where  
- \( \Phi_A, \Phi_B \) are complete implicit update operators,  
- \( \sigma \) is a stiffness proxy computed from local \(f\) and \(y\) differences,  
- \( a(\sigma) \) smoothly varies from 0 → 1 as stiffness increases.

This creates a continuous family of solvers interpolating between *non-stiff* and *stiff* regimes — a new axis of adaptivity orthogonal to step-size control.

---

## ⚙️ Implemented Solvers

Each solver resides in its own folder.  
Only **`blended9-10`** includes sub-folders (`gl5/`, `radau5/`) containing the Butcher-tableau coefficients `(A, b, c)`.

```
BlendedSolvers/
│
├── blended2/
│   └── blended2.py              # Gauss–Legendre 1 × BDF2
│
├── blended3/
│   └── blended3.py              # ESDIRK4(3) × Adams–Moulton 3
│
├── blended4/
│   └── blended4.py              # BDF4 × Lobatto IIIC(3)
│
├── blended9-10/
│   ├── blended9-10.py           # Radau IIA(5) × Gauss–Legendre(5)
│   ├── gl5/                     # Gauss–Legendre 5-stage coefficients
│   │   ├── gauss_legendre_s5_A.txt
│   │   ├── gauss_legendre_s5_b.txt
│   │   └── gauss_legendre_s5_c.txt
|   |   └── gauss_legendre_s5_triplets.txt
│   └── radau5/                  # Radau IIA 5-stage coefficients
│       ├── radau_s5_A.txt
│       ├── radau_s5_b.txt
│       └── radau_s5_c.txt
|       └── radau_s5_triplets.txt
│
└── blendedTRBDFC/
    └── blendedTRBDFC.py         # TR-BDF2 × Implicit Chebyshev
```

---

## 🧩 Summary of Methods

| Folder | Blend | Purpose |
|---------|--------|----------|
| **blended2** | GL1 × BDF2 | Baseline low-order demonstration |
| **blended3** | ESDIRK4(3) × AM3 | Mid-order, cross-family efficiency |
| **blended4** | BDF4 × Lobatto3C | Matched 4th-order, symmetric vs. diffusive |
| **blended9-10** | Radau5 × Gauss5 | High-order collocation frontier |
| **blendedTRBDFC** | TR-BDF2 × Chebyshev | Stiff-PDE / real-axis stability focus |

---

## 🧠 Scientific Context

**BlendedSolvers** defines a new solver family:

> **Blended Implicit Methods (BIMs)**  
> Implicit ODE integrators that interpolate between two integration operators through a continuous stiffness-dependent weighting law.

Distinct from IMEX or additive RK schemes, BIMs maintain a single unified implicit formulation while varying their **numerical character** (L-stable ↔ symplectic, multistep ↔ multistage).

This allows continuous control between:
- energy-preserving and strongly damping regimes,  
- collocation and multistep behavior,  
- mild and severe stiffness handling.

---

## 🔬 Research Value

- Experments with a *new axis* of adaptivity in ODE solver design.  
- Demonstrates smooth transitions between classical implicit families.  
- Serves as a reproducible reference for testing stiffness-adaptive integration.  
- Provides a foundation for future theoretical work on blended stability and order conditions.

Planned next steps:
- Jacobian reuse / W-method variants  
- Embedded error estimation for adaptive steps  
- Benchmarks on Van der Pol, Robertson, and CFD semi-discretizations  
