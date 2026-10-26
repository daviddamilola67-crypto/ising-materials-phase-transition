# Finite-size effects in a 2D Ising material model

This computational materials science example uses Metropolis Monte Carlo to study temperature-dependent magnetic ordering in a square-lattice, nearest-neighbor Ising model. It compares two lattice sizes and reports energy, absolute magnetization, heat capacity, and a finite-size susceptibility estimator over a temperature sweep.

## Run

Install the dependency with `python -m pip install -r requirements.txt`, then run `python ising_materials_study.py`. The script writes `results/ising_temperature_sweep.csv`, `results/study_summary.json`, and `results/ising_curves.svg`. The fixed random seed and simulation settings are recorded in the JSON summary.

## Model and interpretation

The model is a two-dimensional square lattice with periodic boundaries, spins of ±1, and ferromagnetic nearest-neighbor coupling. Reduced units set J = 1 and kB = 1. Checkerboard Metropolis updates use a fixed pseudorandom seed. Heat capacity is estimated from energy fluctuations; susceptibility is estimated from fluctuations in absolute magnetization and labeled accordingly.

For the infinite square-lattice model, the known critical temperature is kB*T/J = 2/ln(1+sqrt(2)) ≈ 2.269. Finite lattices and finite sampling broaden and shift peaks, so this run illustrates finite-size behavior rather than producing a precision critical-point estimate.

## Scope and limitations

This is an educational simulation, not a validated model of a particular magnetic material. It omits magnetic anisotropy, long-range interactions, quantum effects, and material-specific parameters. It reports no uncertainty intervals; conclusions are qualitative unless sampling and finite-size convergence are studied further.
