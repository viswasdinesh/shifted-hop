# HOP Horizon-Optimal Tutorial

This repository provides a compact Python implementation of **HOP**
(Horizon-Optimal Planning), an algorithm for selecting both an optimal control
trajectory and its planning horizon.

The examples are intentionally lightweight: they show how to call HOP-LQR for
horizon selection and HOP-DDP for nonlinear trajectory optimization.

## Files

```text
HOP_colab_notebook.ipynb  # Notebook version of the tutorial
utils.py                  # Notebook utility functions
systems.py                # Double integrator and quadrotor toy systems
lqr.py                    # Brute-force check, augmented model, HOP-LQR search
ddp.py                    # Linearization, iLQR pass, HOP-DDP solver
run_double_integrator.py  # Double Integrator example
run_quadrotor.py          # Quadrotor example
requirements.txt          # Minimal dependencies
environment.yml           # Minimal conda environment
```

## Quick Start

Create the minimal conda environment:

```bash
cd public_HOP_horizon_optimal_tutorial
conda env create -f environment.yml
conda activate hop
python run_double_integrator.py
```

If you are not using conda, install the minimal dependencies with:

```bash
python3 -m pip install -r requirements.txt
python3 run_double_integrator.py
```

The Double Integrator script runs the first tutorial example.

The Quadrotor script runs the second tutorial example:

```bash
python run_quadrotor.py
```

The brute-force Quadrotor check is slower. To run only HOP:

```bash
python run_quadrotor.py --skip-bruteforce
```

## Code Mapping

- `utils.py` contains the utility functions used throughout the notebook.
- `systems.py` defines the Double Integrator and Quadrotor examples.
- `lqr.py` contains the brute-force horizon sweep, augmented-system construction, and HOP-LQR horizon search.
- `ddp.py` contains finite-difference linearization, the iLQR backward/forward passes, and `solve_hop`.
- `run_double_integrator.py` and `run_quadrotor.py` are runnable examples built from the notebook code.
