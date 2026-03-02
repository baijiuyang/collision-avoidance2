# Bai movObst2 — Pedestrian Obstacle Avoidance Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Open%20Source-green.svg)](LICENSE)

A computational modeling framework for pedestrian moving-obstacle avoidance behavior in the **Bai movObst2** experiment, based on dynamical systems approaches (Fajen-style steering and Cohen-style avoidance). This project includes ODE simulation, model fitting via optimization, data import pipelines, and analysis notebooks.

**We welcome contributions** — whether you want to fix bugs, extend models, add analyses, or improve documentation. See [Contributing](#contributing) for how to get involved.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository implements modeling and analysis specifically for the **Bai movObst2** experiments (movObst2a, movObst2b):

- **Approach models** (goal-directed steering): `fajen_approach`, `fajen_approach2`, `acceleration_approach`, `jerk_approach`
- **Avoidance models** (obstacle evasion): `cohen_avoid`, `cohen_avoid2`, `cohen_avoid3`, `cohen_avoid4`, `perpendicular_avoid`, `cohen_avoid_heading`, `cohen_avoid4_heading`
- **ODE simulation** of combined approach + avoidance dynamics
- **Model fitting** via differential evolution, dual annealing, Nelder–Mead, and related methods
- **VR experiment scripts** (`movObst2a`, `movObst2b`) for data collection (requires WorldViz Vizard)

---

## Project Structure

```
├── requirements.txt
├── README.md
└── Bai_movObst2/
    ├── packages/
    │   ├── ode_simulator.py   # ODE integration & trajectory simulation
    │   ├── models.py          # Approach and avoidance model definitions
    │   ├── helper.py          # Geometry, plotting, and utilities
    │   └── data_container.py  # Data loading and preprocessing
    ├── parameters.py          # Trained model parameters and bounds
    ├── fitting.py             # Model fitting via scipy.optimize
    ├── movObst2a.py           # VR experiment (moving obstacle)
    ├── movObst2b.py           # VR experiment variant
    ├── movObst2_inputGenerator.py
    ├── vizconnect_config.py   # Vizard connection config
    ├── import_data_*.ipynb    # Raw data → processed Data objects
    ├── simulation.ipynb       # Run simulations and visualize trajectories
    ├── analysis.ipynb         # Statistical analysis and plots
    ├── start_fitting_cross_validation.bat
    ├── start_fitting_individual.bat
    └── fitting_cmd_input.txt  # Example fitting command
```

---

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, scikit-learn, Matplotlib

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install numpy scipy scikit-learn matplotlib
   ```

4. **Data layout**  
   Place raw experiment data under `Raw_Data/` relative to this project. The notebooks and scripts expect pickle files such as `Bai_movObst2_data_30Hz.pickle`, `Cohen_movObst2_data.pickle`, etc.

---

## Usage

### 1. Run simulations

Use `Bai_movObst2/simulation.ipynb` to:

- Load data with `load_data('Bai_movObst2_data')` (or another dataset)
- Run simulations with `ODESimulator` and fitted models from `parameters.py`
- Visualize trajectories with `sim.play()`

### 2. Fit models

`Bai_movObst2/fitting.py` supports command-line arguments for experiment, subjects, and models.

**Example: fit avoidance model (all subjects)**

```bash
python Bai_movObst2/fitting.py --experiment_name Bai_movObst2 \
  --approach_experiment Bai_movObst2 \
  --subject all \
  --t_start obst_onset \
  --t_end obst_out \
  --ps trial \
  --method differential_evolution \
  --approach_model fajen_approach2 \
  --avoid_model cohen_avoid4 \
  --training_model cohen_avoid4
```

**Example: leave-one-subject-out cross-validation**

Use negative subject indices to exclude subjects (e.g. `--subject -0` excludes subject 0). The batch file `start_fitting_cross_validation.bat` runs multiple subjects in parallel.

**Key arguments**

| Argument | Description |
|----------|-------------|
| `--experiment_name` | `Bai_movObst2`, `Cohen_movObst2`, etc. |
| `--approach_experiment` | Experiment used for approach model parameters |
| `--subject` | Subject ID, `all`, or negative for cross-validation |
| `--t_start` / `--t_end` | Time window, e.g. `obst_onset`, `obst_out`, `dtheta*dpsi_onset` |
| `--method` | `differential_evolution`, `dual_annealing`, `nelder-mead`, `shgo`, `basinhopping` |
| `--approach_model` | `fajen_approach`, `fajen_approach2`, `acceleration_approach`, `jerk_approach` |
| `--avoid_model` | `cohen_avoid`, `cohen_avoid2`, `cohen_avoid3`, `cohen_avoid4`, `perpendicular_avoid`, etc. |
| `--training_model` | Model being fitted (same as approach or avoidance) |

### 3. Import and analyze data

- `Bai_movObst2/import_data_bai_movObst2.ipynb`  
  Convert raw CSV/experiment output to `Data` objects and save as pickle files.
- `Bai_movObst2/analysis.ipynb`  
  Run statistics, model comparisons, MDS, and plots.

---

## Data

- Raw data should be placed in `Raw_Data/` (or the path used in `fitting.py` and notebooks).
- The `Data` class in `Bai_movObst2/packages/data_container.py` handles trial metadata, trajectories, and derived quantities.
- Preprocessed data files (`.pickle`) are produced by the import notebooks.

---

## Contributing

We welcome contributions from the community. Here is how you can help.

### How to contribute

1. **Fork** the repository and create a new branch for your changes.
2. **Discuss** large changes by opening an issue first.
3. **Implement** your changes with clear, documented code.
4. **Test** your changes (e.g. run `fitting.py` or relevant notebooks).
5. **Submit a pull request** with a concise description of the change.

### Contribution areas

- **Models**: New approach/avoidance models in `packages/models.py` and wiring in `ode_simulator.py`
- **Fitting**: New optimization methods or metrics in `fitting.py`
- **Analysis**: New analyses, plots, or metrics in `analysis.ipynb`
- **Documentation**: README, docstrings, tutorials, and examples
- **Data**: Bug fixes or extensions in `data_container.py` and import notebooks
- **VR experiments**: Fixes or extensions for `movObst2a`, `movObst2b` (Vizard-specific)

### Code style

- Use PEP 8 style conventions.
- Add docstrings for new functions and classes.
- Keep commits focused and messages descriptive.

### Reporting issues

- Use the GitHub issue tracker for bugs, feature requests, and questions.
- Include:
  - Clear description of the problem or request
  - Steps to reproduce (for bugs)
  - Environment (OS, Python version)
  - Relevant output or error messages

---

## License

See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on dynamical systems models of steering and obstacle avoidance by Fajen and Warren, and Cohen and collaborators. Created by Jiuyang Bai.
