
# Is Your Phone Linear?

A Python tool for calibrating a smartphone accelerometer and evaluating its linearity using an affine model:

\[
m = Sa + b
\]

where:
- `m` — measured acceleration
- `a` — true acceleration
- `S` — 3×3 calibration matrix
- `b` — bias vector

The model parameters are estimated using least squares on six static phone orientations.

## Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn pytest
```

Run all commands from the project root directory.


## Commands

All commands should be executed from the project root directory.

### Fit

Fit the affine calibration model using the first recording session.

```bash
python -m calib [--data-dir DATA_DIR] [--params PARAMS] fit
```

**Options:**
- `--data-dir DATA_DIR` — directory containing raw CSV recordings (default: `data/raw/`).
- `--params PARAMS` — path to save the fitted model parameters (default: `results/params.json`).

The command estimates the calibration matrix `S` and bias vector `b`, prints the rank, condition number and mean residual error, and saves the parameters to JSON.

**Example:**

```bash
python -m calib --data-dir data/raw --params results/model.json fit
```

### Evaluate

Evaluate the fitted model using the second recording session.

```bash
python -m calib [--data-dir DATA_DIR] [--params PARAMS] evaluate [--plot]
```

**Options:**
- `--data-dir DATA_DIR` — directory containing test recordings (default: `data/raw/`).
- `--params PARAMS` — path to the saved calibration parameters (default: `results/params.json`).
- `--plot` — display a bar chart comparing acceleration errors before and after calibration.

The command calculates acceleration errors in m/s² and angular errors in degrees for each test pose, and prints their mean values.

**Example:**

```bash
python -m calib --params results/model.json evaluate --plot
```

### Apply

Apply the saved calibration to a new CSV recording.

```bash
python -m calib [--params PARAMS] apply RECORDING [--output OUTPUT]
```

**Arguments:**
- `RECORDING` — path to the input CSV file (required).

**Options:**
- `--params PARAMS` — path to the saved calibration parameters (default: `results/params.json`).
- `--output OUTPUT` — path to save corrected measurements (default: `results/corrected.csv`).

The command applies the calibration to each acceleration measurement and saves the corrected X, Y, Z values and timestamps to a CSV file.

**Example:**

```bash
python -m calib --params results/model.json apply data/raw/screen_up_2.csv --output results/screen_up_corrected.csv
```

### Level (demo)

Print the tilt angle of a new recording, raw and calibrated. Optionally compare it with a ruler angle from h and L (tan θ = h/L).

```bash
python -m calib level RECORDING [--h-cm H] [--l-cm L]
```

**Example:**

```bash
python -m calib level "data/raw/h=6,l=16,angle=22.csv" --h-cm 6 --l-cm 16
```

### Ramp

Wall tilt test on the four tilt recordings. The true angle comes from tan θ = h/L, with an uncertainty of about ±2.2° to ±2.4° from a ±0.5 cm tape error. Prints raw and calibrated angle errors in degrees and m/s².

```bash
python -m calib ramp
```

### Rank

Rank of the design matrix for different sets of poses. S and b are determined uniquely only when the rank is 4. The table shows which pose sets are not enough.

```bash
python -m calib rank
```



### Tests

Run all automated tests using pytest.

```bash
python -m pytest -v
```

**Options:**
- `-v` — verbose output with individual test results.
- `-q` — concise output.

Tests cover data loading, stationary interval detection, affine model fitting on synthetic data, calibration application, JSON serialization, and rank validation.

**Example:**

```bash
python -m pytest -q
```

### Help

Display available commands and global options:

```bash
python -m calib --help
```

Display help for a specific command:

```bash
python -m calib fit --help
python -m calib evaluate --help
python -m calib apply --help
```

## Data

The dataset contains:
- 12 static recordings (six orientations, two sessions).
- 4 recordings at known tilt angles.
- 1 shaking experiment.

Original recordings are stored in `data/raw/`. See `data/README.md` for the measurement protocol and data format.

## Project Structure

```text
calib/
    data.py          # Loading and preprocessing
    model.py         # Affine calibration
    evaluation.py    # Evaluation metrics
    __main__.py      # Command-line interface

data/
    raw/             # Original recordings
    README.md        # Measurement protocol

tests/               # Pytest tests
results/             # Calibration parameters and outputs
decisions.md         # Project decisions
```
