# Revisit Simulation

This is a Python tool for computing maximum and mean revisit times of Earth-observing satellite constellations. It supports flexible orbital, sensor, and constellation configurations, including Sun-Synchronous Orbits (SSO) and non-SSO cases. The simulation can generate visualizations to analyze satellite coverage and optimize constellation design.

This implementation is based on:

> N.H. Crisp, S. Livadiotti, P.C.E. Roberts, *A Semi-Analytical Method for Calculating Revisit Time for Satellite Constellations with Discontinuous Coverage*, ArXiv E-Prints, 2018.

> Original MATLAB code: [https://github.com/nhcrisp/RevisitTime](https://github.com/nhcrisp/RevisitTime)

---

## Features

* Configurable orbit parameters: altitude, inclination, eccentricity, argument of perigee.
* Supports both SSO and non-SSO orbits.
* Sensor configuration with field-of-view (FoV) constraints.
* Constellation configuration: number of satellites, planes, and relative spacing. Sets up a standard Walker constellation
* Parallelized computation using `ProcessPoolExecutor` for faster processing.
* Generates plots for:

  * Maximum revisit vs. altitude
  * Sensor FoV vs. altitude contour maps
  * Inclination maps for non-SSO orbits

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ffwieser/revisit_simulation.git
cd revisit_simulation
```

2. Install dependencies:

```bash
pip install numpy matplotlib tqdm
```

> Note: The simulation depends on a local module `revisit_lib` containing `RevisitCalc` and `ReShape`.

---

## Configuration

Simulation parameters are defined in `config.json`. Example:

```json
{
  "orbit": {
    "latitude": 40,
    "one_side_view": 0,
    "descend_ascend": 1,
    "SSO": 1,
    "day_limit": [1, 60],
    "eccentricity": 0,
    "argument_perigee": 0,
    "altitude_min": 100000,
    "altitude_max": 300000,
    "h_res": 10000,
  },
  "sensor": {
    "sensor_half_cone_min": 25,
    "sensor_half_cone_max": 50,
    "elevation": null
  },
  "constellation": {
    "satellites": 1,
    "planes": 1,
    "relative_spacing": 0
  }
}
```

If `config.json` is not found, the simulation uses default parameters. For an explanation of the individual parameters, refer to the original paper.

---

## Usage

```python
import json
from revisit_simulation import RevisitSimulation

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Create simulation object
sim = RevisitSimulation(config)

# Run simulation
sim.run()

# Visualize results
sim.plot()
```

---

## Structure

```
RevisitSimulation/
├─ revisit_simulation.py   # Main simulation code
├─ config.json             # Optional configuration file
├─ revisit_lib/            # Required external library with RevisitCalc & ReShape
└─ README.md
```

---

## Visualization

The `plot()` method generates:

1. **Line plots:** Maximum revisit vs. altitude.
2. **Contour plots:** Sensor half-cone vs. altitude.
3. **Inclination maps:** For non-SSO orbits.


---

## License

This project is licensed under the **GPL-3.0 License**.

