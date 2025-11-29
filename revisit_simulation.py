"""
Revisit Simulation

Compute maximum and mean satellite revisit times for configurtable constellations. Supports SSO
and non-SSO orbits, variable sensor fields-of-view and visualization of results. Based on the 
RevisitTime MATLAB toolkit by N.H. Crisp et al. (https://github.com/nhcrisp/RevisitTime).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from revisit_lib import RevisitCalc, ReShape


def compute_case(
    i,
    a_col,
    lat_col,
    inc_rad,
    var_col,
    ecc,
    rad_AoP,
    f_p,
    f_e,
    elv,
    noSats,
    noPlanes,
    relSpacing,
    descend_ascend,
    one_side_view,
    dayLimit,
):
    coes = (a_col[i], ecc, inc_rad[i], 0.0, rad_AoP)

    if f_p and f_e:
        var_row = np.array([var_col[i], elv])
    elif f_p:
        var_row = var_col[i]
    else:
        var_row = elv

    return i, RevisitCalc(
        coes,
        lat_col[i],
        f_p,
        f_e,
        noSats,
        noPlanes,
        relSpacing,
        descend_ascend,
        one_side_view,
        dayLimit,
        var_row,
    )


class RevisitSimulation:
    def __init__(self, config):
        self.lat = config["orbit"].get("latitude", 40)
        self.one_side_view = config["orbit"].get("one_side_view", 0)
        self.descend_ascend = config["orbit"].get("descend_ascend", 1)
        self.SSO = config["orbit"].get("SSO", 1)
        self.dayLimit = config["orbit"].get("day_limit", (1, 60))
        self.ecc = config["orbit"].get("eccentricity", 0)
        self.AoP = config["orbit"].get("argument_perigee", 0)

        # Altitude
        self.h_min = config["orbit"].get("altitude_min", 100e3)
        self.h_max = config["orbit"].get("altitude_max", self.h_min)
        self.h_delta = config["orbit"].get("h_res", 1e3)

        # Inclination for non-SSO
        self.inc_min = config["orbit"].get("inclination_min", 60)
        self.inc_max = config["orbit"].get("inclination_max", self.inc_min)
        self.inc_delta = config["orbit"].get("inclination_res", 5)

        # Sensor
        self.psi_min = config["sensor"].get("sensor_half_cone_min", 45)
        self.psi_max = config["sensor"].get("sensor_half_cone_max", self.psi_min)
        self.elv = config["sensor"].get("elevation", None)
        self.f_p = self.psi_min is not None and self.psi_max is not None
        self.f_e = self.elv is not None

        # Constellation
        self.noSats = config["constellation"].get("satellites", 1)
        self.noPlanes = config["constellation"].get("planes", 1)
        self.relSpacing = config["constellation"].get("relative_spacing", 0)

        # WGS84 + J2 constants
        self.R_E = 6378.137e3
        self.mu_E = 3.986004418e14
        self.J2 = 1.082629989052e-3
        self.omega_E = 7.292115373194000e-05
        self.Flat = 1 / 298.257223563
        self.ecc_E = 0.081819221456
        self.Tday = 24 * 3600

        # Storage for results
        self.MaxRevisit = None
        self.maxRevisit = None
        self.meanRevisit = None
        self.dayLimitOut = None

        # Precomputed vectors
        self.h_a = None
        self.psi = None
        self.inc = None

    # =====================================================================
    #  UTILITY: SSO inclination
    # =====================================================================
    def compute_sso_inclination(self, a):
        """
        Compute SSO inclination for each semimajor axis value 'a'.
        Clip argument to arccos to [-1,1] to avoid NaNs from tiny round-off.
        Returns array of inclinations in degrees (same shape as a).
        """
        dOmegaSS = 360.0 / (365.2421897 * self.Tday) * (np.pi / 180.0)
        numer = -2.0 * a ** (7.0 / 2.0) * dOmegaSS * (1.0 - self.ecc**2) ** 2
        denom = 3.0 * self.R_E**2 * self.J2 * np.sqrt(self.mu_E)
        arg = numer / denom
        arg = np.clip(arg, -1.0, 1.0)
        return np.degrees(np.arccos(arg))

    # =====================================================================
    #  MAIN
    # =====================================================================

    def run(self):
        # --- Precompute constants ---
        rad_AoP = np.radians(self.AoP)
        f_p, f_e = self.f_p, self.f_e
        elv, ecc = self.elv, self.ecc
        noSats, noPlanes = self.noSats, self.noPlanes
        relSpacing, descend_ascend = self.relSpacing, self.descend_ascend
        one_side_view, dayLimit = self.one_side_view, self.dayLimit

        # --- Build altitude vector ---
        self.h_a = np.arange(self.h_min, self.h_max + self.h_delta, self.h_delta)
        a_vec = (self.R_E + self.h_a) / (1 + ecc)

        # --- FoV vector ---
        self.psi = (
            np.arange(self.psi_min, self.psi_max + 1) if f_p else np.array([np.nan])
        )

        # --- Inclination vector ---
        if self.SSO == 1:
            inc_vec = self.compute_sso_inclination(a_vec)
        else:
            inc_vec = np.arange(
                self.inc_min, self.inc_max + self.inc_delta, self.inc_delta
            )
        self.inc = inc_vec

        # --- Build parameter grid ---
        W = self._build_parameter_grid(a_vec)
        a_col, lat_col, inc_col, var_col = W[:, 0], W[:, 1], W[:, 2], W[:, 3]
        n_cases = W.shape[0]
        inc_rad = np.radians(inc_col)

        # --- Prepare result arrays ---
        maxRev = np.zeros(n_cases)
        meanRev = np.zeros(n_cases)

        # --- Parallel execution ---
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    compute_case,
                    i,
                    a_col,
                    lat_col,
                    inc_rad,
                    var_col,
                    ecc,
                    rad_AoP,
                    f_p,
                    f_e,
                    elv,
                    noSats,
                    noPlanes,
                    relSpacing,
                    descend_ascend,
                    one_side_view,
                    dayLimit,
                )
                for i in range(n_cases)
            ]

            for future in tqdm(
                as_completed(futures), total=n_cases, desc="Computing revisit metrics"
            ):
                i, (max_i, mean_i, flag) = future.result()
                maxRev[i] = max_i
                meanRev[i] = mean_i
                last_flag = flag

        self.maxRevisit = maxRev
        self.meanRevisit = meanRev

        self.MaxRevisit = ReShape(
            self.maxRevisit,
            last_flag,
            self.SSO,
            len(inc_vec),
            len(self.h_a),
            len(self.psi) if f_p else 1,
        )

        return self.MaxRevisit

    # =====================================================================
    #  BUILD ND-GRID OF PARAMETERS
    # =====================================================================
    def _build_parameter_grid(self, a):
        """
        Build MATLAB-equivalent parameter sweep grid W.
        Returns W with columns:
            [a_value, lat_value, inc_value, var_value]
        where var_value = psi or elv (scalar).
        """
        lat_val = float(self.lat)
        rows = []

        # Ensure arrays
        if self.f_p:
            psi = np.arange(self.psi_min, self.psi_max + 1)
        else:
            psi = [None]

        inc = np.array(self.inc)

        # ---------------------------
        # Case A: psi provided (f_p)
        # ---------------------------
        if self.f_p:
            # === SSO ===
            if self.SSO == 1:
                # inc matches altitude vector 1:1 — correct MATLAB behavior
                for psi_val in psi:
                    for k, a_val in enumerate(a):
                        rows.append(
                            [
                                a_val,
                                lat_val,
                                inc[k],  # correct inc for that altitude
                                psi_val,  # var = psi
                            ]
                        )
            # === Non-SSO ===
            else:
                for psi_val in psi:
                    for a_val in a:
                        for inc_val in inc:
                            rows.append([a_val, lat_val, inc_val, psi_val])

        # ---------------------------------------
        # Case B: psi NOT provided, only elv used
        # ---------------------------------------
        else:
            # === SSO ===
            if self.SSO == 1:
                for k, a_val in enumerate(a):
                    rows.append([a_val, lat_val, inc[k], self.elv])
            # === Non-SSO ===
            else:
                for a_val in a:
                    for inc_val in inc:
                        rows.append([a_val, lat_val, inc_val, self.elv])

        return np.array(rows, dtype=float)

    # =====================================================================
    #  PLOTTING
    # =====================================================================
    def plot(self):
        if self.MaxRevisit is None:
            raise RuntimeError("You must call run() before plot().")

        MaxRevisit = self.MaxRevisit
        h_a = self.h_a
        psi = self.psi
        inc = self.inc

        # --------------------------
        # 1 — Line plot vs altitude
        # --------------------------
        plt.figure(figsize=(7, 5))
        plt.plot(h_a / 1e3, np.squeeze(MaxRevisit[0, :, 0]))
        plt.xlabel("Altitude [km]")
        plt.ylabel("Maximum Revisit [day]")
        plt.grid(True)

        # -----------------------------------
        # 2 — Contour (psi vs altitude)
        # -----------------------------------
        if not self.f_e and len(psi) >= 2:
            H, P = np.meshgrid(h_a / 1e3, psi)
            Z = np.squeeze(MaxRevisit[0, :, :]).T

            plt.figure(figsize=(7, 5))
            cp = plt.contourf(H, P, Z, 30)
            plt.xlabel("Altitude [km]")
            plt.ylabel("ψ [deg]")
            c = plt.colorbar(cp)
            c.set_label("Max Revisit [day]")

        # -----------------------------------
        # 3 — Inclination maps (non-SSO)
        # -----------------------------------
        if self.SSO == 0:
            H2, I2 = np.meshgrid(h_a / 1e3, inc)
            Z2 = np.squeeze(MaxRevisit[:, :, 0])

            plt.figure(figsize=(7, 5))
            plt.plot(inc, np.squeeze(MaxRevisit[:, 0, 0]))
            plt.xlabel("Inclination [deg]")
            plt.ylabel("Maximum Revisit [days]")
            plt.grid(True)

            plt.figure(figsize=(7, 5))
            cp = plt.contourf(H2, I2, Z2, 30)
            plt.xlabel("Altitude [km]")
            plt.ylabel("Inclination [deg]")
            c = plt.colorbar(cp)
            c.set_label("Max Revisit [days]")

        plt.show()


if __name__ == "__main__":
    # read configuration file
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        with open(path + "/config.json", "r") as f:
            config = json.load(f)
        print("Running simulation from configuration file.")
    except FileNotFoundError:
        print("Running simulation with default configuration.")
        config = {}
        config["orbit"] = {}
        config["orbit"]["altitude_max"] = 300e3
        config["sensor"] = {}
        config["constellation"] = {}

    sim = RevisitSimulation(config)

    sim.run()
    sim.plot()
