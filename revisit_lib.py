"""
Python translation of the RevisitTime MATLAB toolkit by by N.H. Crisp et al.
"""

import numpy as np

# WGS84 + J2 constants
R_E = 6378.137e3
mu_E = 3.986004418e14
J2 = 1.082629989052e-3
omega_E = 7.292115373194e-05
Flat = 1 / 298.257223563
Tday = 24 * 3600
_TWO_PI = 2.0 * np.pi
_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi


# trigonometric helper functions
def sind(x):
    return np.sin(x * _DEG2RAD)


def cosd(x):
    return np.cos(x * _DEG2RAD)


def tand(x):
    return np.tan(x * _DEG2RAD)


def asind(x):
    return np.arcsin(x) * _RAD2DEG


def acosd(x):
    return np.arccos(x) * _RAD2DEG


def atand(x):
    return np.arctan(x) * _RAD2DEG


def atan2d(y, x):
    return np.arctan2(y, x) * _RAD2DEG


# ---------------------- Optimized functions ---------------------------


def RevisitCalc(
    coes,
    lat,
    f_p,
    f_e,
    noSats,
    noPlanes,
    relSpacing,
    DscPass,
    OneSideView,
    dayLimit,
    var,
):
    """
    Optimized version of RevisitCalc.
    Most heavy work offloaded to vectorized numpy and fewer Python-level loops.
    """
    # unpack
    a, ecc, inc, RAAN, AoP = coes
    perPlane = noSats / noPlanes

    # geodetic to geocentric conversion (single operation)
    lat = atand((1 - Flat) ** 2 * tand(lat))

    # compute theta / elv_psi for ascending
    theta, elv_psi, flag, nu = calcTheta(lat, coes, 0, f_p, f_e, None, var)

    DlonA = 2.0 * acosd((cosd(theta) - sind(lat) ** 2) / (cosd(lat) ** 2))

    # descending pass values if requested
    if DscPass == 1:
        # theta_2, elv_psi_2, flag_2, nu_2 = calcTheta(lat, coes, 1, f_p, f_e, None, var)
        _, _, _, nu_2 = calcTheta(lat, coes, 1, f_p, f_e, None, var)
        # DlonA_2 = 2.0 * acosd((cosd(theta_2) - sind(lat) ** 2) / (cosd(lat) ** 2))
    else:
        # keep placeholders
        nu_2 = None

    # orbit properties
    n = np.sqrt(mu_E / (a**3))
    p = a * (1 - ecc**2)
    P = 2.0 * np.pi * np.sqrt(a**3 / mu_E)

    # nodal period P_n and RAAN drift dRAAN (use scalars)
    sqrt1me2 = np.sqrt(1.0 - ecc**2)
    sin_inc = np.sin(inc)
    sin_inc2 = sin_inc**2
    P_n = P * (
        1.0
        / (
            1.0
            + 0.75
            * J2
            * (R_E / p) ** 2
            * (sqrt1me2 * (2.0 - 3.0 * sin_inc2) + (4.0 - 5.0 * sin_inc2))
        )
    )
    dRAAN = -(1.5) * n * J2 * (R_E / p) ** 2 * np.cos(inc) + (
        3.0 / 32.0
    ) * n * J2**2 * (R_E**4 / p**4) * np.cos(inc) * (
        12.0 - 4.0 * ecc**2 - (80.0 + 5.0 * ecc**2) * sin_inc2
    )

    Dlon0 = P_n * (-omega_E + dRAAN) * _RAD2DEG

    # passNo baseline (one call)
    max_pass_est = int(np.ceil((dayLimit[0] * Tday) / P_n))
    # compute lon_start and (if needed) descending lon
    # compute AoP+nu once
    u_an = AoP + nu
    lon_start = (
        atan2d(
            np.cos(u_an) * np.sin(RAAN) + np.sin(u_an) * np.cos(RAAN) * np.cos(inc),
            np.cos(u_an) * np.cos(RAAN) - np.sin(u_an) * np.sin(RAAN) * np.cos(inc),
        )
        + (((nu + AoP) % _TWO_PI) / _TWO_PI) * P_n * (-omega_E + dRAAN) * _RAD2DEG
    )

    if DscPass == 1:
        u_an2 = AoP + nu_2
        dLon_2 = (
            atan2d(
                np.cos(u_an2) * np.sin(RAAN)
                + np.sin(u_an2) * np.cos(RAAN) * np.cos(inc),
                np.cos(u_an2) * np.cos(RAAN)
                - np.sin(u_an2) * np.sin(RAAN) * np.cos(inc),
            )
            + (((nu_2 + AoP) % _TWO_PI) / _TWO_PI) * P_n * (-omega_E + dRAAN) * _RAD2DEG
        )

    # MAIN LOOP: accumulating lists and early-exit if failure condition is met.
    i = 0
    reloop = True

    # collect per-iteration results in lists and only stack when needed
    inview_list = []
    tviewmin_list = []
    tviewmax_list = []
    passTimes_list = []

    while reloop:
        i += 1
        if i == 1:
            passNo = np.arange(0, max_pass_est + 1, dtype=np.int64)
            passTime_0 = P_n * (((nu + AoP) % _TWO_PI) / _TWO_PI)
            # TODO remove passTime_start = passTime_0
        else:
            # extend passNo estimate based on i
            new_max_pass_est = int(np.ceil((dayLimit[0] * i * Tday) / P_n))
            passNo = np.arange(passNo[-1] + 1, new_max_pass_est + 1, dtype=np.int64)

        # compute pass times (base)
        passTimes_i = (passNo * P_n + passTime_0).astype(np.float64)

        # compute longitudes of ascending passes (base)
        lon = (lon_start + Dlon0 * passNo) % 360.0

        # multiple planes handling vectorized:
        if noPlanes > 1:
            plane_offsets = np.concatenate(
                (np.arange(1, noPlanes) * 360.0 / noPlanes, [0.0])
            )
            dphase = (360.0 * relSpacing) / noSats
            orbfrac_f = np.flip(np.arange(1, noPlanes) * dphase / 360.0)
            Dlon0_f = np.concatenate((orbfrac_f * Dlon0, [0.0]))
            lon = (lon[None, :] + (plane_offsets + Dlon0_f)[:, None]) % 360.0
            # passTimes shape match: each plane has a time offset orbfrac_f * P_n
            passTimes_i = (
                passTimes_i[None, :] + np.concatenate((orbfrac_f * P_n, [0.0]))[:, None]
            )
        else:
            lon = lon.reshape(1, -1)

        # per-plane phasing (multiple sats per plane)
        if perPlane > 1:
            orbfrac_p = np.arange(1, int(perPlane)) / perPlane
            base_rows, ncols = lon.shape
            Dlon0_ms = np.concatenate((Dlon0 - orbfrac_p * Dlon0, [0.0]))
            passTimeDiff = np.concatenate((P_n - orbfrac_p * P_n, [0.0]))
            lon = (
                lon.reshape(base_rows, 1, ncols) + Dlon0_ms.reshape(-1, 1, 1)
            ).reshape(-1, ncols) % 360.0
            passTimes_i = (
                passTimes_i.reshape(base_rows, 1, ncols)
                + passTimeDiff.reshape(-1, 1, 1)
            ).reshape(-1, ncols)

        lon_flat = lon.flatten()
        passTimes_flat = passTimes_i.flatten()

        # descending pass handling: prepare descending lon/time arrays
        if DscPass == 1:
            lon_asc = lon_flat
            lon_2_diff = (dLon_2 - lon_start) % 360.0
            lon_2 = (lon_flat + lon_2_diff) % 360.0
            passTimes_2_i = passTimes_flat + P_n * (((nu_2 - nu) % _TWO_PI) / _TWO_PI)
            # TODO remove lon_all = np.concatenate((lon_flat, lon_2))
        else:
            lon_asc = lon_flat
            # TODO remove lon_all = lon_flat

        # compute passes for ascending passes using listPasses
        inView_a, t_min_a, t_max_a = listPasses(
            lat, lon_asc, coes, 0, f_p, f_e, flag, elv_psi, theta, DlonA
        )

        inview_list.append(inView_a)
        tviewmin_list.append(t_min_a)
        tviewmax_list.append(t_max_a)
        passTimes_list.append(passTimes_flat)

        # add descending passes if requested
        if DscPass == 1:
            inView_d, t_min_d, t_max_d = listPasses(
                lat, lon_2, coes, 1, f_p, f_e, flag, elv_psi, theta, DlonA
            )
            inview_list.append(inView_d)
            tviewmin_list.append(t_min_d)
            tviewmax_list.append(t_max_d)
            passTimes_list.append(passTimes_2_i)

        # combine across all iterations so far
        inview = np.hstack([np.concatenate([m for m in inview_list], axis=1)])

        if isinstance(inview, list):
            inview = inview[0]

        # construct combined t_view_min, t_view_max and passTimes arrays
        t_view_min = np.concatenate(
            [np.concatenate([m for m in tviewmin_list], axis=1)]
        )
        if isinstance(t_view_min, list):
            t_view_min = t_view_min[0]

        t_view_max = np.concatenate(
            [np.concatenate([m for m in tviewmax_list], axis=1)]
        )
        if isinstance(t_view_max, list):
            t_view_max = t_view_max[0]

        passTimes = np.concatenate([np.concatenate(passTimes_list).T])
        if isinstance(passTimes, list):
            passTimes = passTimes[0]

        # check failure condition as in original code
        if np.any(np.sum(inview, axis=1) < 2):
            if dayLimit[0] * i == dayLimit[1]:
                # TODO remove alt = a - R_E
                maxRevisit = dayLimit[1] + 1
                meanRevisit = dayLimit[1] + 1
                return maxRevisit, meanRevisit, flag
        else:
            reloop = False

    # compute revisit statistics (vectorized)
    passMatch = [np.nonzero(inview[row])[0] for row in range(inview.shape[0])]

    # compute passTimeMax and passTimeMin
    passTimeMax = np.empty(len(passMatch), dtype=np.float64)
    passTimeMin = np.empty_like(passTimeMax)
    pass_avg = np.empty_like(passTimeMax)

    for idx_row, idx in enumerate(passMatch):
        times_max = np.sort(passTimes[idx] + t_view_max[idx_row, idx])
        # drop possible NaNs if any (but should be fine)
        times_max = times_max[~np.isnan(times_max)]
        if times_max.size > 1:
            passTimeMax[idx_row] = np.max(np.diff(times_max))
        else:
            passTimeMax[idx_row] = 0.0

        times_min = np.sort(passTimes[idx] + t_view_min[idx_row, idx])
        times_min = times_min[~np.isnan(times_min)]
        if times_min.size > 1:
            passTimeMin[idx_row] = np.max(np.diff(times_min))
        else:
            passTimeMin[idx_row] = 0.0

        times = np.sort(passTimes[idx])
        if times.size > 1:
            pass_avg[idx_row] = np.mean(np.diff(times))
        else:
            pass_avg[idx_row] = 0.0

    maxRevisit = np.max(passTimeMax) / 3600.0 / 24.0
    meanRevisit = np.mean(pass_avg) / 3600.0 / 24.0

    return maxRevisit, meanRevisit, flag


def calcTheta(lat, coes, DscPass, f_p, f_e, flag_in, var):
    """
    Vectorized calcTheta. Handles scalar lat or 1D arrays latRange.
    Returns theta (deg), elv_psi, flag, nu (rad).
    Maintains original API but is faster for arrays.
    """
    a, ecc, inc, _, AoP = coes

    # precompute constants
    R_Emin = R_E - Flat * R_E

    # work with numpy arrays
    lat_arr = np.atleast_1d(lat).astype(np.float64)
    lat_rad = np.radians(lat_arr)

    R_ell = np.sqrt(
        ((R_E**2 * np.cos(lat_rad)) ** 2 + (R_Emin**2 * np.sin(lat_rad)) ** 2)
        / ((R_E * np.cos(lat_rad)) ** 2 + (R_Emin * np.sin(lat_rad)) ** 2)
    )

    # true anomaly from latitude equation (vectorized)
    # arcsin argument might slightly exceed [-1,1] numerically; clip
    asinarg = np.sin(np.radians(lat_arr)) / np.sin(inc)
    asinarg = np.clip(asinarg, -1.0, 1.0)
    nu = (np.arcsin(asinarg) - AoP) % _TWO_PI
    u = (nu + AoP) % _TWO_PI

    if DscPass == 1:
        nu = (np.pi - u - AoP) % _TWO_PI

    # orbit radius r_s (vectorized over nu)
    r_s = (a * (1.0 - ecc**2)) / (1.0 + ecc * np.cos(nu))

    h_ell = (R_E - R_ell) + (r_s - R_E)  # altitude above ellipsoid

    # branching for psi/elv flags.
    if f_p and f_e and (isinstance(var, (list, tuple, np.ndarray)) and len(var) == 2):
        psi = var[0]
        elv = var[1]

        theta_elv = (
            np.degrees(np.arccos(np.cos(np.radians(elv)) / (1.0 + h_ell / R_ell))) - elv
        )

        psi_rad = np.radians(psi)
        gam = 180.0 - np.degrees(np.arcsin(((R_ell + h_ell) * np.sin(psi_rad)) / R_ell))
        rho = R_ell * np.cos(np.radians(gam)) + (R_ell + h_ell) * np.cos(psi_rad)
        theta_psi = np.degrees(np.arcsin((rho * np.sin(psi_rad)) / R_ell))

        # choose limiting angle elementwise
        mask = theta_elv < theta_psi
        theta = np.where(mask, theta_elv, theta_psi)
        elv_psi = np.where(mask, elv, psi)
        flag = np.where(mask, 1, 0) if np.ndim(mask) > 0 else (1 if mask else 0)

    elif f_p and not f_e:
        psi = var
        elv_psi = psi
        flag = 0
        psi_rad = np.radians(psi)
        gam = 180.0 - np.degrees(np.arcsin(((R_ell + h_ell) * np.sin(psi_rad)) / R_ell))
        rho = R_ell * np.cos(np.radians(gam)) + (R_ell + h_ell) * np.cos(psi_rad)
        theta = np.degrees(np.arcsin((rho * np.sin(psi_rad)) / R_ell))

    elif f_e and not f_p:
        elv = var
        elv_psi = elv
        flag = 1
        theta = (
            np.degrees(np.arccos(np.cos(np.radians(elv)) / (1.0 + h_ell / R_ell))) - elv
        )

    elif f_p and f_e:
        # used in listPasses loop variant; use flag_in to decide
        if flag_in == 1:
            elv = var
            elv_psi = elv
            theta = (
                np.degrees(np.arccos(np.cos(np.radians(elv)) / (1.0 + h_ell / R_ell)))
                - elv
            )
            flag = 1
        else:
            psi = var
            elv_psi = psi
            psi_rad = np.radians(psi)
            gam = 180.0 - np.degrees(
                np.arcsin(((R_ell + h_ell) * np.sin(psi_rad)) / R_ell)
            )
            rho = R_ell * np.cos(np.radians(gam)) + (R_ell + h_ell) * np.cos(psi_rad)
            theta = np.degrees(np.arcsin((rho * np.sin(psi_rad)) / R_ell))
            flag = 0
    else:
        raise ValueError("Incorrect psi/elv assignment in input arguments.")

    # if input was scalar, return scalars
    if np.isscalar(lat):
        return (
            float(np.atleast_1d(theta)[0]),
            float(np.atleast_1d(elv_psi)[0]),
            int(np.atleast_1d(flag)[0]),
            float(np.atleast_1d(nu)[0]),
        )
    return theta, elv_psi, flag, nu


def listPasses(lat, passLon, coes, DscPass, f_p, f_e, flag, elv_psi, theta, DlonA):
    """
    Optimized listPasses with vectorized operations and reduced memory churn.
    Note: discLon left at 3600 for compatibility; can be reduced to improve speed.
    """
    a, ecc, inc_r, RAAN, AoP = coes
    P = 2.0 * np.pi * np.sqrt(a**3 / mu_E)

    passLon = np.asarray(passLon, dtype=np.float64).ravel()

    # discretize longitudes (large array; keep dtype to float32 if memory is a concern)
    discLon = 3600
    Longitudes = np.linspace(0.0, 360.0, discLon, dtype=np.float32)

    normDeg = np.mod(Longitudes[:, None] - passLon[None, :], 360.0)
    normDegAbs = np.minimum(360.0 - normDeg, normDeg)

    # passes inside instantaneous sensor width
    inView = normDegAbs <= (DlonA / 2.0)

    # maximum reachable latitude
    lat_max = np.minimum(inc_r, np.pi - inc_r) * _RAD2DEG

    # latitude range across sensor band
    latRange = np.linspace(lat - theta, lat + theta, dtype=np.float64)
    latRange = latRange[latRange <= lat_max]
    latRange = np.append(latRange, lat_max)

    # true anomaly at target latitude
    asinarg = np.sin(np.deg2rad(latRange)) / np.sin(inc_r)
    asinarg = np.clip(asinarg, -1.0, 1.0)
    if DscPass == 0:
        nu = np.arcsin(asinarg) - AoP
    else:
        latRange = latRange[::-1]
        asinarg = np.sin(np.deg2rad(latRange)) / np.sin(inc_r)
        asinarg = np.clip(asinarg, -1.0, 1.0)
        nu = np.pi - np.arcsin(asinarg) - AoP

    t_delta = (nu / _TWO_PI) * P

    theta_range, _, _, _ = calcTheta(latRange, coes, DscPass, f_p, f_e, flag, elv_psi)

    # full equivalent longitude range
    # careful with broadcasting and deg/rad conversions
    latRange_rad = np.deg2rad(latRange)
    theta_range_rad = np.deg2rad(theta_range)
    numerator = np.cos(theta_range_rad) - np.sin(latRange_rad) ** 2
    denom = np.cos(latRange_rad) ** 2
    arg = np.clip(numerator / denom, -1.0, 1.0)
    DlonA_range = 2.0 * np.degrees(np.arccos(arg)) * (1.0 + Flat)

    # center anomaly times around main latitude
    midindx = np.argmin(np.abs(latRange - lat))
    t_delta = t_delta - t_delta[midindx]

    # mean motion and drift rate
    n = np.sqrt(mu_E / a**3)
    p = a * (1.0 - ecc**2)
    lonDrift = (
        -omega_E
        - 1.5 * n * J2 * (R_E / p) ** 2 * np.cos(inc_r)
        + (3.0 / 32.0)
        * n
        * J2**2
        * (R_E**4 / p**4)
        * np.cos(inc_r)
        * (12.0 - 4.0 * ecc**2 - (80.0 + 5.0 * ecc**2) * np.sin(inc_r) ** 2)
    )

    # longitudinal trace of target latitude
    u = AoP + nu
    lonRange = (
        np.degrees(
            np.arctan2(
                np.cos(u) * np.sin(RAAN) + np.sin(u) * np.cos(RAAN) * np.cos(inc_r),
                np.cos(u) * np.cos(RAAN) - np.sin(u) * np.sin(RAAN) * np.cos(inc_r),
            )
        )
        + (nu / _TWO_PI) * P * lonDrift * _RAD2DEG
    )

    lonRangeAbs = lonRange - lonRange[midindx]

    # marginal viewing distance (single scalar)
    marginalDistance = DlonA + np.max(
        np.minimum(np.abs(lonRangeAbs), 360.0 - np.abs(lonRangeAbs))
    )
    marginalView = normDegAbs <= marginalDistance
    idx_pass, idx_lon = np.where(marginalView)

    if idx_pass.size == 0:
        t_view_min = np.zeros_like(inView, dtype=np.float32)
        t_view_max = np.zeros_like(inView, dtype=np.float32)
        inViewMarginal = inView.copy()
        return inViewMarginal, t_view_min, t_view_max

    val = normDeg[marginalView]
    lonMarg = np.mod(passLon[idx_lon], 360.0)

    # build longitude trace matrix: lonRangeAbs repeated for each pass considered
    lonTrace = np.tile(lonRangeAbs[None, :], (len(idx_lon), 1))

    # lonMargMat: center traces around pass longitude
    lonMargMat = np.mod(lonTrace + lonMarg[:, None], 360.0)

    # compute lonDiff between pass centers and marginal trace
    lonDiff = np.mod((passLon[idx_lon] + val)[:, None] - lonMargMat, 360.0)
    lonDiff = np.minimum(360.0 - lonDiff, lonDiff)

    # compute ellipse inclusion test
    a_ellipse = lonDiff**2 / (0.5 * DlonA_range) ** 2
    b_ellipse = (lat - latRange) ** 2 / (theta_range) ** 2

    marginalPass = a_ellipse + b_ellipse
    marginalIdx = np.any(marginalPass <= 1.0, axis=1)

    # build t_view (t_delta repeated for each marginal candidate)
    t_view = np.tile(t_delta[None, :], (len(marginalIdx), 1))
    t_view[~(marginalPass <= 1.0)] = np.nan

    # prepare outputs: inViewMarginal, t_view_min, t_view_max
    inViewMarginal = np.zeros_like(inView, dtype=bool)
    linear_idx = (idx_pass[marginalIdx], idx_lon[marginalIdx])
    inViewMarginal[linear_idx] = True

    t_view_min = np.zeros_like(inView, dtype=np.float32)
    t_view_max = np.zeros_like(inView, dtype=np.float32)

    minima = np.nanmin(t_view[marginalIdx], axis=1)
    maxima = np.nanmax(t_view[marginalIdx], axis=1)

    t_view_min[linear_idx] = minima
    t_view_max[linear_idx] = maxima

    # merge marginal into main inView
    inView[linear_idx] = True

    return inView, t_view_min, t_view_max


def ReShape(maxRevisit, flag, SSO, inclination, altitude, psi):
    """
    Reshapes a 1D vector maxRevisit into a multidimensional array, replicating
    MATLAB logic from the RevisitTime toolkit.
    """

    j = 0  # Python is 0-indexed

    # prepare matrix with correct shape
    # worst-case shape: (inclination, altitude, psi)
    MaxRevisit = np.zeros(
        (
            inclination if inclination > 0 else 1,
            altitude if altitude > 0 else 1,
            psi if psi > 0 else 1,
        )
    )

    if flag == 0 and SSO == 0:
        for ind_inc in range(inclination):
            for ind_alt in range(altitude):
                for ind_psi in range(psi):
                    MaxRevisit[ind_inc, ind_alt, ind_psi] = maxRevisit[j]
                    j += 1

    elif flag == 1 and SSO == 0:
        for ind_inc in range(inclination):
            for ind_alt in range(altitude):
                MaxRevisit[ind_inc, ind_alt, 0] = maxRevisit[j]
                j += 1

    elif flag == 0 and SSO == 1:
        for ind_alt in range(altitude):
            for ind_psi in range(psi):
                MaxRevisit[0, ind_alt, ind_psi] = maxRevisit[j]
                j += 1

    elif flag == 1 and SSO == 1:
        for ind_alt in range(altitude):
            MaxRevisit[0, ind_alt, 0] = maxRevisit[j]
            j += 1

    return MaxRevisit
