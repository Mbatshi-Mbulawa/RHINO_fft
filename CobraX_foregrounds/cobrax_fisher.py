"""
cobrax_fisher.py
================
Core functions for the CobraX Fisher forecast analysis.
Based on Nasirudin & Bull (2026), Section 2.1.

Author: Mbatshi Jerry Junior Mbulawa
Date:  27 March 2026

PHYSICS SUMMARY
---------------
We model the radio brightness temperature of a single sky pixel as a
moment-expanded power law (Eq. 1 of the paper), plus calibration terms.
We then use Fisher matrix forecasting to predict how well a lunar-orbit
radio telescope can constrain the 6 model parameters.

THE 6 FREE PARAMETERS (one set per pixel):
    theta = [g_408, T_offset_408, T_p0, beta_p0, m1, m2]

    g_408        : multiplicative gain error on the Haslam 408 MHz point
    T_offset_408 : additive zero-point offset on the Haslam 408 MHz point
    T_p0         : brightness temperature at pivot frequency nu_0 = 1 GHz  [K]
    beta_p0      : power-law spectral index  (fiducial: -2.75)
    m1           : 1st moment coefficient   (fiducial: 0)
    m2           : 2nd moment coefficient   (fiducial: 0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# CONSTANTS & FIDUCIAL PARAMETERS  (all taken directly from the paper)
# =============================================================================

NU_0        = 1.0e9          # Pivot frequency [Hz]  (1 GHz)
NU_HASLAM   = 0.408e9        # Haslam map frequency  [Hz]

# Fiducial pixel parameters (Section 2.1)
T_P0_FID    = 2.0            # Brightness temp at nu_0  [K]
BETA_FID    = -2.75          # Spectral index
M1_FID      = 0.0            # 1st moment coefficient
M2_FID      = 0.0            # 2nd moment coefficient

# Fiducial calibration errors on the Haslam point (Section 2.1)
G408_FID        = 1.2        # Multiplicative gain error
T_OFFSET408_FID = 1.0        # Additive offset  [K]

# Radiometer equation parameters (Section 2.1)
T_SYS       = 150.0          # System temperature  [K]
T_OBS_PIX   = 2.2 * 3600.0  # Observing time per pixel  [s]  (2.2 h)

# Noise on the Haslam data point — taken as 10% of the brightness temperature.
# The paper does not state an exact value; this is a reasonable assumption
# for the known ~tens-of-percent calibration uncertainty.
SIGMA_HASLAM = 0.1 * T_P0_FID * (NU_HASLAM / NU_0) ** BETA_FID


# =============================================================================
# TASK 1 — TEMPERATURE SPECTRUM MODEL  (Equation 1)
# =============================================================================

def temperature_spectrum(nu, T_p0, beta_p0, m1, m2,
                         g=1.0, T_offset=0.0, nu_0=NU_0):
    """
    Evaluate the brightness temperature model at frequency nu.

    This implements Equation 1 of Nasirudin & Bull (2026):

        T_p(nu) = g(nu) * T_p0 * (nu/nu_0)^beta_p0
                  * (1 + m1*ln(nu/nu_0) + m2*[ln(nu/nu_0)]^2)
                  + T_offset(nu)

    Parameters
    ----------
    nu       : float or np.ndarray — frequency in Hz
    T_p0     : float — brightness temperature at pivot frequency nu_0  [K]
    beta_p0  : float — power-law spectral index
    m1       : float — 1st moment (log-polynomial) coefficient
    m2       : float — 2nd moment (log-polynomial) coefficient
    g        : float — multiplicative gain calibration factor (default 1)
    T_offset : float — additive zero-point offset in K (default 0)
    nu_0     : float — pivot frequency in Hz (default 1 GHz)

    Returns
    -------
    T : float or np.ndarray — brightness temperature in K
    """
    # x = ln(nu / nu_0)
    # This is the natural variable for the moment expansion.
    # At nu = nu_0,  x = 0, so all moment terms vanish and T = g * T_p0.
    x = np.log(nu / nu_0)

    # Power-law term:  (nu/nu_0)^beta  =  exp(beta * ln(nu/nu_0))  =  exp(beta * x)
    power_law = (nu / nu_0) ** beta_p0

    # Moment expansion:  1 + m1*x + m2*x^2
    # At fiducial values (m1=m2=0) this equals 1, giving a pure power law.
    moment_expansion = 1.0 + m1 * x + m2 * x**2

    # Full model
    T = g * T_p0 * power_law * moment_expansion + T_offset
    return T


def plot_temperature_spectrum(nu_min=0.1e9, nu_max=10e9, n_points=500,
                               show_haslam=True, show_cobra_band=True,
                               nu_centre=1.75e9, bandwidth=0.5e9):
    """
    TASK 1: Plot the brightness temperature spectrum T(nu) using fiducial
    parameter values, with optional overlays showing the Haslam data point
    and the CobraX observing band.

    Parameters
    ----------
    nu_min         : float — minimum plot frequency  [Hz]
    nu_max         : float — maximum plot frequency  [Hz]
    n_points       : int   — number of frequency samples
    show_haslam    : bool  — mark the Haslam 408 MHz data point
    show_cobra_band: bool  — shade the CobraX observing band
    nu_centre      : float — centre frequency of CobraX band  [Hz]
    bandwidth      : float — bandwidth of CobraX band  [Hz]
    """
    nu_arr = np.logspace(np.log10(nu_min), np.log10(nu_max), n_points)

    # True (uncontaminated) spectrum
    T_true = temperature_spectrum(nu_arr, T_P0_FID, BETA_FID, M1_FID, M2_FID)

    # Contaminated Haslam point: what the map actually measures
    T_haslam_true  = temperature_spectrum(NU_HASLAM, T_P0_FID, BETA_FID, M1_FID, M2_FID)
    T_haslam_meas  = temperature_spectrum(NU_HASLAM, T_P0_FID, BETA_FID, M1_FID, M2_FID,
                                          g=G408_FID, T_offset=T_OFFSET408_FID)

    fig, ax = plt.subplots(figsize=(9, 5))

    # True spectrum
    ax.plot(nu_arr / 1e9, T_true, color='steelblue', lw=2,
            label=r'True spectrum ($g=1$, $T_{\rm off}=0$)')

    if show_haslam:
        # True value at 408 MHz
        ax.scatter(NU_HASLAM / 1e9, T_haslam_true, color='steelblue',
                   s=60, zorder=5)
        # Measured (contaminated) value at 408 MHz
        ax.scatter(NU_HASLAM / 1e9, T_haslam_meas, color='firebrick',
                   s=80, marker='*', zorder=6,
                   label=fr'Haslam measurement ($g={G408_FID}$, '
                         fr'$T_{{\rm off}}={T_OFFSET408_FID}$ K)')
        # Arrow showing the offset
        ax.annotate('', xy=(NU_HASLAM / 1e9, T_haslam_meas),
                    xytext=(NU_HASLAM / 1e9, T_haslam_true),
                    arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.5))

    if show_cobra_band:
        nu_lo = (nu_centre - bandwidth / 2) / 1e9
        nu_hi = (nu_centre + bandwidth / 2) / 1e9
        ax.axvspan(nu_lo, nu_hi, alpha=0.15, color='goldenrod',
                   label=f'CobraX band ({nu_lo:.2f}–{nu_hi:.2f} GHz)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency  [GHz]', fontsize=13)
    ax.set_ylabel('Brightness Temperature  [K]', fontsize=13)
    ax.set_title('Radio sky brightness temperature spectrum\n'
                 r'(fiducial: $T_{p,0}=2$ K, $\beta=-2.75$, $\nu_0=1$ GHz)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('task1_temperature_spectrum.png', dpi=150)
    plt.show()
    print("Saved: task1_temperature_spectrum.png")


# =============================================================================
# TASK 2 — ANALYTICAL DERIVATIVES  (the 6 partial derivatives of Eq. 1)
# =============================================================================

def compute_derivatives(nu, T_p0, beta_p0, m1, m2, nu_0=NU_0,
                        g_fid=1.0, T_offset_fid=0.0):
    """
    Compute all 6 analytical partial derivatives of the model T(nu) with
    respect to the 6 free parameters, evaluated at the given parameter values.

    These are derived analytically (pen-and-paper) as follows.

    Let:
        x   = ln(nu / nu_0)
        P   = (nu / nu_0)^beta_p0  =  exp(beta_p0 * x)
        M   = 1 + m1*x + m2*x^2       (moment expansion)

    Then the model (ignoring calibration terms for new bands) is:
        T(nu) = T_p0 * P * M

    DERIVATIVE 1 — dT/dT_p0:
        T = T_p0 * P * M
        => dT/dT_p0 = P * M = (nu/nu_0)^beta * (1 + m1*x + m2*x^2)
        At fiducial (m1=m2=0): dT/dT_p0 = (nu/nu_0)^beta

    DERIVATIVE 2 — dT/d(beta_p0):
        P = exp(beta * x), so dP/d(beta) = x * P
        T = T_p0 * P * M
        => dT/d(beta) = T_p0 * (x * P) * M = T_p0 * P * M * x
        At fiducial: dT/d(beta) = T_p0 * (nu/nu_0)^beta * ln(nu/nu_0)

    DERIVATIVE 3 — dT/d(m1):
        M = 1 + m1*x + m2*x^2,  so dM/dm1 = x
        => dT/dm1 = T_p0 * P * x
        At fiducial: dT/dm1 = T_p0 * (nu/nu_0)^beta * ln(nu/nu_0)
        NOTE: At fiducial values, dT/dm1 = dT/d(beta_p0).
              This degeneracy makes the Fisher matrix singular unless
              you have sufficient frequency lever arm.

    DERIVATIVE 4 — dT/d(m2):
        dM/dm2 = x^2
        => dT/dm2 = T_p0 * P * x^2
        At fiducial: dT/dm2 = T_p0 * (nu/nu_0)^beta * [ln(nu/nu_0)]^2

    DERIVATIVE 5 — dT/d(g_408):
        This is non-zero ONLY at nu = 408 MHz.
        At 408 MHz: T = g_408 * T_p0 * P_408 * M_408 + T_offset_408
        => dT/d(g_408) = T_p0 * P_408 * M_408
        At all other nu: dT/d(g_408) = 0

    DERIVATIVE 6 — dT/d(T_offset_408):
        This is non-zero ONLY at nu = 408 MHz.
        At 408 MHz: T = g_408 * T_p0 * P_408 * M_408 + T_offset_408
        => dT/d(T_offset_408) = 1
        At all other nu: dT/d(T_offset_408) = 0

    Parameters
    ----------
    nu       : float or np.ndarray — frequency in Hz
    T_p0     : float — brightness temperature at nu_0
    beta_p0  : float — spectral index
    m1, m2   : float — moment coefficients
    nu_0     : float — pivot frequency in Hz

    Returns
    -------
    derivs : dict with keys matching the 6 parameter names,
             each value being a float or array of same shape as nu.
    """
    x   = np.log(nu / nu_0)                     # ln(nu/nu_0)
    P   = (nu / nu_0) ** beta_p0                 # power law factor
    M   = 1.0 + m1 * x + m2 * x**2              # moment expansion

    # Determine the effective gain at each frequency.
    # At satellite frequencies g is fixed to 1 (not a free parameter).
    # At the Haslam 408 MHz point, g = g_fid (its fiducial value).
    # Derivatives 1-4 are dT/dtheta for the spectral parameters.
    # The full model is T = g * T_p0 * P * M + T_off, so:
    #   dT/dT_p0  = g * P * M
    #   dT/dbeta  = g * T_p0 * P * M * x
    #   dT/dm1    = g * T_p0 * P * x
    #   dT/dm2    = g * T_p0 * P * x^2
    # At satellite channels g=1 so g drops out. At Haslam g=g_fid.
    nu_is_array = isinstance(nu, np.ndarray)

    if nu_is_array:
        is_haslam = np.isclose(nu, NU_HASLAM, rtol=1e-3)
        g_eff = np.where(is_haslam, g_fid, 1.0)
    else:
        g_eff = g_fid if np.isclose(nu, NU_HASLAM, rtol=1e-3) else 1.0

    # Derivatives 1–4: include g_eff so that they are correct at ALL frequencies
    dT_dTp0   = g_eff * P * M                    # dT/dT_p0
    dT_dbeta  = g_eff * T_p0 * P * M * x        # dT/d(beta)
    dT_dm1    = g_eff * T_p0 * P * x            # dT/dm1
    dT_dm2    = g_eff * T_p0 * P * x**2         # dT/dm2

    # Derivatives 5–6: only non-zero at 408 MHz
    if nu_is_array:
        x_408 = np.log(NU_HASLAM / nu_0)
        P_408 = (NU_HASLAM / nu_0) ** beta_p0
        M_408 = 1.0 + m1 * x_408 + m2 * x_408**2

        dT_dg408    = np.where(is_haslam, T_p0 * P_408 * M_408, 0.0)
        dT_dToff408 = np.where(is_haslam, 1.0, 0.0)
    else:
        if np.isclose(nu, NU_HASLAM, rtol=1e-3):
            x_408 = np.log(NU_HASLAM / nu_0)
            P_408 = (NU_HASLAM / nu_0) ** beta_p0
            M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
            dT_dg408    = T_p0 * P_408 * M_408
            dT_dToff408 = 1.0
        else:
            dT_dg408    = 0.0
            dT_dToff408 = 0.0

    return {
        'T_p0'         : dT_dTp0,
        'beta_p0'      : dT_dbeta,
        'm1'           : dT_dm1,
        'm2'           : dT_dm2,
        'g_408'        : dT_dg408,
        'T_offset_408' : dT_dToff408,
    }


# =============================================================================
# RADIOMETER NOISE (Equation 4)
# =============================================================================

def noise_rms(delta_nu, T_sys=T_SYS, t_obs=T_OBS_PIX):
    """
    Compute the noise rms per frequency channel using the radiometer equation
    (Eq. 4 of the paper):

        sigma_nu = T_sys / sqrt(delta_nu * t_obs)

    Parameters
    ----------
    delta_nu : float — channel bandwidth in Hz
    T_sys    : float — system temperature in K  (default 150 K)
    t_obs    : float — observing time per pixel in s  (default 2.2 h)

    Returns
    -------
    sigma : float — noise rms in K
    """
    return T_sys / np.sqrt(delta_nu * t_obs)


# =============================================================================
# TASK 2 — FISHER MATRIX (Equation 3)
# =============================================================================

def build_frequency_grid(nu_centre, bandwidth, delta_nu):
    """
    Build the array of frequency channel centres for the satellite instrument.

    The band runs from (nu_centre - bandwidth/2) to (nu_centre + bandwidth/2),
    with channels of width delta_nu.

    Parameters
    ----------
    nu_centre : float — band centre frequency  [Hz]
    bandwidth : float — total bandwidth  [Hz]
    delta_nu  : float — channel width  [Hz]

    Returns
    -------
    nu_arr : np.ndarray — channel centre frequencies  [Hz]
    """
    nu_lo = nu_centre - bandwidth / 2.0
    nu_hi = nu_centre + bandwidth / 2.0
    # linspace from first channel centre to last channel centre
    n_channels = int(round(bandwidth / delta_nu))
    nu_arr = np.linspace(nu_lo + delta_nu / 2.0,
                         nu_hi - delta_nu / 2.0,
                         n_channels)
    return nu_arr


def compute_fisher_matrix(nu_centre, bandwidth, delta_nu,
                           T_p0=T_P0_FID, beta_p0=BETA_FID,
                           m1=M1_FID, m2=M2_FID,
                           T_sys=T_SYS, t_obs=T_OBS_PIX,
                           sigma_haslam=SIGMA_HASLAM):
    """
    Compute the 6x6 Fisher information matrix F_ab (Equation 3).

    The matrix is constructed as:
        F_ab = sum_i  (1 / sigma_i^2) * (dT/dtheta_a)|_i * (dT/dtheta_b)|_i

    where the sum is over all data points i (Haslam + satellite channels).

    The parameter ordering in the 6x6 matrix is:
        0: g_408
        1: T_offset_408
        2: T_p0
        3: beta_p0
        4: m1
        5: m2

    Parameters
    ----------
    nu_centre   : float — satellite band centre frequency  [Hz]
    bandwidth   : float — satellite band total bandwidth   [Hz]
    delta_nu    : float — satellite channel width          [Hz]
    T_p0 ...    : fiducial parameter values (see module-level constants)
    sigma_haslam: float — noise on Haslam data point  [K]

    Returns
    -------
    F   : np.ndarray, shape (6, 6) — Fisher matrix
    nu_all : np.ndarray — all frequency points used (Haslam + satellite)
    sigma_all : np.ndarray — noise at each frequency point
    """
    # --- Build full frequency grid (Haslam + satellite channels) ---
    nu_sat    = build_frequency_grid(nu_centre, bandwidth, delta_nu)
    sigma_sat = noise_rms(delta_nu, T_sys, t_obs) * np.ones_like(nu_sat)

    # Prepend the Haslam 408 MHz point
    nu_all    = np.concatenate([[NU_HASLAM], nu_sat])
    sigma_all = np.concatenate([[sigma_haslam], sigma_sat])

    # --- Build the derivative matrix D of shape (N_data, 6) ---
    # D[i, a] = dT/dtheta_a evaluated at frequency nu_all[i]
    param_names = ['g_408', 'T_offset_408', 'T_p0', 'beta_p0', 'm1', 'm2']
    N_data   = len(nu_all)
    N_params = 6
    D = np.zeros((N_data, N_params))

    for i, nu_i in enumerate(nu_all):
        derivs = compute_derivatives(nu_i, T_p0, beta_p0, m1, m2)
        for j, name in enumerate(param_names):
            D[i, j] = derivs[name]

    # --- Build the noise covariance matrix N (diagonal) ---
    # N is N_data x N_data, diagonal entries = sigma_i^2
    N_cov = np.diag(sigma_all**2)
    N_inv = np.diag(1.0 / sigma_all**2)   # Inverse is trivial for diagonal

    # --- Compute Fisher matrix via matrix multiplication ---
    # F = D^T * N^{-1} * D
    # This is the discretised form of Equation 3.
    # Shape: (6, N_data) * (N_data, N_data) * (N_data, 6) = (6, 6)
    F = D.T @ N_inv @ D

    return F, nu_all, sigma_all


def compute_parameter_uncertainties(F):
    """
    Invert the Fisher matrix to get the parameter covariance matrix,
    then extract the diagonal (variances) and take square roots (1-sigma errors).

    C = F^{-1}
    sigma_a = sqrt(C_aa)

    If F is singular (not invertible), returns NaN for all uncertainties.
    This happens when parameters are degenerate — a physically important signal.

    Parameters
    ----------
    F : np.ndarray, shape (6, 6) — Fisher matrix

    Returns
    -------
    sigmas : np.ndarray, shape (6,) — 1-sigma uncertainties on each parameter
    C      : np.ndarray, shape (6, 6) — covariance matrix (or None if singular)
    """
    try:
        # Check condition number before inverting.
        # A very large condition number means the matrix is nearly singular —
        # i.e. some parameters are nearly degenerate.
        cond = np.linalg.cond(F)
        if cond > 1e15:
            print(f"  WARNING: Fisher matrix is nearly singular (cond={cond:.2e}). "
                  "Parameters are degenerate at this configuration.")
            return np.full(6, np.nan), None

        C = np.linalg.inv(F)
        sigmas = np.sqrt(np.diag(C))
        return sigmas, C

    except np.linalg.LinAlgError:
        return np.full(6, np.nan), None


# =============================================================================
# TASK 3 — FISHER FORECAST PLOTS  (Figure 3-style)
# =============================================================================

def run_fisher_forecast_grid(nu_centres, bandwidths,
                              delta_nu=25e6,
                              param_idx=3,
                              param_name=r'$\sigma(\beta_{p,0})$'):
    """
    Sweep over a 2D grid of (nu_centre, bandwidth) values and compute
    the predicted 1-sigma uncertainty on one chosen parameter.
    This produces a Figure 3-style plot.

    Parameters
    ----------
    nu_centres  : array-like — centre frequencies to sweep  [Hz]
    bandwidths  : array-like — bandwidths to sweep  [Hz]
    delta_nu    : float — channel width (fixed)  [Hz]
    param_idx   : int   — which parameter (0–5) to plot uncertainty for
    param_name  : str   — label for the colour bar

    Returns
    -------
    sigma_grid : np.ndarray, shape (len(bandwidths), len(nu_centres))
                 Predicted uncertainty at each grid point.
    """
    nu_centres = np.asarray(nu_centres)
    bandwidths = np.asarray(bandwidths)

    sigma_grid = np.zeros((len(bandwidths), len(nu_centres)))

    for i, bw in enumerate(bandwidths):
        for j, nu_c in enumerate(nu_centres):
            # Skip cases where band would go below 0 Hz
            if nu_c - bw / 2 < 1e6:
                sigma_grid[i, j] = np.nan
                continue
            try:
                F, _, _ = compute_fisher_matrix(nu_c, bw, delta_nu)
                sigmas, _ = compute_parameter_uncertainties(F)
                sigma_grid[i, j] = sigmas[param_idx]
            except Exception:
                sigma_grid[i, j] = np.nan

    return sigma_grid


def plot_fisher_forecast(nu_centres, bandwidths, delta_nu=25e6,
                         param_idx=3,
                         param_name=r'$\sigma(\beta_{p,0})$  [dimensionless]'):
    """
    Produce a 2D colour map of predicted parameter uncertainty as a function
    of band centre frequency and bandwidth — analogous to Figure 3 of the paper.

    Parameters
    ----------
    nu_centres : array-like — centre frequencies  [Hz]
    bandwidths : array-like — bandwidths  [Hz]
    delta_nu   : float — channel width  [Hz]
    param_idx  : int   — which parameter index to display (0–5)
    param_name : str   — colour bar label
    """
    sigma_grid = run_fisher_forecast_grid(nu_centres, bandwidths,
                                          delta_nu, param_idx, param_name)

    # Convert axes to GHz and MHz for readable tick labels
    nu_GHz = np.array(nu_centres) / 1e9
    bw_MHz = np.array(bandwidths) / 1e6

    fig, ax = plt.subplots(figsize=(8, 6))

    # pcolormesh expects edges, not centres, so we pad by half a cell
    # on each side to keep ticks aligned with cell centres.
    dnu = (nu_GHz[1] - nu_GHz[0]) / 2 if len(nu_GHz) > 1 else 0.05
    dbw = (bw_MHz[1] - bw_MHz[0]) / 2 if len(bw_MHz) > 1 else 25.0
    nu_edges = np.concatenate([[nu_GHz[0] - dnu],
                                (nu_GHz[:-1] + nu_GHz[1:]) / 2,
                                [nu_GHz[-1] + dnu]])
    bw_edges = np.concatenate([[bw_MHz[0] - dbw],
                                (bw_MHz[:-1] + bw_MHz[1:]) / 2,
                                [bw_MHz[-1] + dbw]])

    # Use log scale for colour axis so dynamic range is visible
    from matplotlib.colors import LogNorm
    valid = sigma_grid[np.isfinite(sigma_grid) & (sigma_grid > 0)]
    vmin  = valid.min() if len(valid) > 0 else 1e-5
    vmax  = valid.max() if len(valid) > 0 else 1.0

    pcm = ax.pcolormesh(nu_edges, bw_edges, sigma_grid,
                         norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap='viridis_r', shading='flat')

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(param_name, fontsize=12)

    ax.set_xlabel('Band centre frequency  [GHz]', fontsize=13)
    ax.set_ylabel('Bandwidth  [MHz]', fontsize=13)
    ax.set_title(f'Fisher forecast: {param_name}\n'
                 f'(channel width = {delta_nu/1e6:.0f} MHz)',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'task3_fisher_forecast_param{param_idx}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")
    return sigma_grid


# =============================================================================
# BONUS — PLOT ALL 6 PARAMETER UNCERTAINTIES for a single configuration
# =============================================================================

def plot_all_uncertainties_single_config(nu_centre, bandwidth, delta_nu=25e6):
    """
    For a single (nu_centre, bandwidth, delta_nu) configuration, compute and
    display the predicted 1-sigma uncertainty on all 6 parameters.

    Also prints the Fisher matrix and its inverse for inspection.

    Parameters
    ----------
    nu_centre : float — band centre frequency  [Hz]
    bandwidth : float — total bandwidth  [Hz]
    delta_nu  : float — channel width  [Hz]
    """
    param_labels = [r'$g_{408}$', r'$T_{\rm off,408}$ [K]',
                    r'$T_{p,0}$ [K]', r'$\beta_{p,0}$',
                    r'$m_1$', r'$m_2$']

    F, nu_all, sigma_all = compute_fisher_matrix(nu_centre, bandwidth, delta_nu)
    sigmas, C = compute_parameter_uncertainties(F)

    print("\n" + "="*55)
    print(f"Configuration: nu_c={nu_centre/1e9:.2f} GHz, "
          f"BW={bandwidth/1e6:.0f} MHz, "
          f"delta_nu={delta_nu/1e6:.0f} MHz")
    print(f"Number of data points (Haslam + satellite): {len(nu_all)}")
    print("="*55)
    print("\nFisher matrix F  (6x6):")
    print(np.array2string(F, precision=3, suppress_small=True))
    if C is not None:
        print("\nCovariance matrix C = F^{-1}  (6x6):")
        print(np.array2string(C, precision=6, suppress_small=True))
        print("\n1-sigma parameter uncertainties:")
        for label, sig in zip(param_labels, sigmas):
            print(f"  {label:30s}:  {sig:.6f}")

    # Bar chart of uncertainties
    if np.any(np.isfinite(sigmas)):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['firebrick', 'firebrick', 'steelblue', 'steelblue',
                  'goldenrod', 'goldenrod']
        bars = ax.bar(range(6), sigmas, color=colors, edgecolor='k', alpha=0.8)
        ax.set_xticks(range(6))
        ax.set_xticklabels(param_labels, fontsize=11)
        ax.set_ylabel(r'1$\sigma$ uncertainty', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'Parameter uncertainties\n'
                     f'($\\nu_c$={nu_centre/1e9:.2f} GHz, '
                     f'BW={bandwidth/1e6:.0f} MHz, '
                     f'$\\delta\\nu$={delta_nu/1e6:.0f} MHz)',
                     fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('task3_uncertainties_single_config.png', dpi=150)
        plt.show()
        print("Saved: task3_uncertainties_single_config.png")