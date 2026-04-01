"""
cobrax_fisher.py
================
CobraX Fisher forecast — beta_p0 FIXED at -2.75 (supervisor instruction).

PARAMETER SUMMARY
-----------------
beta_p0 is FIXED at -2.75. Not a free parameter. Not in the Fisher matrix.

The 5 remaining free parameters are:
    [g_408, T_offset_408, T_p0, m1, m2]

However, the full 5x5 Fisher matrix is numerically singular because
g_408 and T_offset_408 are constrained only by a SINGLE data point
(the Haslam 408 MHz measurement). This creates a near-rank-deficient
coupling that makes the full 5x5 matrix non-invertible without a prior.

Since Phil says no prior is needed, the working assumption is that
we forecast for the 3 SPECTRAL parameters only:
    theta_spectral = [T_p0, m1, m2]

The calibration parameters (g_408, T_offset_408) are treated as
nuisance parameters handled separately (by the Haslam recalibration
itself), not as unknowns to be forecast here.

The full 5x5 matrix is still built and documented. The 3x3 spectral
block is the object we invert for the forecast plots.

THIS IS FLAGGED AS AN OPEN QUESTION FOR PHIL.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

NU_0        = 1.0e9           # Pivot frequency [Hz]
NU_HASLAM   = 0.408e9         # Haslam map frequency [Hz]
BETA_FIXED  = -2.75           # Spectral index — FIXED, not a free parameter

# Fiducial values
T_P0_FID        = 2.0         # Brightness temp at nu_0 [K]
M1_FID          = 0.0         # 1st moment coefficient
M2_FID          = 0.0         # 2nd moment coefficient
G408_FID        = 1.2         # Gain error on Haslam (fiducial)
T_OFFSET408_FID = 1.0         # Additive offset on Haslam [K] (fiducial)

T_SYS       = 150.0           # System temperature [K]
T_OBS_PIX   = 2.2 * 3600.0   # Observing time per pixel [s]

SIGMA_HASLAM = 0.1 * T_P0_FID * (NU_HASLAM / NU_0) ** BETA_FIXED

# Full 5-parameter system
PARAM_NAMES_5  = ['g_408', 'T_offset_408', 'T_p0', 'm1', 'm2']
PARAM_LABELS_5 = [
    r'$g_{408}$',
    r'$T_{\rm off,408}$ [K]',
    r'$T_{p,0}$ [K]',
    r'$m_1$',
    r'$m_2$',
]
N_PARAMS_5 = 5

# 3-parameter spectral system (working default — see module docstring)
PARAM_NAMES_3  = ['T_p0', 'm1', 'm2']
PARAM_LABELS_3 = [r'$T_{p,0}$ [K]', r'$m_1$', r'$m_2$']
N_PARAMS_3 = 3


# =============================================================================
# TEMPERATURE SPECTRUM MODEL
# =============================================================================

def temperature_spectrum(nu, T_p0, m1, m2,
                         g=1.0, T_offset=0.0,
                         nu_0=NU_0, beta=BETA_FIXED):
    """
    Evaluate brightness temperature model at frequency nu.

    T_p(nu) = g * T_p0 * (nu/nu_0)^beta * (1 + m1*x + m2*x^2) + T_offset
    where x = ln(nu/nu_0) and beta is FIXED.
    """
    x                = np.log(nu / nu_0)
    power_law        = (nu / nu_0) ** beta
    moment_expansion = 1.0 + m1 * x + m2 * x**2
    return g * T_p0 * power_law * moment_expansion + T_offset


# =============================================================================
# ANALYTICAL DERIVATIVES
# (5 derivatives — dT/d(beta) removed since beta is fixed)
# =============================================================================

def compute_derivatives(nu, T_p0, m1, m2,
                        nu_0=NU_0, beta=BETA_FIXED,
                        g_fid=1.0, T_offset_fid=0.0):
    """
    Compute all 5 analytical partial derivatives of T_p(nu).
    beta is FIXED so dT/d(beta) is NOT included.

    Shorthand: x = ln(nu/nu_0), P = (nu/nu_0)^beta [fixed], M = 1+m1*x+m2*x^2

    (1) dT/dT_p0  = g_eff * P * M         -> at fid (M=1): g_eff * P
    (2) dT/dm1    = g_eff * T_p0 * P * x  -> at fid: g_eff * T_p0 * P * x
    (3) dT/dm2    = g_eff * T_p0 * P * x^2
    (4) dT/dg_408     = T_p0*P_408*M_408 at 408 MHz, else 0
    (5) dT/dT_off_408 = 1 at 408 MHz, else 0

    g_eff = g_fid at 408 MHz, = 1 at satellite frequencies.
    """
    x   = np.log(nu / nu_0)
    P   = (nu / nu_0) ** beta
    M   = 1.0 + m1 * x + m2 * x**2

    nu_is_array = isinstance(nu, np.ndarray)

    if nu_is_array:
        is_haslam = np.isclose(nu, NU_HASLAM, rtol=1e-3)
        g_eff     = np.where(is_haslam, g_fid, 1.0)
    else:
        g_eff = g_fid if np.isclose(nu, NU_HASLAM, rtol=1e-3) else 1.0

    # Derivatives 1-3: active at all frequencies
    dT_dTp0 = g_eff * P * M
    dT_dm1  = g_eff * T_p0 * P * x
    dT_dm2  = g_eff * T_p0 * P * x**2

    # Derivatives 4-5: active only at 408 MHz
    if nu_is_array:
        x_408 = np.log(NU_HASLAM / nu_0)
        P_408 = (NU_HASLAM / nu_0) ** beta
        M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
        dT_dg408    = np.where(is_haslam, T_p0 * P_408 * M_408, 0.0)
        dT_dToff408 = np.where(is_haslam, 1.0, 0.0)
    else:
        if np.isclose(nu, NU_HASLAM, rtol=1e-3):
            x_408 = np.log(NU_HASLAM / nu_0)
            P_408 = (NU_HASLAM / nu_0) ** beta
            M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
            dT_dg408    = T_p0 * P_408 * M_408
            dT_dToff408 = 1.0
        else:
            dT_dg408    = 0.0
            dT_dToff408 = 0.0

    return {
        'g_408'        : dT_dg408,
        'T_offset_408' : dT_dToff408,
        'T_p0'         : dT_dTp0,
        'm1'           : dT_dm1,
        'm2'           : dT_dm2,
    }


# =============================================================================
# RADIOMETER NOISE
# =============================================================================

def noise_rms(delta_nu, T_sys=T_SYS, t_obs=T_OBS_PIX):
    """sigma = T_sys / sqrt(delta_nu * t_obs)"""
    return T_sys / np.sqrt(delta_nu * t_obs)


# =============================================================================
# FREQUENCY GRID
# =============================================================================

def build_frequency_grid(nu_centre, bandwidth, delta_nu):
    """Build array of satellite channel centre frequencies."""
    n_channels = int(round(bandwidth / delta_nu))
    nu_lo      = nu_centre - bandwidth / 2.0
    nu_arr     = np.linspace(nu_lo + delta_nu / 2.0,
                             nu_lo + delta_nu / 2.0 + (n_channels - 1) * delta_nu,
                             n_channels)
    return nu_arr


# =============================================================================
# FULL 5x5 FISHER MATRIX  (documented but numerically singular)
# =============================================================================

def compute_fisher_matrix_5x5(nu_centre, bandwidth, delta_nu,
                               T_p0=T_P0_FID, m1=M1_FID, m2=M2_FID,
                               T_sys=T_SYS, t_obs=T_OBS_PIX,
                               sigma_haslam=SIGMA_HASLAM):
    """
    Compute the full 5x5 Fisher matrix for all 5 free parameters.

    WARNING: This matrix is numerically singular in most configurations
    because g_408 and T_offset_408 are constrained only by a single
    data point (Haslam 408 MHz), creating a near-degenerate coupling.
    The condition number is typically ~10^18 to 10^19 and direct
    inversion fails. This is documented for completeness; the 3x3
    spectral block is used for the actual forecast.

    Parameter ordering: [g_408, T_offset_408, T_p0, m1, m2]
    """
    nu_sat    = build_frequency_grid(nu_centre, bandwidth, delta_nu)
    sigma_sat = noise_rms(delta_nu, T_sys, t_obs) * np.ones_like(nu_sat)
    nu_all    = np.concatenate([[NU_HASLAM], nu_sat])
    sigma_all = np.concatenate([[sigma_haslam], sigma_sat])

    D = np.zeros((len(nu_all), N_PARAMS_5))
    for i, nu_i in enumerate(nu_all):
        g_fid_i = G408_FID if np.isclose(nu_i, NU_HASLAM, rtol=1e-3) else 1.0
        derivs  = compute_derivatives(nu_i, T_p0, m1, m2, g_fid=g_fid_i)
        for j, name in enumerate(PARAM_NAMES_5):
            D[i, j] = derivs[name]

    N_inv = np.diag(1.0 / sigma_all**2)
    F     = D.T @ N_inv @ D

    return F, nu_all, sigma_all


# =============================================================================
# 3x3 SPECTRAL FISHER MATRIX  (working default — well-conditioned)
# =============================================================================

def compute_fisher_matrix_3x3(nu_centre, bandwidth, delta_nu,
                               T_p0=T_P0_FID, m1=M1_FID, m2=M2_FID,
                               T_sys=T_SYS, t_obs=T_OBS_PIX,
                               include_haslam=True,
                               sigma_haslam=SIGMA_HASLAM):
    """
    Compute the 3x3 Fisher matrix for the spectral parameters only:
        theta_spectral = [T_p0, m1, m2]

    The calibration parameters g_408 and T_offset_408 are excluded.
    This is the well-conditioned block that can be directly inverted.

    This is equivalent to assuming the Haslam calibration errors are
    either known or handled separately, and we are forecasting only
    how well the spectral shape can be measured.

    Parameter ordering: [T_p0, m1, m2]

    Parameters
    ----------
    include_haslam : bool — whether to include Haslam point in forecast.
                    If True, uses only the spectral-parameter derivatives
                    at 408 MHz (not the calibration derivatives).
    """
    nu_sat    = build_frequency_grid(nu_centre, bandwidth, delta_nu)
    sigma_sat = noise_rms(delta_nu, T_sys, t_obs) * np.ones_like(nu_sat)

    if include_haslam:
        nu_all    = np.concatenate([[NU_HASLAM], nu_sat])
        sigma_all = np.concatenate([[sigma_haslam], sigma_sat])
    else:
        nu_all    = nu_sat
        sigma_all = sigma_sat

    D = np.zeros((len(nu_all), N_PARAMS_3))
    for i, nu_i in enumerate(nu_all):
        # For the 3x3 block, use g_fid at 408 MHz so the T_p0 derivative
        # is correct at the Haslam frequency
        g_fid_i = G408_FID if np.isclose(nu_i, NU_HASLAM, rtol=1e-3) else 1.0
        derivs  = compute_derivatives(nu_i, T_p0, m1, m2, g_fid=g_fid_i)
        D[i, 0] = derivs['T_p0']
        D[i, 1] = derivs['m1']
        D[i, 2] = derivs['m2']

    N_inv = np.diag(1.0 / sigma_all**2)
    F     = D.T @ N_inv @ D

    return F, nu_all, sigma_all


# =============================================================================
# DEFAULT: compute_fisher_matrix points to 3x3
# (pending Phil's clarification on the full 5x5 singularity)
# =============================================================================

def compute_fisher_matrix(nu_centre, bandwidth, delta_nu, **kwargs):
    """
    Default Fisher matrix computation.
    Currently uses the 3x3 spectral block [T_p0, m1, m2].
    The full 5x5 matrix is available via compute_fisher_matrix_5x5().
    """
    return compute_fisher_matrix_3x3(nu_centre, bandwidth, delta_nu, **kwargs)

# For convenience, expose relevant constants and param info for current system
PARAM_NAMES  = PARAM_NAMES_3
PARAM_LABELS = PARAM_LABELS_3
N_PARAMS     = N_PARAMS_3


# =============================================================================
# INVERT FISHER MATRIX AND EXTRACT STANDARD DEVIATIONS
# =============================================================================

def compute_parameter_uncertainties(F):
    """
    Invert the Fisher matrix. Return 1-sigma uncertainties and covariance.

    For the 3x3 spectral block this should work directly without a prior.
    """
    try:
        cond = np.linalg.cond(F)
        if cond > 1e15:
            print(f"  WARNING: Fisher matrix nearly singular (cond={cond:.2e}).")
            return np.full(F.shape[0], np.nan), None
        C      = np.linalg.inv(F)
        sigmas = np.sqrt(np.diag(C))
        return sigmas, C
    except np.linalg.LinAlgError:
        return np.full(F.shape[0], np.nan), None


# =============================================================================
# FISHER FORECAST GRID (Figure 3-style)
# =============================================================================

def run_fisher_forecast_grid(nu_centres, bandwidths,
                              delta_nu=25e6,
                              param_idx=0,
                              param_name=r'$\sigma(T_{p,0})$'):
    """
    Sweep over (nu_centre, bandwidth) grid and compute predicted
    1-sigma uncertainty for one parameter.

    For the 3x3 system:
        param_idx 0 = T_p0
        param_idx 1 = m1
        param_idx 2 = m2
    """
    nu_centres = np.asarray(nu_centres)
    bandwidths = np.asarray(bandwidths)
    sigma_grid = np.zeros((len(bandwidths), len(nu_centres)))

    for i, bw in enumerate(bandwidths):
        for j, nu_c in enumerate(nu_centres):
            if nu_c - bw / 2 < 1e6:
                sigma_grid[i, j] = np.nan
                continue
            try:
                F, _, _          = compute_fisher_matrix(nu_c, bw, delta_nu)
                sigmas, _        = compute_parameter_uncertainties(F)
                sigma_grid[i, j] = sigmas[param_idx]
            except Exception:
                sigma_grid[i, j] = np.nan

    return sigma_grid


# =============================================================================
# DIAGNOSTIC PLOT: single-configuration uncertainties
# =============================================================================

def plot_all_uncertainties_single_config(nu_centre, bandwidth, delta_nu=25e6):
    """Print and plot uncertainties for all 3 spectral parameters."""
    F, nu_all, _ = compute_fisher_matrix(nu_centre, bandwidth, delta_nu)
    sigmas, C    = compute_parameter_uncertainties(F)

    print(f"\n{'='*60}")
    print(f"  3x3 Spectral Fisher Forecast  (beta FIXED at {BETA_FIXED})")
    print(f"  Parameters: T_p0, m1, m2")
    print(f"  nu_c={nu_centre/1e9:.2f} GHz | "
          f"BW={bandwidth/1e6:.0f} MHz | "
          f"delta_nu={delta_nu/1e6:.0f} MHz")
    print(f"  N_data = {len(nu_all)}")
    print(f"{'='*60}")
    print("\nFisher matrix F (3x3):")
    print(np.array2string(F, precision=3, suppress_small=True))
    print(f"Condition number: {np.linalg.cond(F):.2e}")

    if C is not None:
        print("\nCovariance matrix C = F^{-1} (3x3):")
        print(np.array2string(C, precision=6, suppress_small=True))
        print("\n1-sigma uncertainties:")
        for lbl, sig in zip(PARAM_LABELS_3, sigmas):
            print(f"  {lbl:25s}: {sig:.4e}")

    if np.any(np.isfinite(sigmas)):
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['steelblue', 'goldenrod', 'firebrick']
        ax.bar(range(N_PARAMS_3), sigmas, color=colors, edgecolor='k', alpha=0.85)
        ax.set_xticks(range(N_PARAMS_3))
        ax.set_xticklabels(PARAM_LABELS_3, fontsize=12)
        ax.set_ylabel(r'$1\sigma$ uncertainty', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(
            rf'Spectral parameter uncertainties ($\beta$ fixed at {BETA_FIXED})'
            f'\n$\\nu_c$={nu_centre/1e9:.2f} GHz, BW={bandwidth/1e6:.0f} MHz',
            fontsize=11
        )
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('fisher_3param_single_config.png', dpi=150)
        plt.show()
        print("Saved: fisher_3param_single_config.png")