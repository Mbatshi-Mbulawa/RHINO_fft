"""
cobrax_fisher.py
================
CobraX Fisher forecast.

CHANGES FROM PHIL (April 2026):
  1. Haslam noise: sigma_408 = 800 mK = 0.8 K  (was ~2.35 K)
  2. T_offset_408 EXCLUDED from parameter vector (keep g_408 only)
  3. Add a 5 GHz anchor data point (analogous to Haslam)
  4. beta_p0 remains FIXED at -2.75

PARAMETER VECTOR (4 free parameters):
    [g_408, T_p0, m1, m2]
    T_offset_408 is excluded per Phil's instruction.

DATA POINTS USED IN FISHER MATRIX:
    - Haslam 408 MHz  (sigma = 800 mK,  g_408 is a free parameter)
    - 5 GHz anchor    (sigma = TBD,      g = 1, T_off = 0)
    - N satellite channels in the CobraX band

THE WORKING FISHER MATRIX IS 3x3 (spectral block: T_p0, m1, m2).
g_408 constrained only by the Haslam point — still excluded from
the invertible block but included in the full 4x4 for documentation.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

NU_0        = 1.0e9           # Pivot frequency [Hz]
NU_HASLAM   = 0.408e9         # Haslam map frequency [Hz]
NU_5GHZ     = 5.0e9           # C-BASS frequency [Hz] — centred at 5 GHz
BETA_FIXED  = -2.75           # Spectral index — FIXED

# Fiducial values
T_P0_FID    = 2.0             # Brightness temp at nu_0 [K]
M1_FID      = 0.0
M2_FID      = 0.0
G408_FID    = 1.2             # Gain error on Haslam (fiducial)
# T_offset_408 is EXCLUDED per Phil — no longer in the model

T_SYS       = 150.0           # System temperature [K]
T_OBS_PIX   = 2.2 * 3600.0   # Observing time per pixel [s]

# Noise values
SIGMA_HASLAM = 0.8            # [K] — 800 mK, as specified by Phil

# C-BASS noise: Phil says assume 0.1 mK or 1 mK per pixel rms.
# The C-BASS published specification is <0.1 mK/beam rms (cbass.web.ox.ac.uk).
# We use 0.1 mK as the optimistic assumption; 1 mK as conservative.
# Change SIGMA_CBASS below to switch between the two.
SIGMA_CBASS_OPTIMISTIC  = 0.1e-3   # [K] — 0.1 mK (published C-BASS spec)
SIGMA_CBASS_CONSERVATIVE = 1.0e-3  # [K] — 1 mK (conservative)
SIGMA_5GHZ = SIGMA_CBASS_OPTIMISTIC   # default: use optimistic value

# Parameter name lists
PARAM_NAMES_4  = ['g_408', 'T_p0', 'm1', 'm2']   # full 4-param (T_off excluded)
PARAM_LABELS_4 = [r'$g_{408}$', r'$T_{p,0}$ [K]', r'$m_1$', r'$m_2$']
N_PARAMS_4 = 4

PARAM_NAMES_3  = ['T_p0', 'm1', 'm2']             # working spectral block
PARAM_LABELS_3 = [r'$T_{p,0}$ [K]', r'$m_1$', r'$m_2$']
N_PARAMS_3 = 3

# Default aliases
PARAM_NAMES  = PARAM_NAMES_3
PARAM_LABELS = PARAM_LABELS_3
N_PARAMS     = N_PARAMS_3


# =============================================================================
# TEMPERATURE SPECTRUM MODEL
# =============================================================================

def temperature_spectrum(nu, T_p0, m1, m2,
                         g=1.0, T_offset=0.0,
                         nu_0=NU_0, beta=BETA_FIXED):
    """
    T_p(nu) = g * T_p0 * (nu/nu_0)^beta * (1 + m1*x + m2*x^2) + T_offset
    beta is FIXED. T_offset included for backward compatibility but
    T_offset_408 is no longer a free parameter.
    """
    x   = np.log(nu / nu_0)
    P   = (nu / nu_0) ** beta
    M   = 1.0 + m1 * x + m2 * x**2
    return g * T_p0 * P * M + T_offset


# =============================================================================
# ANALYTICAL DERIVATIVES
# 4 derivatives remain: dT/dT_p0, dT/dm1, dT/dm2, dT/dg_408
# dT/dT_offset_408 is removed (T_off excluded per Phil)
# dT/dbeta is removed (beta fixed)
# =============================================================================

def compute_derivatives(nu, T_p0, m1, m2,
                        nu_0=NU_0, beta=BETA_FIXED,
                        g_fid=1.0, T_offset_fid=0.0):
    """
    Compute analytical partial derivatives of T_p(nu) w.r.t. free parameters.

    FREE PARAMETERS (4 total, T_off excluded):
        T_p0, m1, m2  -- at all frequencies
        g_408         -- only at 408 MHz (zero elsewhere)

    DERIVATIVES:
    (1) dT/dT_p0 = g_eff * P * M       -> fid: g_eff * (nu/nu_0)^beta
    (2) dT/dm1   = g_eff * T_p0 * P * x
    (3) dT/dm2   = g_eff * T_p0 * P * x^2
    (4) dT/dg_408 = T_p0 * P_408 * M_408  at nu=408 MHz, else 0

    g_eff = g_fid at Haslam 408 MHz, = 1.0 everywhere else.
    The 5 GHz anchor point uses g=1 (assumed perfectly calibrated).
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

    # Derivative 4: g_408 only at 408 MHz
    if nu_is_array:
        x_408 = np.log(NU_HASLAM / nu_0)
        P_408 = (NU_HASLAM / nu_0) ** beta
        M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
        dT_dg408 = np.where(is_haslam, T_p0 * P_408 * M_408, 0.0)
    else:
        if np.isclose(nu, NU_HASLAM, rtol=1e-3):
            x_408 = np.log(NU_HASLAM / nu_0)
            P_408 = (NU_HASLAM / nu_0) ** beta
            M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
            dT_dg408 = T_p0 * P_408 * M_408
        else:
            dT_dg408 = 0.0

    return {
        'g_408' : dT_dg408,
        'T_p0'  : dT_dTp0,
        'm1'    : dT_dm1,
        'm2'    : dT_dm2,
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
# 3x3 SPECTRAL FISHER MATRIX
# Now includes: Haslam (800 mK), 5 GHz anchor, satellite channels
# =============================================================================

def compute_fisher_matrix_3x3(nu_centre, bandwidth, delta_nu,
                               T_p0=T_P0_FID, m1=M1_FID, m2=M2_FID,
                               T_sys=T_SYS, t_obs=T_OBS_PIX,
                               sigma_haslam=SIGMA_HASLAM,
                               sigma_5ghz=SIGMA_5GHZ,
                               include_haslam=True,
                               include_5ghz=True):
    """
    Compute the 3x3 Fisher matrix for spectral parameters [T_p0, m1, m2].

    Data points included:
      - Haslam 408 MHz  (sigma = sigma_haslam = 800 mK)
      - 5 GHz anchor    (sigma = sigma_5ghz, g=1, T_off=0)
      - N satellite channels

    The 5 GHz anchor contributes to all three spectral columns of D,
    providing high-frequency leverage that helps constrain m1 and m2
    through ln(5/1) = 1.609.
    """
    nu_sat    = build_frequency_grid(nu_centre, bandwidth, delta_nu)
    sigma_sat = noise_rms(delta_nu, T_sys, t_obs) * np.ones_like(nu_sat)

    # Build data point arrays
    nu_list    = list(nu_sat)
    sigma_list = list(sigma_sat)

    if include_haslam:
        nu_list    = [NU_HASLAM]    + nu_list
        sigma_list = [sigma_haslam] + sigma_list

    if include_5ghz:
        nu_list    = nu_list    + [NU_5GHZ]
        sigma_list = sigma_list + [sigma_5ghz]

    nu_all    = np.array(nu_list)
    sigma_all = np.array(sigma_list)

    D = np.zeros((len(nu_all), N_PARAMS_3))
    for i, nu_i in enumerate(nu_all):
        # Use g_fid at Haslam only; 5 GHz and satellite use g=1
        g_fid_i = G408_FID if np.isclose(nu_i, NU_HASLAM, rtol=1e-3) else 1.0
        derivs  = compute_derivatives(nu_i, T_p0, m1, m2, g_fid=g_fid_i)
        D[i, 0] = derivs['T_p0']
        D[i, 1] = derivs['m1']
        D[i, 2] = derivs['m2']

    N_inv = np.diag(1.0 / sigma_all**2)
    F     = D.T @ N_inv @ D

    return F, nu_all, sigma_all


# =============================================================================
# DEFAULT compute_fisher_matrix
# =============================================================================

def compute_fisher_matrix(nu_centre, bandwidth, delta_nu, **kwargs):
    """Default: 3x3 spectral block with Haslam + 5 GHz + satellite."""
    return compute_fisher_matrix_3x3(nu_centre, bandwidth, delta_nu, **kwargs)


# =============================================================================
# INVERT AND EXTRACT UNCERTAINTIES
# =============================================================================

def compute_parameter_uncertainties(F):
    """Invert Fisher matrix, return 1-sigma uncertainties and covariance."""
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
# FISHER FORECAST GRID
# =============================================================================

def run_fisher_forecast_grid(nu_centres, bandwidths,
                              delta_nu=25e6,
                              param_idx=0,
                              param_name=r'$\sigma(T_{p,0})$'):
    """
    Sweep over (nu_centre, bandwidth) grid.
    param_idx: 0=T_p0, 1=m1, 2=m2
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
    """Print and plot uncertainties for the 3 spectral parameters."""
    F, nu_all, sigma_all = compute_fisher_matrix(nu_centre, bandwidth, delta_nu)
    sigmas, C            = compute_parameter_uncertainties(F)

    n_sat     = len(nu_all) - 2  # minus Haslam and 5 GHz
    print(f"\n{'='*65}")
    print(f"  3x3 Spectral Fisher Forecast")
    print(f"  beta FIXED={BETA_FIXED}, T_off EXCLUDED, g_408 kept")
    print(f"  nu_c={nu_centre/1e9:.2f} GHz | BW={bandwidth/1e6:.0f} MHz | "
          f"delta_nu={delta_nu/1e6:.0f} MHz")
    print(f"  Data: 1 Haslam (800 mK) + {n_sat} satellite + 1 x 5GHz anchor")
    print(f"  sigma_5GHz = {SIGMA_5GHZ*1e3:.1f} mK")
    print(f"{'='*65}")
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
            r'Spectral uncertainties ($\beta$ fixed, $T_{\rm off}$ excluded, '
            r'5\,GHz anchor added)'
            f'\n$\\nu_c$={nu_centre/1e9:.2f} GHz, BW={bandwidth/1e6:.0f} MHz',
            fontsize=10
        )
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('fisher_3param_single_config.png', dpi=150)
        plt.close()
        print("Saved: fisher_3param_single_config.png")