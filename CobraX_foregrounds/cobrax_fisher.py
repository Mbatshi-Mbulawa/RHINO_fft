"""
cobrax_fisher.py
================
CobraX Fisher forecast.

CURRENT SPECIFICATION (Phil's instructions, April 2026):
  1. beta_p0 FIXED at -2.75
  2. Haslam noise = 800 mK
  3. T_offset_408 EXCLUDED
  4. C-BASS 5 GHz anchor added (not absolutely calibrated)
  5. L-BASS 1.4 GHz anchor added (absolutely calibrated, sigma = 0.1 K)
  6. C-BASS has good multiplicative calibration, poor offset calibration.
     Include both g_CBASS and T_off_CBASS as free parameters with PRIORS.
  7. F_posterior = F_data + F_prior   (prior is diagonal: 1/sigma_prior^2)
     Prior values:
       g_408:       sigma_prior = 0.2   => F_prior entry = 25
       g_CBASS:     sigma_prior = 0.01  => F_prior entry = 10000
       T_off_CBASS: sigma_prior = TBD   (pending Phil's confirmation)

FULL PARAMETER VECTOR (6 free parameters):
    [g_408, g_CBASS, T_off_CBASS, T_p0, m1, m2]

THE SPECTRAL BLOCK [T_p0, m1, m2] is obtained by marginalising over
the calibration parameters using F_posterior^{-1}.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

NU_0        = 1.0e9           # Pivot frequency [Hz]
NU_HASLAM   = 0.408e9         # Haslam map frequency [Hz]
NU_LBASS    = 1.4e9           # L-BASS frequency [Hz]
NU_CBASS    = 5.0e9           # C-BASS frequency [Hz]
NU_5GHZ     = NU_CBASS        # alias for backward compatibility
BETA_FIXED  = -2.75           # Spectral index — FIXED

# Fiducial parameter values
T_P0_FID        = 2.0         # Brightness temp at nu_0 [K]
M1_FID          = 0.0
M2_FID          = 0.0
G408_FID        = 1.2         # Gain error on Haslam (fiducial)
GCBASS_FID      = 1.0         # Gain error on C-BASS (fiducial = 1, good cal.)
TOFF_CBASS_FID  = 0.0         # Offset on C-BASS [K] (fiducial = 0)

T_SYS       = 150.0           # System temperature [K]
T_OBS_PIX   = 2.2 * 3600.0   # Observing time per pixel [s]

# ---- Noise on data points ----
SIGMA_HASLAM             = 0.8      # [K]  — 800 mK
SIGMA_CBASS_OPTIMISTIC   = 0.1e-3   # [K]  — 0.1 mK
SIGMA_CBASS_CONSERVATIVE = 1.0e-3   # [K]  — 1 mK
SIGMA_5GHZ               = SIGMA_CBASS_OPTIMISTIC
SIGMA_LBASS              = 0.1      # [K]  — 100 mK (Zerafa et al. 2025)

# ---- Prior widths on calibration parameters (Phil's specification) ----
# F_prior is diagonal: entry = 1 / sigma_prior^2
# For parameters with no prior, entry = 0.
SIGMA_PRIOR_G408        = 0.2     # 20% prior on Haslam gain
SIGMA_PRIOR_GCBASS      = 0.01    # 1% prior on C-BASS gain (good mult. cal.)
SIGMA_PRIOR_TOFF_CBASS  = 1.0     # [K] prior on C-BASS offset (pending Phil)
# Set to np.inf (=> 0 in F_prior) to remove a prior entirely.

# ---- Full parameter system (6 parameters) ----
# Ordering: [g_408, g_CBASS, T_off_CBASS, T_p0, m1, m2]
# This ordering puts calibration params first, spectral params last,
# so the 3x3 spectral block is always the bottom-right sub-matrix.
PARAM_NAMES_6  = ['g_408', 'g_CBASS', 'T_off_CBASS', 'T_p0', 'm1', 'm2']
PARAM_LABELS_6 = [
    r'$g_{408}$',
    r'$g_{\rm CBASS}$',
    r'$T_{\rm off,CBASS}$ [K]',
    r'$T_{p,0}$ [K]',
    r'$m_1$',
    r'$m_2$',
]
N_PARAMS_6 = 6

# ---- 3-parameter spectral block (marginalised result) ----
PARAM_NAMES_3  = ['T_p0', 'm1', 'm2']
PARAM_LABELS_3 = [r'$T_{p,0}$ [K]', r'$m_1$', r'$m_2$']
N_PARAMS_3     = 3

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
# ANALYTICAL DERIVATIVES — full 6-parameter system
# Parameters: [g_408, g_CBASS, T_off_CBASS, T_p0, m1, m2]
# beta is FIXED; T_off_408 is EXCLUDED.
# =============================================================================

def compute_derivatives(nu, T_p0, m1, m2,
                        nu_0=NU_0, beta=BETA_FIXED,
                        g_fid=1.0, T_offset_fid=0.0,
                        g_cbass_fid=GCBASS_FID):
    """
    Compute all non-zero analytical partial derivatives of T_p(nu).

    Active at each frequency:
        All frequencies : dT/dT_p0,  dT/dm1,  dT/dm2
        408 MHz only    : dT/dg_408
        5 GHz only      : dT/dg_CBASS,  dT/dT_off_CBASS
        L-BASS 1.4 GHz  : dT/dT_p0, dT/dm1, dT/dm2 only (g=1, no offset)

    Derivations:
    At frequency nu with effective gain g_eff and shorthand
    x = ln(nu/nu_0), P = (nu/nu_0)^beta, M = 1 + m1*x + m2*x^2:

    dT/dT_p0       = g_eff * P * M
    dT/dm1         = g_eff * T_p0 * P * x
    dT/dm2         = g_eff * T_p0 * P * x^2
    dT/dg_408      = T_p0 * P_408 * M_408      [at 408 MHz only]
    dT/dg_CBASS    = T_p0 * P_5 * M_5          [at 5 GHz only]
    dT/dT_off_CBASS = 1                         [at 5 GHz only]
    """
    x = np.log(nu / nu_0)
    P = (nu / nu_0) ** beta
    M = 1.0 + m1 * x + m2 * x**2

    is_haslam = np.isclose(nu, NU_HASLAM, rtol=1e-3)
    is_cbass  = np.isclose(nu, NU_CBASS,  rtol=1e-3)

    # Effective gain for spectral derivatives:
    # g_408 at Haslam, g_cbass at C-BASS, 1.0 everywhere else
    if isinstance(nu, np.ndarray):
        g_eff = np.where(is_haslam, g_fid,
                np.where(is_cbass,  g_cbass_fid, 1.0))
    else:
        if is_haslam:
            g_eff = g_fid
        elif is_cbass:
            g_eff = g_cbass_fid
        else:
            g_eff = 1.0

    # Spectral derivatives — active at all frequencies
    dT_dTp0 = g_eff * P * M
    dT_dm1  = g_eff * T_p0 * P * x
    dT_dm2  = g_eff * T_p0 * P * x**2

    # Calibration derivative: g_408 — only at Haslam
    x_408 = np.log(NU_HASLAM / nu_0)
    P_408 = (NU_HASLAM / nu_0) ** beta
    M_408 = 1.0 + m1 * x_408 + m2 * x_408**2
    dT_dg408_val = T_p0 * P_408 * M_408

    if isinstance(nu, np.ndarray):
        dT_dg408 = np.where(is_haslam, dT_dg408_val, 0.0)
    else:
        dT_dg408 = dT_dg408_val if is_haslam else 0.0

    # Calibration derivatives: g_CBASS and T_off_CBASS — only at C-BASS
    x_5   = np.log(NU_CBASS / nu_0)
    P_5   = (NU_CBASS / nu_0) ** beta
    M_5   = 1.0 + m1 * x_5 + m2 * x_5**2
    dT_dgcbass_val    = T_p0 * P_5 * M_5
    dT_dToffcbass_val = 1.0

    if isinstance(nu, np.ndarray):
        dT_dgcbass    = np.where(is_cbass, dT_dgcbass_val,    0.0)
        dT_dToffcbass = np.where(is_cbass, dT_dToffcbass_val, 0.0)
    else:
        dT_dgcbass    = dT_dgcbass_val    if is_cbass else 0.0
        dT_dToffcbass = dT_dToffcbass_val if is_cbass else 0.0

    return {
        'g_408'        : dT_dg408,
        'g_CBASS'      : dT_dgcbass,
        'T_off_CBASS'  : dT_dToffcbass,
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
# 3x3 SPECTRAL FISHER MATRIX
# Now includes: Haslam (800 mK), 5 GHz anchor, satellite channels
# =============================================================================


# =============================================================================
# PRIOR FISHER MATRIX
# F_prior is diagonal: entry = 1/sigma_prior^2
# Zero entry = no prior on that parameter
# =============================================================================

def build_prior_fisher(sigma_prior_g408=SIGMA_PRIOR_G408,
                       sigma_prior_gcbass=SIGMA_PRIOR_GCBASS,
                       sigma_prior_toff_cbass=SIGMA_PRIOR_TOFF_CBASS):
    """
    Build the 6x6 diagonal prior Fisher matrix.
    Parameter ordering: [g_408, g_CBASS, T_off_CBASS, T_p0, m1, m2]

    For parameters with no prior, set sigma_prior = np.inf => entry = 0.
    """
    F_prior = np.zeros((N_PARAMS_6, N_PARAMS_6))

    def safe_prior(sigma):
        return 0.0 if (sigma is None or np.isinf(sigma)) else 1.0 / sigma**2

    # Calibration parameters (indices 0, 1, 2)
    F_prior[0, 0] = safe_prior(sigma_prior_g408)
    F_prior[1, 1] = safe_prior(sigma_prior_gcbass)
    F_prior[2, 2] = safe_prior(sigma_prior_toff_cbass)
    # Spectral parameters (indices 3, 4, 5) — no priors
    return F_prior


# =============================================================================
# FULL 6x6 DATA FISHER MATRIX
# =============================================================================

def compute_fisher_matrix_full(nu_centre, bandwidth, delta_nu,
                                T_p0=T_P0_FID, m1=M1_FID, m2=M2_FID,
                                T_sys=T_SYS, t_obs=T_OBS_PIX,
                                sigma_haslam=SIGMA_HASLAM,
                                sigma_5ghz=SIGMA_5GHZ,
                                sigma_lbass=SIGMA_LBASS,
                                include_haslam=True,
                                include_cbass=True,
                                include_lbass=True):
    """
    Compute the full 6x6 data Fisher matrix for all free parameters:
        [g_408, g_CBASS, T_off_CBASS, T_p0, m1, m2]

    Data sources:
      - Haslam 408 MHz    (sigma = sigma_haslam)
      - L-BASS 1.4 GHz    (sigma = sigma_lbass, g=1 exactly)
      - C-BASS 5 GHz      (sigma = sigma_5ghz,  g_CBASS free, T_off_CBASS free)
      - N satellite channels
    """
    nu_sat    = build_frequency_grid(nu_centre, bandwidth, delta_nu)
    sigma_sat = noise_rms(delta_nu, T_sys, t_obs) * np.ones_like(nu_sat)

    nu_list    = list(nu_sat)
    sigma_list = list(sigma_sat)

    if include_haslam:
        nu_list    = [NU_HASLAM]    + nu_list
        sigma_list = [sigma_haslam] + sigma_list

    if include_lbass:
        nu_list    = nu_list    + [NU_LBASS]
        sigma_list = sigma_list + [sigma_lbass]

    if include_cbass:
        nu_list    = nu_list    + [NU_CBASS]
        sigma_list = sigma_list + [sigma_5ghz]

    nu_all    = np.array(nu_list)
    sigma_all = np.array(sigma_list)

    D = np.zeros((len(nu_all), N_PARAMS_6))
    for i, nu_i in enumerate(nu_all):
        d = compute_derivatives(nu_i, T_p0, m1, m2,
                                g_fid=G408_FID,
                                g_cbass_fid=GCBASS_FID)
        for j, name in enumerate(PARAM_NAMES_6):
            D[i, j] = d[name]

    N_inv = np.diag(1.0 / sigma_all**2)
    F     = D.T @ N_inv @ D
    return F, nu_all, sigma_all


# =============================================================================
# POSTERIOR FISHER MATRIX = F_data + F_prior
# Then marginalise over calibration parameters by inverting the full matrix
# and extracting the spectral sub-block of the covariance.
# =============================================================================

def compute_fisher_matrix_3x3(nu_centre, bandwidth, delta_nu,
                               T_p0=T_P0_FID, m1=M1_FID, m2=M2_FID,
                               T_sys=T_SYS, t_obs=T_OBS_PIX,
                               sigma_haslam=SIGMA_HASLAM,
                               sigma_5ghz=SIGMA_5GHZ,
                               sigma_lbass=SIGMA_LBASS,
                               include_haslam=True,
                               include_cbass=True,
                               include_lbass=True,
                               include_5ghz=None,   # alias for include_cbass
                               sigma_prior_g408=SIGMA_PRIOR_G408,
                               sigma_prior_gcbass=SIGMA_PRIOR_GCBASS,
                               sigma_prior_toff_cbass=SIGMA_PRIOR_TOFF_CBASS):
    """
    Compute the effective 3x3 Fisher matrix for the spectral parameters
    [T_p0, m1, m2] after marginalising over calibration parameters
    [g_408, g_CBASS, T_off_CBASS] using Gaussian priors.

    Method:
        1. Build 6x6 data Fisher matrix F_data
        2. Build 6x6 diagonal prior Fisher matrix F_prior
        3. F_posterior = F_data + F_prior
        4. Invert F_posterior to get full covariance C_posterior
        5. Extract the 3x3 spectral sub-block: C_spectral = C_posterior[3:,3:]
        6. Invert C_spectral to get the effective spectral Fisher matrix

    This is the correct way to marginalise over nuisance parameters with priors.
    The resulting F_eff is NOT simply the bottom-right 3x3 block of F_posterior —
    it is the Schur complement, automatically handled by full inversion.

    Returns F_eff (3x3), nu_all, sigma_all
    """
    # Handle the include_5ghz alias
    if include_5ghz is not None:
        include_cbass = include_5ghz

    F_data, nu_all, sigma_all = compute_fisher_matrix_full(
        nu_centre, bandwidth, delta_nu,
        T_p0=T_p0, m1=m1, m2=m2,
        T_sys=T_sys, t_obs=t_obs,
        sigma_haslam=sigma_haslam,
        sigma_5ghz=sigma_5ghz,
        sigma_lbass=sigma_lbass,
        include_haslam=include_haslam,
        include_cbass=include_cbass,
        include_lbass=include_lbass,
    )

    F_prior = build_prior_fisher(
        sigma_prior_g408=sigma_prior_g408,
        sigma_prior_gcbass=sigma_prior_gcbass,
        sigma_prior_toff_cbass=sigma_prior_toff_cbass,
    )

    F_posterior = F_data + F_prior

    # Invert full posterior Fisher matrix
    try:
        C_posterior = np.linalg.inv(F_posterior)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan), nu_all, sigma_all

    # Extract spectral sub-block of covariance (bottom-right 3x3)
    # This automatically accounts for marginalisation over calibration params
    C_spectral = C_posterior[3:, 3:]

    # Invert to get effective spectral Fisher matrix
    try:
        F_eff = np.linalg.inv(C_spectral)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.nan), nu_all, sigma_all

    return F_eff, nu_all, sigma_all



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