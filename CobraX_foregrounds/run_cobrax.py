"""
run_cobrax.py
=============
Run all three tasks for the CobraX Fisher forecast analysis.
Execute this file directly:  python run_cobrax.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for script mode
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from cobrax_fisher import (
    temperature_spectrum,
    compute_derivatives,
    compute_fisher_matrix,
    compute_fisher_matrix_3x3,
    compute_parameter_uncertainties,
    plot_all_uncertainties_single_config,
    run_fisher_forecast_grid,
    NU_0, NU_HASLAM,
    T_P0_FID, BETA_FIXED, M1_FID, M2_FID,
    G408_FID, T_OFFSET408_FID,
    PARAM_LABELS_3, PARAM_NAMES_3,
    SIGMA_HASLAM,
)

# Alias so any remaining references to BETA_FID still work
BETA_FID = BETA_FIXED

print("=" * 65)
print("  CobraX Fisher Forecast — Nasirudin & Bull (2026)")
print("  Running all tasks...")
print("=" * 65)

# =============================================================================
# TASK 1: Plot the temperature spectrum
# =============================================================================
print("\n--- TASK 1: Temperature spectrum ---")

nu_arr = np.logspace(np.log10(0.1e9), np.log10(10e9), 500)
T_true = temperature_spectrum(nu_arr, T_P0_FID, M1_FID, M2_FID)

T_haslam_true = temperature_spectrum(NU_HASLAM, T_P0_FID, M1_FID, M2_FID)
T_haslam_meas = temperature_spectrum(NU_HASLAM, T_P0_FID, M1_FID, M2_FID,
                                     g=G408_FID, T_offset=T_OFFSET408_FID)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nu_arr / 1e9, T_true, color='steelblue', lw=2,
        label=r'True spectrum ($g=1$, $T_{\rm off}=0$)')
ax.scatter(NU_HASLAM / 1e9, T_haslam_true, color='steelblue', s=60, zorder=5)
ax.scatter(NU_HASLAM / 1e9, T_haslam_meas, color='firebrick',
           s=80, marker='*', zorder=6,
           label=fr'Haslam measurement ($g={G408_FID}$, '
                 fr'$T_{{\rm off}}={T_OFFSET408_FID}$ K)')
ax.annotate('', xy=(NU_HASLAM / 1e9, T_haslam_meas),
            xytext=(NU_HASLAM / 1e9, T_haslam_true),
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.5))

nu_lo = (1.75e9 - 0.5e9 / 2) / 1e9
nu_hi = (1.75e9 + 0.5e9 / 2) / 1e9
ax.axvspan(nu_lo, nu_hi, alpha=0.15, color='goldenrod',
           label=f'CobraX band ({nu_lo:.2f}–{nu_hi:.2f} GHz)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency  [GHz]', fontsize=13)
ax.set_ylabel('Brightness Temperature  [K]', fontsize=13)
ax.set_title('Radio sky brightness temperature spectrum\n'
             r'(fiducial: $T_{p,0}=2$ K, $\beta=-2.75$ fixed, $\nu_0=1$ GHz)',
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('task1_temperature_spectrum.png', dpi=150)
plt.close()
print("Saved: task1_temperature_spectrum.png")

# =============================================================================
# TASK 2: Print analytical derivatives for inspection
# =============================================================================
print("\n--- TASK 2: Analytical derivatives at fiducial values ---")
print("Evaluating at a few representative frequencies:\n")
print("NOTE: beta is now FIXED at -2.75 — dT/d(beta) is not a free parameter.\n")

test_freqs  = [0.408e9, 1.0e9, 1.5e9, 1.75e9, 2.0e9]
param_names = ['T_p0', 'm1', 'm2', 'g_408', 'T_offset_408']

print(f"{'nu [GHz]':>10}  " + "  ".join(f"{p:>16}" for p in param_names))
print("-" * 100)
for nu in test_freqs:
    g_fid_i = G408_FID if abs(nu - NU_HASLAM) < 1e7 else 1.0
    d = compute_derivatives(nu, T_P0_FID, M1_FID, M2_FID, g_fid=g_fid_i)
    row = f"{nu/1e9:>10.3f}  " + "  ".join(f"{d[p]:>16.6f}" for p in param_names)
    print(row)

print("\nNote: beta is fixed so dT/d(beta) is not computed.")
print("The beta-m1 degeneracy is resolved. Fisher matrix should be invertible.\n")

# =============================================================================
# TASK 3a: Single-configuration 3x3 Fisher matrix
# =============================================================================
print("\n--- TASK 3a: Single configuration (1.75 GHz centre, 400 MHz BW, 25 MHz channels) ---")
plot_all_uncertainties_single_config(
    nu_centre=1.75e9,
    bandwidth=0.4e9,
    delta_nu=25e6,
)

# =============================================================================
# TASK 3b: Fisher forecast grid (3x3 spectral system)
# =============================================================================
print("\n--- TASK 3b: Fisher forecast grid (3x3 spectral system) ---")

nu_centres = np.linspace(0.6e9, 2.0e9, 10)
bandwidths = np.linspace(0.1e9, 0.6e9, 8)
delta_nu   = 25e6

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

param_info = [
    (0, r'$\sigma(T_{p,0})$ [K]'),
    (1, r'$\sigma(m_1)$'),
    (2, r'$\sigma(m_2)$'),
]

nu_GHz   = nu_centres / 1e9
bw_MHz   = bandwidths / 1e6
dnu      = (nu_GHz[1] - nu_GHz[0]) / 2
dbw      = (bw_MHz[1] - bw_MHz[0]) / 2
nu_edges = np.concatenate([[nu_GHz[0] - dnu],
                            (nu_GHz[:-1] + nu_GHz[1:]) / 2,
                            [nu_GHz[-1] + dnu]])
bw_edges = np.concatenate([[bw_MHz[0] - dbw],
                            (bw_MHz[:-1] + bw_MHz[1:]) / 2,
                            [bw_MHz[-1] + dbw]])

for ax, (pidx, plabel) in zip(axes, param_info):
    sigma_grid = run_fisher_forecast_grid(
        nu_centres, bandwidths, delta_nu,
        param_idx=pidx, param_name=plabel
    )
    valid = sigma_grid[np.isfinite(sigma_grid) & (sigma_grid > 0)]
    if len(valid) == 0:
        ax.set_title(f'{plabel} — all NaN')
        continue
    pcm = ax.pcolormesh(nu_edges, bw_edges, sigma_grid,
                        norm=LogNorm(vmin=valid.min(), vmax=valid.max()),
                        cmap='viridis_r', shading='flat')
    fig.colorbar(pcm, ax=ax, label=plabel)
    ax.set_xlabel('Band centre [GHz]', fontsize=10)
    ax.set_ylabel('Bandwidth [MHz]', fontsize=10)
    ax.set_title(plabel, fontsize=11)
    ax.grid(True, alpha=0.2)

fig.suptitle(
    r'Fisher forecast: $\sigma$ on spectral parameters ($\beta$ fixed at $-2.75$)',
    fontsize=12
)
plt.tight_layout()
plt.savefig('task3_fisher_3param.png', dpi=150)
plt.close()
print("Saved: task3_fisher_3param.png")

# =============================================================================
# TASK 3c: Bandwidth comparison — sigma(T_p0) vs band centre
# =============================================================================
print("\n--- TASK 3c: Bandwidth comparison ---")

configs = [
    ('Narrow 200 MHz, 25 MHz ch', 200e6, 25e6,  '--'),
    ('Wide 400 MHz, 25 MHz ch',   400e6, 25e6,  '-'),
    ('Narrow 200 MHz, 10 MHz ch', 200e6, 10e6,  ':'),
    ('Wide 400 MHz, 10 MHz ch',   400e6, 10e6,  '-.'),
]

nu_sweep = np.linspace(0.6e9, 2.0e9, 30)

fig, ax = plt.subplots(figsize=(8, 5))

for label, bw, dnu_c, ls in configs:
    sigma_Tp0 = []
    for nu_c in nu_sweep:
        try:
            F, _, _ = compute_fisher_matrix_3x3(nu_c, bw, dnu_c)
            sigmas, _ = compute_parameter_uncertainties(F)
            sigma_Tp0.append(sigmas[0])   # index 0 = T_p0
        except Exception:
            sigma_Tp0.append(np.nan)
    ax.plot(nu_sweep / 1e9, sigma_Tp0, ls=ls, lw=2, label=label)

ax.set_xlabel('Band centre frequency  [GHz]', fontsize=13)
ax.set_ylabel(r'$\sigma(T_{p,0})$ [K]', fontsize=13)
ax.set_yscale('log')
ax.set_title(r'$T_{p,0}$ uncertainty vs band placement', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('task3c_bandwidth_comparison.png', dpi=150)
plt.close()
print("Saved: task3c_bandwidth_comparison.png")

print("\n" + "=" * 65)
print("  All tasks complete.")
print("=" * 65)