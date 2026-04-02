"""
run_cobrax.py
=============
CobraX Fisher forecast — updated per Phil's instructions (April 2026):
  1. Haslam noise = 800 mK
  2. T_offset_408 excluded from parameter vector
  3. 5 GHz anchor data point added
  4. beta fixed at -2.75 (unchanged)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    NU_0, NU_HASLAM, NU_CBASS, NU_LBASS, NU_5GHZ,
    T_P0_FID, BETA_FIXED, M1_FID, M2_FID,
    G408_FID, GCBASS_FID,
    SIGMA_HASLAM, SIGMA_5GHZ, SIGMA_LBASS,
    SIGMA_CBASS_OPTIMISTIC, SIGMA_CBASS_CONSERVATIVE,
    PARAM_LABELS_3, PARAM_NAMES_3,
)
T_OFFSET408_FID = 1.0   # kept for spectrum plot only
BETA_FID = BETA_FIXED

print("=" * 65)
print("  CobraX Fisher Forecast — updated per Phil (April 2026)")
print("  Haslam 800 mK | T_off excluded | C-BASS + L-BASS anchors")
print("=" * 65)

# =============================================================================
# TASK 1: Temperature spectrum
# =============================================================================
print("\n--- TASK 1: Temperature spectrum ---")

nu_arr   = np.logspace(np.log10(0.1e9), np.log10(10e9), 500)
T_true   = temperature_spectrum(nu_arr, T_P0_FID, M1_FID, M2_FID)
T_h_true = temperature_spectrum(NU_HASLAM, T_P0_FID, M1_FID, M2_FID)
T_h_meas = temperature_spectrum(NU_HASLAM, T_P0_FID, M1_FID, M2_FID,
                                 g=G408_FID, T_offset=T_OFFSET408_FID)
T_lbass  = temperature_spectrum(NU_LBASS, T_P0_FID, M1_FID, M2_FID)
T_cbass  = temperature_spectrum(NU_CBASS, T_P0_FID, M1_FID, M2_FID)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nu_arr / 1e9, T_true, color='steelblue', lw=2,
        label=r'True spectrum ($g=1$)')
ax.scatter(NU_HASLAM/1e9, T_h_true, color='steelblue', s=60, zorder=5)
ax.scatter(NU_HASLAM/1e9, T_h_meas, color='firebrick', s=90, marker='*',
           zorder=6, label=fr'Haslam 408 MHz ($g={G408_FID}$, $\sigma=800$\,mK)')
ax.scatter(NU_LBASS/1e9,  T_lbass,  color='seagreen',  s=120, marker='s',
           zorder=6, label=r'L-BASS 1.4 GHz (abs. cal., $\sigma=0.1$\,K)')
ax.scatter(NU_CBASS/1e9,  T_cbass,  color='darkorange', s=120, marker='D',
           zorder=6, label=fr'C-BASS 5 GHz (not abs. cal., $\sigma={SIGMA_5GHZ*1e3:.1f}$\,mK)')
ax.annotate('', xy=(NU_HASLAM/1e9, T_h_meas),
            xytext=(NU_HASLAM/1e9, T_h_true),
            arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.5))
nu_lo = (1.75e9 - 0.5e9/2)/1e9
nu_hi = (1.75e9 + 0.5e9/2)/1e9
ax.axvspan(nu_lo, nu_hi, alpha=0.15, color='goldenrod',
           label=f'CobraX band ({nu_lo:.2f}–{nu_hi:.2f} GHz)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Frequency  [GHz]', fontsize=13)
ax.set_ylabel('Brightness Temperature  [K]', fontsize=13)
ax.set_title('Radio sky brightness temperature spectrum\n'
             r'(Haslam + L-BASS + C-BASS anchors, $\beta=-2.75$ fixed)',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('task1_temperature_spectrum.png', dpi=150)
plt.close()
print("Saved: task1_temperature_spectrum.png")

# =============================================================================
# TASK 2: Derivative table
# =============================================================================
print("\n--- TASK 2: Analytical derivatives at fiducial values ---")
print("beta FIXED | T_off EXCLUDED | g_408 kept | 5 GHz anchor included\n")

test_freqs  = [0.408e9, 1.0e9, 1.5e9, 1.75e9, 2.0e9, 5.0e9]
param_names = ['T_p0', 'm1', 'm2', 'g_408']
print(f"{'nu [GHz]':>10}  " + "  ".join(f"{p:>12}" for p in param_names))
print("-" * 70)
for nu in test_freqs:
    g_fid_i = G408_FID if abs(nu - NU_HASLAM) < 1e7 else 1.0
    d = compute_derivatives(nu, T_P0_FID, M1_FID, M2_FID, g_fid=g_fid_i)
    row = f"{nu/1e9:>10.3f}  " + "  ".join(f"{d[p]:>12.6f}" for p in param_names)
    print(row)
print(f"\nAt 5 GHz: x = ln(5/1) = {np.log(5):.4f} (strong lever arm for m1, m2)")

# =============================================================================
# TASK 3a: Single-configuration uncertainties
# =============================================================================
print("\n--- TASK 3a: Single configuration (1.75 GHz, 400 MHz, 25 MHz ch) ---")
plot_all_uncertainties_single_config(1.75e9, 0.4e9, 25e6)

# =============================================================================
# TASK 3b: 2D Fisher forecast grid
# =============================================================================
print("\n--- TASK 3b: Fisher forecast grid ---")

nu_centres = np.linspace(0.6e9, 2.0e9, 10)
bandwidths = np.linspace(0.1e9, 0.6e9, 8)
delta_nu   = 25e6

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
param_info = [(0, r'$\sigma(T_{p,0})$ [K]'),
              (1, r'$\sigma(m_1)$'),
              (2, r'$\sigma(m_2)$')]

nu_GHz = nu_centres / 1e9; bw_MHz = bandwidths / 1e6
dnu = (nu_GHz[1]-nu_GHz[0])/2; dbw = (bw_MHz[1]-bw_MHz[0])/2
nu_edges = np.concatenate([[nu_GHz[0]-dnu],(nu_GHz[:-1]+nu_GHz[1:])/2,[nu_GHz[-1]+dnu]])
bw_edges = np.concatenate([[bw_MHz[0]-dbw],(bw_MHz[:-1]+bw_MHz[1:])/2,[bw_MHz[-1]+dbw]])

for ax, (pidx, plabel) in zip(axes, param_info):
    sg = run_fisher_forecast_grid(nu_centres, bandwidths, delta_nu,
                                  param_idx=pidx, param_name=plabel)
    valid = sg[np.isfinite(sg) & (sg > 0)]
    if len(valid) == 0:
        ax.set_title(f'{plabel} — all NaN'); continue
    pcm = ax.pcolormesh(nu_edges, bw_edges, sg,
                        norm=LogNorm(vmin=valid.min(), vmax=valid.max()),
                        cmap='viridis_r', shading='flat')
    fig.colorbar(pcm, ax=ax, label=plabel)
    ax.set_xlabel('Band centre [GHz]', fontsize=10)
    ax.set_ylabel('Bandwidth [MHz]', fontsize=10)
    ax.set_title(plabel, fontsize=11)
    ax.grid(True, alpha=0.2)

fig.suptitle(r'Fisher forecast ($\beta$ fixed, $T_{\rm off}$ excl., '
             r'5\,GHz anchor, $\sigma_{408}=800$\,mK)',
             fontsize=11)
plt.tight_layout()
plt.savefig('task3_fisher_3param.png', dpi=150)
plt.close()
print("Saved: task3_fisher_3param.png")

# =============================================================================
# TASK 3c: Bandwidth comparison
# =============================================================================
print("\n--- TASK 3c: Bandwidth comparison ---")

configs = [('Narrow 200 MHz, 25 MHz ch', 200e6, 25e6, '--'),
           ('Wide 400 MHz, 25 MHz ch',   400e6, 25e6, '-'),
           ('Narrow 200 MHz, 10 MHz ch', 200e6, 10e6, ':'),
           ('Wide 400 MHz, 10 MHz ch',   400e6, 10e6, '-.')]
nu_sweep = np.linspace(0.6e9, 2.0e9, 30)

fig, ax = plt.subplots(figsize=(8, 5))
for label, bw, dnu_c, ls in configs:
    vals = []
    for nu_c in nu_sweep:
        try:
            F, _, _ = compute_fisher_matrix_3x3(nu_c, bw, dnu_c)
            s, _ = compute_parameter_uncertainties(F)
            vals.append(s[0])
        except Exception:
            vals.append(np.nan)
    ax.plot(nu_sweep/1e9, vals, ls=ls, lw=2, label=label)

ax.set_xlabel('Band centre frequency  [GHz]', fontsize=13)
ax.set_ylabel(r'$\sigma(T_{p,0})$ [K]', fontsize=13)
ax.set_yscale('log')
ax.set_title(r'$T_{p,0}$ uncertainty vs band placement'
             '\n(5 GHz anchor included, '
             r'$\sigma_{408}=800$\,mK, $T_{\rm off}$ excl.)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('task3c_bandwidth_comparison.png', dpi=150)
plt.close()
print("Saved: task3c_bandwidth_comparison.png")

# =============================================================================
# TASK 3d: Effect of 5 GHz anchor on moment constraints
# This answers Phil's question about preferred frequencies
# =============================================================================
print("\n--- TASK 3d: Effect of 5 GHz anchor on m1 and m2 constraints ---")

nu_sweep2 = np.linspace(0.6e9, 2.0e9, 30)
bw_fixed  = 0.4e9
dnu_fixed = 25e6

fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))

for col, (pidx, plabel) in enumerate([(1, r'$\sigma(m_1)$'),
                                       (2, r'$\sigma(m_2)$')]):
    ax = axes2[col]
    with5, without5 = [], []
    for nu_c in nu_sweep2:
        try:
            Fw, _,  _ = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                                                   include_5ghz=True)
            Fwo, _, _ = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                                                   include_5ghz=False)
            sw,  _ = compute_parameter_uncertainties(Fw)
            swo, _ = compute_parameter_uncertainties(Fwo)
            with5.append(sw[pidx]); without5.append(swo[pidx])
        except Exception:
            with5.append(np.nan); without5.append(np.nan)

    ax.plot(nu_sweep2/1e9, without5, 'steelblue', lw=2, ls='--',
            label='Without 5 GHz anchor')
    ax.plot(nu_sweep2/1e9, with5,    'firebrick', lw=2,
            label='With 5 GHz anchor')
    ax.set_xlabel('Band centre frequency  [GHz]', fontsize=12)
    ax.set_ylabel(plabel, fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'Effect of 5 GHz anchor on {plabel}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    r'5 GHz anchor improves $m_1$, $m_2$ constraints via '
    r'$\ln(5\,\mathrm{GHz}/1\,\mathrm{GHz})=1.61$ lever arm'
    f'\n(BW={bw_fixed/1e6:.0f} MHz, '
    r'$\sigma_{408}=800$\,mK, $T_{\rm off}$ excluded)',
    fontsize=11)
plt.tight_layout()
plt.savefig('task3d_5ghz_anchor_effect.png', dpi=150)
plt.close()
print("Saved: task3d_5ghz_anchor_effect.png")

print("\n--- TASK 3e: C-BASS noise assumption comparison (0.1 mK vs 1 mK) ---")
# Phil specified two possible C-BASS noise values.
# The published C-BASS spec is <0.1 mK/beam. 1 mK is the conservative case.
# This plot shows how much the choice matters for each parameter.

nu_sweep3  = np.linspace(0.6e9, 2.0e9, 30)
bw_fixed   = 0.4e9
dnu_fixed  = 25e6

fig, axes3 = plt.subplots(1, 3, figsize=(14, 5))

for col, (pidx, plabel) in enumerate([(0, r'$\sigma(T_{p,0})$'),
                                       (1, r'$\sigma(m_1)$'),
                                       (2, r'$\sigma(m_2)$')]):
    ax = axes3[col]
    opt_vals, cons_vals, no5_vals = [], [], []

    for nu_c in nu_sweep3:
        try:
            Fo, _, _  = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                            sigma_5ghz=SIGMA_CBASS_OPTIMISTIC,  include_5ghz=True)
            Fc, _, _  = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                            sigma_5ghz=SIGMA_CBASS_CONSERVATIVE, include_5ghz=True)
            Fn, _, _  = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                            include_5ghz=False)
            so, _  = compute_parameter_uncertainties(Fo)
            sc, _  = compute_parameter_uncertainties(Fc)
            sn, _  = compute_parameter_uncertainties(Fn)
            opt_vals.append(so[pidx])
            cons_vals.append(sc[pidx])
            no5_vals.append(sn[pidx])
        except Exception:
            opt_vals.append(np.nan)
            cons_vals.append(np.nan)
            no5_vals.append(np.nan)

    ax.plot(nu_sweep3/1e9, no5_vals,   'gray',       lw=2, ls=':',
            label='No C-BASS')
    ax.plot(nu_sweep3/1e9, cons_vals,  'steelblue',  lw=2, ls='--',
            label='C-BASS 1 mK (conservative)')
    ax.plot(nu_sweep3/1e9, opt_vals,   'firebrick',  lw=2,
            label='C-BASS 0.1 mK (optimistic)')
    ax.set_xlabel('Band centre [GHz]', fontsize=11)
    ax.set_ylabel(plabel, fontsize=11)
    ax.set_yscale('log')
    ax.set_title(plabel, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    r'Effect of C-BASS noise assumption on spectral parameter uncertainties'
    '\n'
    r'(BW=400 MHz, $\sigma_{408}=800$ mK, $T_{\rm off}$ excluded)',
    fontsize=11)
plt.tight_layout()
plt.savefig('task3e_cbass_noise_comparison.png', dpi=150)
plt.close()
print("Saved: task3e_cbass_noise_comparison.png")

print("\n--- TASK 3f: Effect of L-BASS anchor on all three parameters ---")

nu_sweep4  = np.linspace(0.6e9, 2.0e9, 30)
bw_fixed   = 0.4e9
dnu_fixed  = 25e6

fig, axes4 = plt.subplots(1, 3, figsize=(14, 5))

for col, (pidx, plabel) in enumerate([(0, r'$\sigma(T_{p,0})$'),
                                       (1, r'$\sigma(m_1)$'),
                                       (2, r'$\sigma(m_2)$')]):
    ax = axes4[col]
    cbass_only, lbass_only, both_anchors, no_anchors = [], [], [], []

    for nu_c in nu_sweep4:
        try:
            F_no, _, _   = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                               include_5ghz=False, include_lbass=False)
            F_cb, _, _   = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                               include_5ghz=True,  include_lbass=False)
            F_lb, _, _   = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                               include_5ghz=False, include_lbass=True)
            F_both, _, _ = compute_fisher_matrix_3x3(nu_c, bw_fixed, dnu_fixed,
                               include_5ghz=True,  include_lbass=True)
            s_no,   _ = compute_parameter_uncertainties(F_no)
            s_cb,   _ = compute_parameter_uncertainties(F_cb)
            s_lb,   _ = compute_parameter_uncertainties(F_lb)
            s_both, _ = compute_parameter_uncertainties(F_both)
            no_anchors.append(s_no[pidx])
            cbass_only.append(s_cb[pidx])
            lbass_only.append(s_lb[pidx])
            both_anchors.append(s_both[pidx])
        except Exception:
            no_anchors.append(np.nan);   cbass_only.append(np.nan)
            lbass_only.append(np.nan);   both_anchors.append(np.nan)

    ax.plot(nu_sweep4/1e9, no_anchors,   'gray',      lw=1.5, ls=':',
            label='Satellite only')
    ax.plot(nu_sweep4/1e9, cbass_only,   'darkorange', lw=2,  ls='--',
            label='+ C-BASS 5 GHz')
    ax.plot(nu_sweep4/1e9, lbass_only,   'seagreen',   lw=2,  ls='-.',
            label='+ L-BASS 1.4 GHz')
    ax.plot(nu_sweep4/1e9, both_anchors, 'firebrick',  lw=2.5,
            label='+ C-BASS + L-BASS')
    ax.set_xlabel('Band centre [GHz]', fontsize=11)
    ax.set_ylabel(plabel, fontsize=11)
    ax.set_yscale('log')
    ax.set_title(plabel, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(
    'Effect of anchor data points on spectral parameter constraints\n'
    r'(BW=400 MHz, $\sigma_{408}=800$\,mK, $\sigma_{\rm L-BASS}=0.1$\,K, '
    r'$\sigma_{\rm C-BASS}=0.1$\,mK)',
    fontsize=11)
plt.tight_layout()
plt.savefig('task3f_lbass_effect.png', dpi=150)
plt.close()
print("Saved: task3f_lbass_effect.png")

print("\n" + "=" * 65)
print("  All tasks complete.")
print("=" * 65)