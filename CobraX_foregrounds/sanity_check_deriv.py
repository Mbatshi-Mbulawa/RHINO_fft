"""
sanity_check_derivatives.py
============================
Independent numerical validation of all analytical derivatives.

NOTE: beta_p0 is now FIXED at -2.75 (supervisor instruction).
The sanity check still validates it as a check that the model
function itself is correct, even though it is not a free parameter
in the Fisher matrix.

The method is the CENTRAL DIFFERENCE formula:
    dT/dtheta  ~=  [ T(theta + delta) - T(theta - delta) ] / (2 * delta)

Pass criterion: relative error < 1e-5 at all frequencies.

AUTHOR: Mbatshi Jerry Junior Mbulawa
DATE:   27 March 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Import the model and analytical derivatives from our main code
# ============================================================
from cobrax_fisher import (
    temperature_spectrum,
    compute_derivatives,
    NU_0, NU_HASLAM,
    T_P0_FID, BETA_FIXED, M1_FID, M2_FID,
    G408_FID, T_OFFSET408_FID,
)

# Alias: BETA_FID still works anywhere it appears in this script
BETA_FID = BETA_FIXED

# ============================================================
# FIDUCIAL PARAMETER VALUES
# ============================================================
FIDUCIAL_PARAMS = {
    'T_p0'         : T_P0_FID,
    'beta_p0'      : BETA_FID,
    'm1'           : M1_FID,
    'm2'           : M2_FID,
    'g_408'        : G408_FID,
    'T_offset_408' : T_OFFSET408_FID,
}

NU_SAT  = np.linspace(0.5e9, 2.5e9, 200)
NU_ALL  = np.concatenate([[NU_HASLAM], NU_SAT])

DELTA_RELATIVE = 1e-5
PASS_THRESHOLD = 1e-5


# ============================================================
# CORE FUNCTION: numerical derivative via central difference
# ============================================================

def numerical_derivative(nu_arr, param_name, fid_params, delta_rel=DELTA_RELATIVE):
    theta_fid = fid_params[param_name]
    delta     = 1e-5 if np.abs(theta_fid) < 1e-12 else np.abs(theta_fid) * delta_rel

    params_plus  = fid_params.copy()
    params_minus = fid_params.copy()
    params_plus[param_name]  = theta_fid + delta
    params_minus[param_name] = theta_fid - delta

    T_plus  = np.zeros(len(nu_arr))
    T_minus = np.zeros(len(nu_arr))

    for i, nu in enumerate(nu_arr):
        if np.isclose(nu, NU_HASLAM, rtol=1e-3):
            # Haslam point: beta is a positional arg named beta, not beta_p0
            T_plus[i] = temperature_spectrum(
                nu,
                params_plus['T_p0'],
                params_plus['m1'],
                params_plus['m2'],
                g        = params_plus['g_408'],
                T_offset = params_plus['T_offset_408'],
                beta     = params_plus['beta_p0'],
            )
            T_minus[i] = temperature_spectrum(
                nu,
                params_minus['T_p0'],
                params_minus['m1'],
                params_minus['m2'],
                g        = params_minus['g_408'],
                T_offset = params_minus['T_offset_408'],
                beta     = params_minus['beta_p0'],
            )
        else:
            T_plus[i] = temperature_spectrum(
                nu,
                params_plus['T_p0'],
                params_plus['m1'],
                params_plus['m2'],
                g=1.0, T_offset=0.0,
                beta=params_plus['beta_p0'],
            )
            T_minus[i] = temperature_spectrum(
                nu,
                params_minus['T_p0'],
                params_minus['m1'],
                params_minus['m2'],
                g=1.0, T_offset=0.0,
                beta=params_minus['beta_p0'],
            )

    return (T_plus - T_minus) / (2.0 * delta), delta


# ============================================================
# ANALYTICAL DERIVATIVE
# ============================================================

def analytical_derivative(nu_arr, param_name, fid_params):
    deriv_analytical = np.zeros(len(nu_arr))
    for i, nu in enumerate(nu_arr):
        g_fid_i = G408_FID if np.isclose(nu, NU_HASLAM, rtol=1e-3) else 1.0
        d = compute_derivatives(
            nu,
            fid_params['T_p0'],
            fid_params['m1'],
            fid_params['m2'],
            beta         = fid_params['beta_p0'],
            g_fid        = g_fid_i,
            T_offset_fid = fid_params['T_offset_408'],
        )
        # beta_p0 is fixed so it has no entry in d — handle gracefully
        if param_name == 'beta_p0':
            # Compute it manually for the sanity check
            x   = np.log(nu / NU_0)
            P   = (nu / NU_0) ** fid_params['beta_p0']
            M   = 1.0 + fid_params['m1'] * x + fid_params['m2'] * x**2
            g_e = g_fid_i
            deriv_analytical[i] = g_e * fid_params['T_p0'] * P * M * x
        else:
            deriv_analytical[i] = d[param_name]
    return deriv_analytical


# ============================================================
# RELATIVE ERROR
# ============================================================

def relative_error(analytical, numerical):
    floor = 1e-30
    denom = np.maximum(np.abs(numerical), floor)
    return np.abs(analytical - numerical) / denom


# ============================================================
# MAIN SANITY CHECK
# ============================================================

def run_sanity_check():
    param_names = ['T_p0', 'beta_p0', 'm1', 'm2', 'g_408', 'T_offset_408']

    param_labels = {
        'T_p0'         : r'$T_{p,0}$',
        'beta_p0'      : r'$\beta_{p,0}$',
        'm1'           : r'$m_1^{(p)}$',
        'm2'           : r'$m_2^{(p)}$',
        'g_408'        : r'$g_{408}$',
        'T_offset_408' : r'$T_{\mathrm{off},408}$',
    }

    results = {}

    print("=" * 65)
    print("  SANITY CHECK: Analytical vs Numerical Derivatives")
    print(f"  beta is FIXED at {BETA_FIXED} (checked but not a free parameter)")
    print("  Central difference step: delta_rel = {:.0e}".format(DELTA_RELATIVE))
    print("  Pass threshold: rel_error < {:.0e}".format(PASS_THRESHOLD))
    print("=" * 65)

    for pname in param_names:
        d_analytical        = analytical_derivative(NU_ALL, pname, FIDUCIAL_PARAMS)
        d_numerical, delta  = numerical_derivative(NU_ALL, pname, FIDUCIAL_PARAMS)
        rel_err             = relative_error(d_analytical, d_numerical)

        both_zero     = (np.abs(d_analytical) < 1e-20) & (np.abs(d_numerical) < 1e-20)
        rel_err_valid = rel_err[~both_zero]
        max_rel_err   = np.max(rel_err_valid) if len(rel_err_valid) > 0 else 0.0
        passed        = max_rel_err < PASS_THRESHOLD

        results[pname] = {
            'analytical'  : d_analytical,
            'numerical'   : d_numerical,
            'rel_err'     : rel_err,
            'max_rel_err' : max_rel_err,
            'delta'       : delta,
            'passed'      : passed,
        }

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"\n  Parameter: {pname}")
        print(f"    Step size used:     delta = {delta:.2e}")
        print(f"    Max relative error: {max_rel_err:.2e}")
        print(f"    Result:             {status}")

    print("\n" + "=" * 65)
    n_pass = sum(r['passed'] for r in results.values())
    print(f"  {n_pass}/6 derivatives passed the sanity check.")
    print("=" * 65 + "\n")

    # Combined 6x2 figure
    fig, axes = plt.subplots(6, 2, figsize=(13, 22))
    fig.suptitle(
        'Sanity Check: Analytical vs Numerical Derivatives\n'
        r'(Central difference, $\delta_{\rm rel} = 10^{-5}$)',
        fontsize=13, y=0.995
    )
    nu_plot = NU_ALL / 1e9

    for row, pname in enumerate(param_names):
        r   = results[pname]
        lbl = param_labels[pname]
        ax_deriv = axes[row, 0]
        ax_err   = axes[row, 1]

        ax_deriv.plot(nu_plot, r['analytical'], color='steelblue', lw=2.5,
                      label='Analytical', zorder=3)
        ax_deriv.plot(nu_plot, r['numerical'], color='firebrick', lw=1.2,
                      ls='--', label='Numerical (central diff.)', zorder=2)
        idx_haslam = np.argmin(np.abs(NU_ALL - NU_HASLAM))
        ax_deriv.axvline(NU_HASLAM / 1e9, color='gray', lw=0.8, ls=':', alpha=0.7)
        ax_deriv.scatter([NU_HASLAM / 1e9], [r['analytical'][idx_haslam]],
                         color='steelblue', s=40, zorder=5)
        ax_deriv.set_xlabel('Frequency [GHz]', fontsize=9)
        ax_deriv.set_title(f'Parameter: {lbl}', fontsize=10)
        ax_deriv.legend(fontsize=8, loc='best')
        ax_deriv.grid(True, alpha=0.3)

        sc = 'green' if r['passed'] else 'red'
        st = f"PASS\nmax err = {r['max_rel_err']:.1e}" if r['passed'] \
             else f"FAIL\nmax err = {r['max_rel_err']:.1e}"
        ax_deriv.text(0.97, 0.97, st, transform=ax_deriv.transAxes,
                      ha='right', va='top', fontsize=8, color=sc,
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8))

        both_zero = (np.abs(r['analytical']) < 1e-20) & \
                    (np.abs(r['numerical'])   < 1e-20)
        nu_nz  = nu_plot[~both_zero]
        err_nz = r['rel_err'][~both_zero]

        if len(nu_nz) > 0:
            ax_err.semilogy(nu_nz, err_nz, color='darkorange', lw=1.5)
            ax_err.axhline(PASS_THRESHOLD, color='green', lw=1.0, ls='--',
                           label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
            ax_err.set_ylim(bottom=1e-16)
        else:
            ax_err.text(0.5, 0.5, 'All values ≈ 0',
                        ha='center', va='center',
                        transform=ax_err.transAxes, fontsize=9)

        ax_err.set_xlabel('Frequency [GHz]', fontsize=9)
        ax_err.set_ylabel('Relative error', fontsize=9)
        ax_err.set_title(f'Relative error: {lbl}', fontsize=10)
        ax_err.legend(fontsize=8)
        ax_err.grid(True, which='both', alpha=0.3)
        ax_err.axvline(NU_HASLAM / 1e9, color='gray', lw=0.8, ls=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.993])
    plt.savefig('sanity_check_all_params.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sanity_check_all_params.png")

    # Individual figures
    for pname in param_names:
        r   = results[pname]
        lbl = param_labels[pname]
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        nu_plot_local = NU_ALL / 1e9
        idx_haslam    = np.argmin(np.abs(NU_ALL - NU_HASLAM))

        ax1.plot(nu_plot_local, r['analytical'], color='steelblue', lw=2.5,
                 label='Analytical (closed form)', zorder=3)
        ax1.plot(nu_plot_local, r['numerical'], color='firebrick', lw=1.5,
                 ls='--', label=r'Numerical ($\delta_{\rm rel}=10^{-5}$)', zorder=2)
        ax1.axvline(NU_HASLAM / 1e9, color='gray', lw=1.0, ls=':', alpha=0.7,
                    label='Haslam 408 MHz')
        ax1.scatter([NU_HASLAM / 1e9], [r['analytical'][idx_haslam]],
                    color='steelblue', s=50, zorder=5)
        ax1.set_xlabel('Frequency  [GHz]', fontsize=11)
        ax1.set_title(f'Derivative w.r.t. {lbl}', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        both_zero = (np.abs(r['analytical']) < 1e-20) & \
                    (np.abs(r['numerical'])   < 1e-20)
        nu_nz  = nu_plot_local[~both_zero]
        err_nz = r['rel_err'][~both_zero]

        if len(nu_nz) > 0:
            ax2.semilogy(nu_nz, err_nz, color='darkorange', lw=1.8,
                         label='Relative error')
            ax2.axhline(PASS_THRESHOLD, color='green', lw=1.5, ls='--',
                        label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
            ax2.set_ylim(bottom=1e-16, top=1e-1)
        else:
            ax2.text(0.5, 0.5, 'All values ≈ 0\n(parameter only active at 408 MHz)',
                     ha='center', va='center',
                     transform=ax2.transAxes, fontsize=10)

        ax2.axvline(NU_HASLAM / 1e9, color='gray', lw=1.0, ls=':', alpha=0.7)
        ax2.set_xlabel('Frequency  [GHz]', fontsize=11)
        ax2.set_ylabel('Relative error', fontsize=9)
        ax2.set_title(f'Relative error: {lbl}', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, which='both', alpha=0.3)

        status = "PASS" if r['passed'] else "FAIL"
        color  = "green" if r['passed'] else "red"
        fig2.suptitle(
            f'Sanity check for {lbl}  —  '
            f'Max relative error = {r["max_rel_err"]:.2e}  [{status}]',
            fontsize=11, color=color
        )
        plt.tight_layout()
        plt.savefig(f'sanity_{pname}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: sanity_{pname}.png")

    # Step-size robustness check
    print("\nRunning step-size robustness check for beta_p0...")
    deltas     = np.logspace(-10, -1, 30)
    max_errors = []
    for drel in deltas:
        d_a = analytical_derivative(NU_SAT[:20], 'beta_p0', FIDUCIAL_PARAMS)
        d_n, _ = numerical_derivative(NU_SAT[:20], 'beta_p0',
                                      FIDUCIAL_PARAMS, delta_rel=drel)
        max_errors.append(np.max(relative_error(d_a, d_n)))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.loglog(deltas, max_errors, color='steelblue', lw=2,
               marker='o', ms=4,
               label=r'Max rel. error vs step size $\delta$')
    ax3.axhline(PASS_THRESHOLD, color='green', lw=1.5, ls='--',
                label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
    ax3.axvspan(1e-8, 1e-4, alpha=0.08, color='green',
                label='Recommended delta range')
    ax3.set_xlabel(r'Step size $\delta_{\rm rel}$', fontsize=12)
    ax3.set_ylabel('Max relative error', fontsize=12)
    ax3.set_title(r'Robustness check: error vs step size for $\beta_{p,0}$',
                  fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('sanity_step_size_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sanity_step_size_robustness.png")

    return results


if __name__ == '__main__':
    results = run_sanity_check()