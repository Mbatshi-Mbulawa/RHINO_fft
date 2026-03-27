"""
sanity_check_derivatives.py
============================
Independent numerical validation of all 6 analytical derivatives.

WHAT THIS DOES AND WHY
-----------------------
Phil's instruction: do an independent sanity check on the derivatives.
Validate each one by comparing it against a numerical finite-difference
approximation.

The method is the CENTRAL DIFFERENCE formula:

    dT/dtheta  ~=  [ T(theta + delta) - T(theta - delta) ] / (2 * delta)

This is a second-order accurate approximation of the true derivative.
If our ANALYTICAL derivative is correct, it should match this numerical
approximation to many decimal places (typically 6-10 significant figures
for a well-chosen delta).

If they disagree --> there is a mistake in the algebra.
If they agree   --> we have independent confidence the derivative is right.

HOW TO READ THE OUTPUT
-----------------------
For each parameter, this script:
  1. Computes the analytical derivative at every frequency
  2. Computes the numerical (finite difference) derivative at every frequency
  3. Plots both on the same axes -- they should be visually indistinguishable
  4. Computes the RELATIVE ERROR:
         rel_error = |analytical - numerical| / |numerical|
     and plots it. Good agreement means rel_error < 1e-6 everywhere.
  5. Prints a PASS/FAIL summary to the terminal.

The relative error plot is the key diagnostic. If it is flat and tiny
(~1e-7 to 1e-10), your derivative is correct. If it is large or varies
wildly with frequency, there is a mistake.

AUTHOR: Mbatshi Jerry Junior Mbulawa
DATE:  27 March 2026
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
    T_P0_FID, BETA_FID, M1_FID, M2_FID,
    G408_FID, T_OFFSET408_FID,
)

# ============================================================
# FIDUCIAL PARAMETER VALUES
# We evaluate all derivatives at these values.
# ============================================================
FIDUCIAL_PARAMS = {
    'T_p0'         : T_P0_FID,
    'beta_p0'      : BETA_FID,
    'm1'           : M1_FID,
    'm2'           : M2_FID,
    'g_408'        : G408_FID,       # gain at Haslam -- used in model call
    'T_offset_408' : T_OFFSET408_FID # offset at Haslam -- used in model call
}

# ============================================================
# FREQUENCY GRID FOR THE SANITY CHECK
# We test across a wide range of frequencies so we can see
# the derivatives behave correctly everywhere in the band.
# Include the Haslam point explicitly.
# ============================================================
NU_SAT  = np.linspace(0.5e9, 2.5e9, 200)   # 200 points across satellite range
NU_ALL  = np.concatenate([[NU_HASLAM], NU_SAT])  # Haslam + satellite

# ============================================================
# CENTRAL DIFFERENCE STEP SIZE
# This needs careful thought:
#   - Too large  --> approximation error (we're measuring curvature not slope)
#   - Too small  --> floating point cancellation error
# delta ~ 1e-4 to 1e-5 relative to the parameter value is usually optimal.
# We will test multiple values to confirm robustness.
# ============================================================
DELTA_RELATIVE = 1e-5   # 0.001% of each parameter value


# ============================================================
# CORE FUNCTION: numerical derivative via central difference
# ============================================================

def numerical_derivative(nu_arr, param_name, fid_params, delta_rel=DELTA_RELATIVE):
    """
    Compute the numerical (central difference) derivative of T_p(nu)
    with respect to the parameter named `param_name`, evaluated at
    all frequencies in nu_arr.

    Uses the central difference formula:
        dT/dtheta ~ [ T(theta+delta) - T(theta-delta) ] / (2*delta)

    Parameters
    ----------
    nu_arr     : np.ndarray  -- frequencies at which to evaluate [Hz]
    param_name : str         -- one of the 6 parameter names
    fid_params : dict        -- fiducial parameter values
    delta_rel  : float       -- step size as fraction of parameter value

    Returns
    -------
    deriv_num : np.ndarray  -- numerical derivative at each frequency
    delta     : float       -- actual step size used
    """
    # Get the fiducial value of the parameter we're differentiating
    theta_fid = fid_params[param_name]

    # Step size: if parameter is zero (e.g. m1=0), use an absolute step
    if np.abs(theta_fid) < 1e-12:
        delta = 1e-5    # absolute step for zero-valued parameters
    else:
        delta = np.abs(theta_fid) * delta_rel

    # --- Build T(theta + delta) ---
    params_plus = fid_params.copy()
    params_plus[param_name] = theta_fid + delta

    # --- Build T(theta - delta) ---
    params_minus = fid_params.copy()
    params_minus[param_name] = theta_fid - delta

    # --- Evaluate the model at each frequency ---
    # We need to handle g and T_offset carefully:
    # at satellite frequencies g=1, T_offset=0 regardless of their
    # "free" values -- only the Haslam point uses the free values.
    T_plus  = np.zeros(len(nu_arr))
    T_minus = np.zeros(len(nu_arr))

    for i, nu in enumerate(nu_arr):
        if np.isclose(nu, NU_HASLAM, rtol=1e-3):
            # Haslam point: use free g and T_offset
            T_plus[i] = temperature_spectrum(
                nu,
                params_plus['T_p0'],
                params_plus['beta_p0'],
                params_plus['m1'],
                params_plus['m2'],
                g        = params_plus['g_408'],
                T_offset = params_plus['T_offset_408'],
            )
            T_minus[i] = temperature_spectrum(
                nu,
                params_minus['T_p0'],
                params_minus['beta_p0'],
                params_minus['m1'],
                params_minus['m2'],
                g        = params_minus['g_408'],
                T_offset = params_minus['T_offset_408'],
            )
        else:
            # Satellite channels: g=1, T_offset=0 always
            T_plus[i] = temperature_spectrum(
                nu,
                params_plus['T_p0'],
                params_plus['beta_p0'],
                params_plus['m1'],
                params_plus['m2'],
                g=1.0, T_offset=0.0,
            )
            T_minus[i] = temperature_spectrum(
                nu,
                params_minus['T_p0'],
                params_minus['beta_p0'],
                params_minus['m1'],
                params_minus['m2'],
                g=1.0, T_offset=0.0,
            )

    # Central difference
    deriv_num = (T_plus - T_minus) / (2.0 * delta)

    return deriv_num, delta


# ============================================================
# ANALYTICAL DERIVATIVE (from our main code)
# ============================================================

def analytical_derivative(nu_arr, param_name, fid_params):
    """
    Compute the analytical derivative at each frequency in nu_arr.
    Calls compute_derivatives() from cobrax_fisher.py.
    """
    deriv_analytical = np.zeros(len(nu_arr))
    for i, nu in enumerate(nu_arr):
        d = compute_derivatives(
            nu,
            fid_params['T_p0'],
            fid_params['beta_p0'],
            fid_params['m1'],
            fid_params['m2'],
            g_fid       = fid_params['g_408'],
            T_offset_fid= fid_params['T_offset_408'],
        )
        deriv_analytical[i] = d[param_name]
    return deriv_analytical


# ============================================================
# RELATIVE ERROR
# ============================================================

def relative_error(analytical, numerical):
    """
    Compute relative error: |analytical - numerical| / max(|numerical|, floor)
    The floor prevents division by zero when numerical ~ 0
    (e.g. at nu = nu_0 where dT/dbeta = 0 exactly).
    """
    floor = 1e-30
    denom = np.maximum(np.abs(numerical), floor)
    return np.abs(analytical - numerical) / denom


# ============================================================
# PASS/FAIL THRESHOLD
# We expect agreement to at least 5 significant figures.
# Relative error < 1e-5 is a pass.
# ============================================================
PASS_THRESHOLD = 1e-5


# ============================================================
# MAIN SANITY CHECK: run for all 6 parameters and make plots
# ============================================================

def run_sanity_check():
    """
    Run the full sanity check for all 6 parameters.
    Produces:
      - One 2-panel figure per parameter (derivative comparison + error)
      - A combined 6x2 summary figure
      - Terminal output with PASS/FAIL for each parameter
    """

    param_names = ['T_p0', 'beta_p0', 'm1', 'm2', 'g_408', 'T_offset_408']

    param_labels = {
        'T_p0'         : r'$T_{p,0}$',
        'beta_p0'      : r'$\beta_{p,0}$',
        'm1'           : r'$m_1^{(p)}$',
        'm2'           : r'$m_2^{(p)}$',
        'g_408'        : r'$g_{408}$',
        'T_offset_408' : r'$T_{\mathrm{off},408}$',
    }

    # Storage for results
    results = {}

    print("=" * 65)
    print("  SANITY CHECK: Analytical vs Numerical Derivatives")
    print("  Central difference step: delta_rel = {:.0e}".format(DELTA_RELATIVE))
    print("  Pass threshold: rel_error < {:.0e}".format(PASS_THRESHOLD))
    print("=" * 65)

    for pname in param_names:

        # --- Compute both derivatives ---
        d_analytical        = analytical_derivative(NU_ALL, pname, FIDUCIAL_PARAMS)
        d_numerical, delta  = numerical_derivative(NU_ALL, pname, FIDUCIAL_PARAMS)
        rel_err             = relative_error(d_analytical, d_numerical)

        # --- Assess pass/fail ---
        # Exclude points where both derivatives are essentially zero
        # (e.g. g_408 and T_off at satellite frequencies) to avoid
        # 0/0 in the relative error causing spurious failures.
        both_zero = (np.abs(d_analytical) < 1e-20) & (np.abs(d_numerical) < 1e-20)
        rel_err_valid = rel_err[~both_zero]

        if len(rel_err_valid) == 0:
            max_rel_err = 0.0
        else:
            max_rel_err = np.max(rel_err_valid)

        passed = max_rel_err < PASS_THRESHOLD

        results[pname] = {
            'analytical'  : d_analytical,
            'numerical'   : d_numerical,
            'rel_err'     : rel_err,
            'max_rel_err' : max_rel_err,
            'delta'       : delta,
            'passed'      : passed,
        }

        # --- Terminal output ---
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"\n  Parameter: {pname}")
        print(f"    Step size used:     delta = {delta:.2e}")
        print(f"    Max relative error: {max_rel_err:.2e}")
        print(f"    Result:             {status}")

    print("\n" + "=" * 65)
    n_pass = sum(r['passed'] for r in results.values())
    print(f"  {n_pass}/6 derivatives passed the sanity check.")
    print("=" * 65 + "\n")

    # --------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------
    # Figure 1: Combined 6x2 panel figure
    # Left column: analytical vs numerical
    # Right column: relative error (log scale)
    # --------------------------------------------------------

    fig, axes = plt.subplots(6, 2, figsize=(13, 22))
    fig.suptitle(
        'Sanity Check: Analytical vs Numerical Derivatives\n'
        r'(Central difference, $\delta_{\rm rel} = 10^{-5}$)',
        fontsize=13, y=0.995
    )

    nu_plot = NU_ALL / 1e9  # convert to GHz for x-axis

    for row, pname in enumerate(param_names):
        r   = results[pname]
        lbl = param_labels[pname]

        ax_deriv = axes[row, 0]
        ax_err   = axes[row, 1]

        # --- Left panel: overlay of both derivatives ---
        ax_deriv.plot(nu_plot, r['analytical'],
                      color='steelblue', lw=2.5,
                      label='Analytical', zorder=3)
        ax_deriv.plot(nu_plot, r['numerical'],
                      color='firebrick', lw=1.2, ls='--',
                      label='Numerical (central diff.)', zorder=2)

        # Mark the Haslam point
        idx_haslam = np.argmin(np.abs(NU_ALL - NU_HASLAM))
        ax_deriv.axvline(NU_HASLAM / 1e9, color='gray',
                         lw=0.8, ls=':', alpha=0.7)
        ax_deriv.scatter([NU_HASLAM / 1e9], [r['analytical'][idx_haslam]],
                         color='steelblue', s=40, zorder=5)

        ax_deriv.set_xlabel('Frequency [GHz]', fontsize=9)
        ax_deriv.set_ylabel(r'$\partial T_p/\partial\,' + lbl[1:-1] + '$',
                             fontsize=9)
        ax_deriv.set_title(f'Parameter: {lbl}', fontsize=10)
        ax_deriv.legend(fontsize=8, loc='best')
        ax_deriv.grid(True, alpha=0.3)

        # Add pass/fail text on the plot
        status_color = 'green' if r['passed'] else 'red'
        status_text  = f"PASS\nmax err = {r['max_rel_err']:.1e}" \
                       if r['passed'] \
                       else f"FAIL\nmax err = {r['max_rel_err']:.1e}"
        ax_deriv.text(0.97, 0.97, status_text,
                      transform=ax_deriv.transAxes,
                      ha='right', va='top', fontsize=8,
                      color=status_color, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8))

        # --- Right panel: relative error on log scale ---
        # Only plot where at least one of the derivatives is non-negligible
        both_zero = (np.abs(r['analytical']) < 1e-20) & \
                    (np.abs(r['numerical'])   < 1e-20)
        nu_nz  = nu_plot[~both_zero]
        err_nz = r['rel_err'][~both_zero]

        if len(nu_nz) > 0:
            ax_err.semilogy(nu_nz, err_nz,
                            color='darkorange', lw=1.5)
            ax_err.axhline(PASS_THRESHOLD, color='green',
                           lw=1.0, ls='--',
                           label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
            ax_err.set_ylim(bottom=1e-16)
        else:
            ax_err.text(0.5, 0.5, 'All values ≈ 0\n(no non-trivial points)',
                        ha='center', va='center',
                        transform=ax_err.transAxes, fontsize=9)

        ax_err.set_xlabel('Frequency [GHz]', fontsize=9)
        ax_err.set_ylabel('Relative error', fontsize=9)
        ax_err.set_title(f'Relative error: {lbl}', fontsize=10)
        ax_err.legend(fontsize=8)
        ax_err.grid(True, which='both', alpha=0.3)
        ax_err.axvline(NU_HASLAM / 1e9, color='gray',
                       lw=0.8, ls=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.993])
    plt.savefig('sanity_check_all_params.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sanity_check_all_params.png")

    # --------------------------------------------------------
    # Figure 2: Individual clean figures for each parameter
    # These are the presentation-quality ones
    # --------------------------------------------------------
    for pname in param_names:
        r   = results[pname]
        lbl = param_labels[pname]

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        nu_plot_local = NU_ALL / 1e9
        idx_haslam    = np.argmin(np.abs(NU_ALL - NU_HASLAM))

        # Left: derivative comparison
        ax1.plot(nu_plot_local, r['analytical'],
                 color='steelblue', lw=2.5,
                 label='Analytical (closed form)', zorder=3)
        ax1.plot(nu_plot_local, r['numerical'],
                 color='firebrick', lw=1.5, ls='--',
                 label=r'Numerical ($\delta_{\rm rel}=10^{-5}$)', zorder=2)
        ax1.axvline(NU_HASLAM / 1e9, color='gray',
                    lw=1.0, ls=':', alpha=0.7, label='Haslam 408 MHz')
        ax1.scatter([NU_HASLAM / 1e9], [r['analytical'][idx_haslam]],
                    color='steelblue', s=50, zorder=5)
        ax1.set_xlabel('Frequency  [GHz]', fontsize=11)
        ax1.set_ylabel(r'$\partial T_p\,/\,\partial\,' + lbl[1:-1] + '$',
                       fontsize=11)
        ax1.set_title(f'Derivative w.r.t. {lbl}', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right: relative error
        both_zero = (np.abs(r['analytical']) < 1e-20) & \
                    (np.abs(r['numerical'])   < 1e-20)
        nu_nz  = nu_plot_local[~both_zero]
        err_nz = r['rel_err'][~both_zero]

        if len(nu_nz) > 0:
            ax2.semilogy(nu_nz, err_nz,
                         color='darkorange', lw=1.8,
                         label='Relative error')
            ax2.axhline(PASS_THRESHOLD, color='green',
                        lw=1.5, ls='--',
                        label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
            ax2.set_ylim(bottom=1e-16, top=1e-1)
        else:
            ax2.text(0.5, 0.5,
                     'All values ≈ 0\n(parameter only active at 408 MHz)',
                     ha='center', va='center',
                     transform=ax2.transAxes, fontsize=10)

        ax2.axvline(NU_HASLAM / 1e9, color='gray',
                    lw=1.0, ls=':', alpha=0.7)
        ax2.set_xlabel('Frequency  [GHz]', fontsize=11)
        ax2.set_ylabel('Relative error  |analytical - numerical| / |numerical|',
                       fontsize=9)
        ax2.set_title(f'Relative error: {lbl}', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, which='both', alpha=0.3)

        status = "PASS" if r['passed'] else "FAIL"
        color  = "green" if r['passed'] else "red"
        fig2.suptitle(
            f'Sanity check for {lbl}  —  '
            f'Max relative error = {r["max_rel_err"]:.2e}  '
            f'[{status}]',
            fontsize=11, color=color
        )
        plt.tight_layout()
        fname = f'sanity_{pname}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    # --------------------------------------------------------
    # Figure 3: Step-size robustness check for beta
    # This shows that the agreement holds across a wide range
    # of delta values -- confirming it's genuine agreement,
    # not coincidental cancellation at one specific delta.
    # --------------------------------------------------------
    print("\nRunning step-size robustness check for beta_p0...")

    deltas      = np.logspace(-10, -1, 30)
    max_errors  = []

    for drel in deltas:
        d_a = analytical_derivative(NU_SAT[:20], 'beta_p0', FIDUCIAL_PARAMS)
        d_n, _ = numerical_derivative(NU_SAT[:20], 'beta_p0',
                                       FIDUCIAL_PARAMS, delta_rel=drel)
        err = relative_error(d_a, d_n)
        max_errors.append(np.max(err))

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.loglog(deltas, max_errors,
               color='steelblue', lw=2, marker='o', ms=4,
               label=r'Max rel. error vs step size $\delta$')
    ax3.axhline(PASS_THRESHOLD, color='green', lw=1.5, ls='--',
                label=f'Pass threshold ({PASS_THRESHOLD:.0e})')
    ax3.axvspan(1e-8, 1e-4, alpha=0.08, color='green',
                label='Recommended delta range')
    ax3.set_xlabel(r'Step size $\delta_{\rm rel}$', fontsize=12)
    ax3.set_ylabel('Max relative error', fontsize=12)
    ax3.set_title(r'Robustness check: error vs step size for $\beta_{p,0}$'
                  '\n(Sweet spot between approximation error and cancellation error)',
                  fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.annotate('Too large:\napproximation\nerror dominates',
                 xy=(1e-1, max_errors[-1]),
                 xytext=(3e-3, 1e-2),
                 fontsize=8, color='firebrick',
                 arrowprops=dict(arrowstyle='->', color='firebrick'))
    ax3.annotate('Too small:\nfloating point\ncancellation',
                 xy=(1e-10, max_errors[0]),
                 xytext=(1e-9, 1e-4),
                 fontsize=8, color='firebrick',
                 arrowprops=dict(arrowstyle='->', color='firebrick'))

    plt.tight_layout()
    plt.savefig('sanity_step_size_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sanity_step_size_robustness.png")

    return results


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    results = run_sanity_check()