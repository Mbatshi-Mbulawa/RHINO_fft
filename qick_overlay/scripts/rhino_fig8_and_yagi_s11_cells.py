# ##########################################################################
#  RHINO — two verification cells
#    CELL A : Fig 8 radiometer R-values, computed DIRECTLY from the
#             integration snapshots (no estimates, no simulation).
#    CELL B : Yagi S11 spline-order comparison (k = 3, 4, 5) with residuals,
#             to test the supervisor's "increase the order" suggestion.
#  Paste each cell into rhino_thesis_figures_v1.ipynb (or a scratch notebook).
# ##########################################################################


# ==========================================================================
# CELL A — Fig 8 R-values from data
# ==========================================================================
import numpy as np, glob, os, re
import matplotlib.pyplot as plt

# ---- CONFIG: edit these paths/patterns to match your folders -------------
BASE = '/Users/user/Downloads/Manny-Masters/Project/Data'
DATASETS = {
    # label                : (folder,                 snapshot glob)
    'Load+LNA (3h)'        : (f'{BASE}/load_long_v2',  'integ_snap_*_N*.npy'),
    'Discone+LNA'          : (f'{BASE}/discone_lna_v2','integ_snap_*_N*.npy'),
    'Discone No-LNA'       : (f'{BASE}/discone_nolna_v2','integ_snap_*_N*.npy'),
    'Yagi+CobraX'          : (f'{BASE}/yagi_lna_v2',   'integ_snap_*_N*.npy'),
}
FREQ_FILE   = f'{BASE}/discone_lna_v2/freq_coarse.npy'   # one shared coarse axis
REF_BAND    = (60.0, 75.0)        # MHz — the doc's reference sub-band
# 'adjacent' = std of differences between neighbouring channels (the literal
#   "channel-to-channel scatter", removes smooth foreground); 'raw' = std of
#   the in-band spectrum. Run both; whichever reproduces your existing figure
#   is the metric that figure used.
SCATTER_METHODS = ('adjacent', 'raw')
# --------------------------------------------------------------------------

def to_mhz(freq):
    """Normalise a frequency axis to MHz regardless of stored units.
    Known gotcha: freq_coarse is stored in THz (e.g. 0.000060..0.0022)."""
    f = np.asarray(freq, dtype=float)
    fmax = np.nanmax(np.abs(f))
    if fmax < 1.0:        # THz  (60-85 MHz -> 6e-5..8.5e-5 ; full band -> ~2e-3)
        f = f * 1e6
    elif fmax > 1e6:      # Hz
        f = f / 1e6
    return f              # else: already MHz

def parse_N(path):
    """Extract integration depth N from a filename. Returns int or None."""
    m = re.search(r'_N0*([0-9]+)', os.path.basename(path))
    return int(m.group(1)) if m else None

def in_band_scatter(spec, mask, method):
    s = np.asarray(spec, dtype=float)
    if s.ndim > 1:            # if a snapshot is 2D, collapse to the integrated row
        s = s.mean(axis=0)
    seg = s[mask]
    seg = seg[np.isfinite(seg)]
    if seg.size < 4:
        return np.nan
    if method == 'adjacent':
        return np.std(np.diff(seg)) / np.sqrt(2.0)   # noise proxy, foreground-removed
    return np.std(seg)                                # raw in-band std

# ---- load shared frequency axis & build the reference-band mask ----------
freq_mhz = to_mhz(np.load(FREQ_FILE))
band = (freq_mhz >= REF_BAND[0]) & (freq_mhz <= REF_BAND[1])
print(f'Freq axis: {freq_mhz.min():.2f}–{freq_mhz.max():.2f} MHz, '
      f'{band.sum()} channels in {REF_BAND[0]}–{REF_BAND[1]} MHz')
assert band.sum() > 10, 'Reference band matched too few channels — check units/axis.'

def compute_curve(folder, pattern, method):
    files = glob.glob(os.path.join(folder, pattern))
    pairs = [(parse_N(f), f) for f in files]
    pairs = [(n, f) for n, f in pairs if n is not None]
    pairs.sort(key=lambda t: t[0])          # *** NUMERIC sort by N — the fix ***
    Ns, sig = [], []
    for n, f in pairs:
        spec = np.load(f, allow_pickle=False)
        if spec.shape[-1] != freq_mhz.shape[0]:
            # snapshot length must match the freq axis; skip mismatches loudly
            print(f'  SKIP {os.path.basename(f)}: len {spec.shape[-1]} != {freq_mhz.shape[0]}')
            continue
        s = in_band_scatter(spec, band, method)
        if np.isfinite(s) and s > 0:
            Ns.append(n); sig.append(s)
    return np.array(Ns, float), np.array(sig, float)

def two_point_R(N, sig):
    """Doc Eq. (5): R = sigma_min*sqrt(N_min) / (sigma_max*sqrt(N_max))."""
    i0, i1 = np.argmin(N), np.argmax(N)
    return (sig[i0]*np.sqrt(N[i0])) / (sig[i1]*np.sqrt(N[i1]))

def slope_R(N, sig):
    """Robust cross-check: fit log-sigma vs log-N; ideal slope = -0.5.
    Returns (beta, R_endpoints_from_fit)."""
    b, a = np.polyfit(np.log10(N), np.log10(sig), 1)   # b = slope = -beta
    beta = -b
    # what the two-point R would be if the data followed the fitted slope exactly
    R_fit = (10**(a) * N.min()**b) * np.sqrt(N.min()) / \
            ((10**(a) * N.max()**b) * np.sqrt(N.max()))
    return beta, R_fit

print('\n================  Fig 8 R-values from data  ================')
results = {}
for label, (folder, pat) in DATASETS.items():
    print(f'\n{label}:')
    for method in SCATTER_METHODS:
        N, sig = compute_curve(folder, pat, method)
        if N.size < 2:
            print(f'  [{method:8s}] not enough snapshots found in {folder}')
            continue
        R2 = two_point_R(N, sig)
        beta, Rfit = slope_R(N, sig)
        print(f'  [{method:8s}] n={N.size:3d}  N={int(N.min())}–{int(N.max())}  '
              f'R(2-pt)={R2:.3f}  slope β={beta:.3f} (ideal 0.50)  R(from slope)={Rfit:.3f}')
        results[(label, method)] = (N, sig, R2, beta)

# ---- plot Fig 8 with data-derived R (using the 'adjacent' metric) ---------
plt.figure(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))
for (label, c) in zip(DATASETS, colors):
    key = (label, 'adjacent')
    if key not in results:
        continue
    N, sig, R2, beta = results[key]
    order = np.argsort(N)
    plt.loglog(N[order], sig[order], 'o-', ms=3, color=c, label=f'{label}  R={R2:.2f}')
# ideal 1/sqrt(N) reference, anchored to the first point of the first curve
anchor = next(iter(results.values()))
N0, s0 = anchor[0], anchor[1]
i0 = np.argmin(N0)
Ngrid = np.array([N0.min(), N0.max()], float)
plt.loglog(Ngrid, s0[i0]*np.sqrt(N0[i0])/np.sqrt(Ngrid), 'k--', lw=1,
           label=r'ideal $\sigma\propto N^{-1/2}$')
plt.xlabel('N (averaged spectra)'); plt.ylabel(f'scatter in {REF_BAND[0]}–{REF_BAND[1]} MHz (dB)')
plt.title('Fig 8 — Radiometer compliance (data-derived R)')
plt.legend(fontsize=8); plt.grid(True, which='both', alpha=0.25)
plt.tight_layout()
plt.savefig('fig8_radiometer_equation.png', dpi=150, bbox_inches='tight')
print('\nSaved fig8_radiometer_equation.png')
print('Compare R(2-pt) above against the values printed on your current figure.')


# ==========================================================================
# CELL B — Yagi S11 spline-order comparison
# ==========================================================================
import numpy as np, h5py
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# ---- CONFIG --------------------------------------------------------------
YAGI_S11_FILE = f'{BASE}/yaggi_55_85MHz.hd5f'
FIT_BAND      = (55.0, 85.0)     # fit window (the focused sweep range)
EVAL_BAND     = (60.0, 85.0)     # RHINO science band to evaluate the correction on
ORDERS        = (3, 4, 5)        # k: cubic (current), quartic, quintic
SMOOTH_FACTORS = None            # None -> auto; or e.g. (0.0, 0.5, 2.0, 5.0)
# --------------------------------------------------------------------------

def load_s11_hdf5(path):
    """Open an S11 HDF5 and return (freq_MHz, s11_dB). Tries common layouts;
    prints the file structure so you can adapt if the keys differ."""
    with h5py.File(path, 'r') as h:
        keys = list(h.keys())
        print('HDF5 top-level keys:', keys)
        def find(*names):
            for n in names:
                for k in keys:
                    if n.lower() in k.lower():
                        return np.array(h[k])
            return None
        freq = find('freq', 'frequency', 'hz', 'mhz')
        s_re = find('real', 're')
        s_im = find('imag', 'im')
        s_db = find('db', 'logmag', 'mag_db')
        s_lin= find('mag', 'magnitude', 's11', 'gamma')
        if freq is None:
            raise KeyError(f'No frequency dataset found. Keys present: {keys}')
        if s_re is not None and s_im is not None:
            gamma = s_re + 1j*s_im
            s11_db = 20*np.log10(np.maximum(np.abs(gamma), 1e-12))
        elif s_db is not None:
            s11_db = np.asarray(s_db, float)
        elif s_lin is not None:
            v = np.asarray(s_lin)
            if np.iscomplexobj(v):
                s11_db = 20*np.log10(np.maximum(np.abs(v), 1e-12))
            else:                       # already a magnitude (linear)
                s11_db = 20*np.log10(np.maximum(np.abs(v), 1e-12))
        else:
            raise KeyError(f'Could not identify S11 data. Keys: {keys}')
    # normalise frequency to MHz
    f = np.asarray(freq, float)
    if np.nanmax(f) > 1e6:  f /= 1e6       # Hz -> MHz
    elif np.nanmax(f) < 1:  f *= 1e6       # THz -> MHz
    return f, np.asarray(s11_db, float)

# ---- load, clean, restrict to the fit band -------------------------------
f, s11_db = load_s11_hdf5(YAGI_S11_FILE)
order = np.argsort(f); f, s11_db = f[order], s11_db[order]
f, idx = np.unique(f, return_index=True); s11_db = s11_db[idx]   # strictly increasing
m = (f >= FIT_BAND[0]) & (f <= FIT_BAND[1])
fb, sb = f[m], s11_db[m]
print(f'Yagi S11: {fb.size} points in {FIT_BAND[0]}–{FIT_BAND[1]} MHz, '
      f'|S11| range {sb.min():.2f}..{sb.max():.2f} dB')

# auto smoothing grid if not given: scaled by data variance
if SMOOTH_FACTORS is None:
    base = fb.size * np.var(sb)
    SMOOTH_FACTORS = (0.0, 0.2*base, base, 3*base)   # 0.0 = interpolating

# ---- fit each order, report residuals + a curvature (wiggliness) measure --
eval_grid = np.linspace(EVAL_BAND[0], EVAL_BAND[1], 2000)
print('\n  k   s         RMS resid (dB)   max|2nd deriv| (wiggliness)')
fits = {}
for k in ORDERS:
    for s in SMOOTH_FACTORS:
        try:
            sp = UnivariateSpline(fb, sb, k=k, s=s)
        except Exception as e:
            print(f'  {k}   {s:9.3g}   FAILED: {e}'); continue
        resid = sb - sp(fb)
        rms = np.sqrt(np.mean(resid**2))
        d2 = np.gradient(np.gradient(sp(eval_grid), eval_grid), eval_grid)
        wig = np.max(np.abs(d2))
        fits[(k, s)] = sp
        print(f'  {k}   {s:9.3g}   {rms:12.4f}     {wig:10.3g}')

print("""
Reading this table:
  * Lower RMS is a better fit to the points — BUT s=0 interpolates every
    noisy VNA point, and high k with low s injects oscillation.
  * 'wiggliness' (max |2nd derivative|) flags exactly that. A fit that
    minimises RMS while keeping wiggliness LOW is what you want — a wiggly
    S11 fit puts a wiggly ripple INTO the correction ΔP=-10log10(1-|Γ|²),
    i.e. it manufactures the very spectral structure the experiment is
    trying to avoid. Prefer the lowest order / highest smoothing that still
    follows the real trend.
""")

# ---- plot the candidate fits ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(fb, sb, '.', ms=4, color='0.5', label='measured |S11|')
for k in ORDERS:
    s = SMOOTH_FACTORS[2]   # the mid 'base' smoothing, one curve per order
    if (k, s) in fits:
        ax[0].plot(eval_grid, fits[(k, s)](eval_grid), lw=1.4, label=f'k={k}, s={s:.2g}')
ax[0].set_xlabel('Frequency (MHz)'); ax[0].set_ylabel('|S11| (dB)')
ax[0].set_title('Yagi S11 — spline order comparison'); ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.25)

# resulting correction ΔP for each order (this is what actually hits the spectrum)
for k in ORDERS:
    s = SMOOTH_FACTORS[2]
    if (k, s) in fits:
        gamma = 10**(fits[(k, s)](eval_grid)/20.0)
        eta = 1 - gamma**2
        dP = -10*np.log10(np.clip(eta, 1e-6, None))
        ax[1].plot(eval_grid, dP, lw=1.4, label=f'k={k}')
ax[1].set_xlabel('Frequency (MHz)'); ax[1].set_ylabel(r'correction $\Delta P$ (dB)')
ax[1].set_title('Induced S11 correction (watch for spurious ripple)')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig('yagi_s11_spline_orders.png', dpi=150, bbox_inches='tight')
print('Saved yagi_s11_spline_orders.png')
