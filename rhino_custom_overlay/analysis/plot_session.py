#!/usr/bin/env python3
"""plot_session.py  -  plot RHINO spectrometer captures (.npy) from one folder.

usage: python plot_session.py FOLDER --baseline BASE.npy --freq 69.9975 [--out plots] [--zoom 45 95]

File naming (needed so the script can sort files):
  baselines contain 'off' or 'base' in the name  (e.g. b2_ampon_off_125602)
  tone files contain the generator level as _m50_ (= -50 dBm) or _p5_ (= +5 dBm)
Data format (from tone_excess.py): uint64 array, 32770 words: [timestamp, (K<<32)|seq, 32768 power sums]
Counts per frame = power sum / K.  Bin width 16.875 kHz (552.96 MS/s / 32768).  Unique bins 0..16383.
"""
import sys, os, re, glob, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BW = 552.96 / 32768          # MHz per bin
CAL = 21213.575              # counts/frame ~ -29.9 dBm at ADC, 69.9975 MHz (19 Oct calibration)
CAL_DBM = -29.9

def load(path):
    a = np.load(path)
    if a.shape != (32770,):
        raise ValueError(f"{path}: expected 32770 words, got {a.shape}")
    K = int(a[1]) >> 32
    return a[2:].astype(float) / K, K

def level_of(name):
    m = re.search(r"_([mp])(\d+)(?=_|\.|$)", name)
    if not m: return None
    v = int(m.group(2))
    return -v if m.group(1) == "m" else v

def is_base(name):
    n = name.lower()
    return "off" in n or "base" in n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--baseline", required=True)
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--zoom", type=float, nargs=2, default=[45, 95])
    ap.add_argument("--log", default=None, help="captures_log.csv written by cap.sh (adds PC UTC time to the table)")
    ap.add_argument("--exclude", nargs="*", default=[], help="file names (no .npy) to leave out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tlog = {}
    if a.log and os.path.exists(a.log):
        for r in csv.DictReader(open(a.log)): tlog[r["file"][:-4]] = r["pc_utc_after_capture"][11:19]
    base, Kb = load(a.baseline)
    bname = os.path.basename(a.baseline)[:-4]
    f = np.arange(16384) * BW
    bin_t = int(round(a.freq / BW)); bin_m = 32768 - bin_t
    files = sorted(glob.glob(os.path.join(a.folder, "*.npy")), key=os.path.getmtime)
    rows, tones, others = [], [], []
    for p in files:
        n = os.path.basename(p)[:-4]
        if n in a.exclude or n == bname: continue
        try: d, K = load(p)
        except Exception as e: print("skip", n, e); continue
        if is_base(n): others.append((n, d)); continue
        lv = level_of(n)
        if lv is None: print("skip (no level in name):", n); continue
        tones.append((lv, n, d))
    tones.sort(key=lambda t: (t[0], t[1]))
    # table
    print(f"baseline {bname}: tone bin {bin_t} ({bin_t*BW:.4f} MHz), mirror bin {bin_m}")
    hdr = "level_dBm  excess_peak  excess_mirror  pk-mir_dB  step_dB  exp_dB  est_ADC_dBm  pcUTC     file"
    print(hdr); lines = [hdr]; prev = None
    for lv, n, d in tones:
        ep, em = d[bin_t] - base[bin_t], d[bin_m] - base[bin_m]
        pm = 10*np.log10(ep/em) if ep > 0 and em > 0 else float("nan")
        st = 10*np.log10(ep/prev[1]) if prev and ep > 0 and prev[1] > 0 else float("nan")
        est = CAL_DBM + 10*np.log10(ep/CAL) if ep > 0 else float("nan")
        l = f"{lv:9.1f} {ep:12.3f} {em:14.3f} {pm:10.2f} {st:8.2f} {(lv-prev[0]) if prev else float('nan'):7.1f} {est:12.1f}  {tlog.get(n, "-"):8s}  {n}"
        print(l); lines.append(l); prev = (lv, ep)
        rows.append((lv, n, ep, em))
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("note: est_ADC_dBm uses the 70 MHz calibration (about -29.9 dBm = 21213.6 counts/frame); only valid near 70 MHz;"
          " steps are only meaningful if the level was really changed by the nominal amount")

    # fig 1 full band
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(f, np.maximum(base[:16384], 1e-3), lw=0.6, color="k", label="baseline: " + bname)
    for n, d in others[:4]: ax.semilogy(f, np.maximum(d[:16384], 1e-3), lw=0.5, alpha=0.6, label=n)
    for lv, n, d in tones: ax.semilogy(f, np.maximum(d[:16384], 1e-3), lw=0.5, alpha=0.8, label=f"{lv:g} dBm  {n}")
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("counts per frame per bin"); ax.set_xlim(0, 276.48)
    ax.set_title("Full band (bins 0 to 16383)"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "1_fullband.png"), dpi=130); plt.close(fig)

    # fig 2 zoom
    lo, hi = a.zoom; m = (f >= lo) & (f <= hi)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(f[m], np.maximum(base[:16384][m], 1e-3), lw=0.8, color="k", label="baseline")
    for n, d in others[:4]: ax.semilogy(f[m], np.maximum(d[:16384][m], 1e-3), lw=0.7, alpha=0.6, label=n)
    for lv, n, d in tones: ax.semilogy(f[m], np.maximum(d[:16384][m], 1e-3), lw=0.7, alpha=0.8, label=f"{lv:g} dBm")
    ax.axvline(a.freq, color="r", ls=":", lw=0.8)
    ax.set_xlabel("frequency (MHz)"); ax.set_ylabel("counts per frame per bin"); ax.set_title(f"Zoom {lo:g} to {hi:g} MHz (red dotted = generator frequency)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "2_zoom.png"), dpi=130); plt.close(fig)

    # fig 3 excess vs level
    if rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        L = np.array([r[0] for r in rows]); EP = np.array([r[2] for r in rows]); EM = np.array([r[3] for r in rows])
        okp, okm = EP > 0, EM > 0
        ax.semilogy(L[okp], EP[okp], "o", label="peak bin"); ax.semilogy(L[okm], EM[okm], "s", mfc="none", label="mirror bin")
        if okp.sum() >= 1:
            i = np.where(okp)[0][okp.sum() // 2]
            xs = np.linspace(L.min() - 3, L.max() + 3, 20)
            ax.semilogy(xs, EP[i] * 10 ** ((xs - L[i]) / 10), "k--", lw=0.8, label="ideal 10 dB per 10 dB")
        ax.set_xlabel("generator level (dBm, from file name)"); ax.set_ylabel("tone excess (counts per frame)")
        ax.set_title(f"Tone excess vs level, {a.freq:g} MHz"); ax.legend(); ax.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "3_excess_vs_level.png"), dpi=130); plt.close(fig)
    print("wrote", os.path.abspath(a.out))

main()
