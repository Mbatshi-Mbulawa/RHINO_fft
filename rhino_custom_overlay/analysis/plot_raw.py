#!/usr/bin/env python3
"""plot_raw.py FILE.npy  - plot a raw ADC capture from s12_v2_capture.py (131072 int16 samples at 552.96 MS/s).
Assumes the .npy holds the samples in time order (not confirmed from a real capture)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
FS = 552.96e6
x = np.load(sys.argv[1]).astype(float).ravel()
print("samples", x.size, "min", x.min(), "max", x.max(), "mean %.2f" % x.mean(), "std %.2f" % x.std())
x0 = x - x.mean()
seg = 8192; n = x0.size // seg
w = np.hanning(seg)
P = np.mean([np.abs(np.fft.rfft(x0[i*seg:(i+1)*seg] * w))**2 for i in range(n)], axis=0)
f = np.fft.rfftfreq(seg, 1/FS) / 1e6
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(x[:512], lw=0.8); ax[0].set_title("first 512 samples"); ax[0].set_xlabel("sample"); ax[0].set_ylabel("ADC code")
ax[1].semilogy(f, P + 1e-12, lw=0.7); ax[1].set_xlabel("MHz"); ax[1].set_ylabel("power (arb.)")
ax[1].set_title(f"averaged spectrum ({n} segments of {seg}); peak at {f[np.argmax(P[1:])+1]:.3f} MHz"); ax[1].grid(alpha=0.3, which="both")
fig.tight_layout(); out = sys.argv[1][:-4] + ".png"; fig.savefig(out, dpi=120); print("wrote", out)
