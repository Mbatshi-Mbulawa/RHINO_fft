import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FS = 552.96e6
N = 32768
d = np.load(sys.argv[1])
out = sys.argv[2]
assert d.shape == (N + 2,), 'bad shape %s' % (d.shape,)
ts = int(d[0])
k = int(d[1]) >> 32
sq = int(d[1]) & 0xFFFFFFFF
half = d[2:2 + N // 2].astype(np.float64)
f = np.arange(N // 2) * FS / N / 1e6
pb = int(np.argmax(half))
med = float(np.median(half))
print('timestamp', ts, 'K field', k, 'seq', sq)
print('peak bin', pb, 'freq %.4f MHz' % f[pb], 'power', int(half[pb]))
print('median power', med, 'peak/median %.1f dB' % (10 * np.log10(half[pb] / max(med, 1))))
print('zero bins in first half:', int(np.count_nonzero(half == 0)))
plt.figure(figsize=(9, 4))
plt.plot(f, 10 * np.log10(np.maximum(half, 1)), lw=0.6)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dB, raw counts)')
plt.title('seq %d K %d peak %.3f MHz' % (sq, k, f[pb]))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out, dpi=120)
print('saved', out)
