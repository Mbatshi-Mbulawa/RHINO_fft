import sys
import numpy as np
FS = 552.96e6
N = 32768
BW = FS / N
p = np.load(sys.argv[1])
K = int(p[1] >> np.uint64(32))
q = p[2:].astype(float)
if len(q) != N:
    sys.exit("expected %d power words, got %d" % (N, len(q)))
if len(sys.argv) > 2:
    kc = int(round(float(sys.argv[2]) * 1e6 / BW))
    k = kc - 20 + int(np.argmax(q[kc - 20:kc + 21]))
else:
    lo = q[100:N // 2]
    k = int(np.argmax(lo)) + 100
km = N - k
w = 3
mask = np.ones(N, bool)
mask[0] = False
for c in (k, km):
    mask[max(c - w, 0):c + w + 1] = False
floor = np.median(q[mask])
def near(c):
    return q[c - 1:c + 2].sum()
print("file", sys.argv[1], " K", K)
print("peak bin %d  freq %.4f MHz  block %d" % (k, k * BW / 1e6, k // 4096))
print("peak counts %.0f   (per frame %.1f)" % (q[k], q[k] / K))
print("mirror bin %d  block %d  counts %.0f" % (km, km // 4096, q[km]))
print("peak minus mirror %.2f dB (single bin)" % (10 * np.log10(q[k] / q[km])))
print("3-bin sum, peak minus mirror %.2f dB" % (10 * np.log10(near(k) / near(km))))
print("median of other bins %.1f counts (per frame %.3f)" % (floor, floor / K))
print("peak above median %.1f dB" % (10 * np.log10(q[k] / floor)))
