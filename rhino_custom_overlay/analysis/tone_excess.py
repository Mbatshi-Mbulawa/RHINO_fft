import sys
import numpy as np
N = 32768
BW = 552.96e6 / N
t = np.load(sys.argv[1])
b = np.load(sys.argv[2])
kc = int(round(float(sys.argv[3]) * 1e6 / BW))
Kt = int(t[1] >> np.uint64(32))
Kb = int(b[1] >> np.uint64(32))
qt = t[2:].astype(float) / Kt
qb = b[2:].astype(float) / Kb
ex = qt - qb
k = kc - 20 + int(np.argmax(ex[kc - 20:kc + 21]))
km = N - k
e1 = ex[k - 1:k + 2].sum()
e2 = ex[km - 1:km + 2].sum()
print("tone", sys.argv[1], "baseline", sys.argv[2])
print("peak bin %d  freq %.4f MHz  block %d" % (k, k * BW / 1e6, k // 4096))
print("floor per frame at peak %.3f, at mirror %.3f" % (qb[k], qb[km]))
print("excess per frame: peak %.3f  mirror %.3f  (3-bin: %.3f %.3f)" % (ex[k], ex[km], e1, e2))
if ex[km] > 0 and ex[k] > 0:
    print("peak minus mirror, excess only: %.2f dB (single bin)" % (10 * np.log10(ex[k] / ex[km])))
if e1 > 0 and e2 > 0:
    print("peak minus mirror, excess only: %.2f dB (3-bin)" % (10 * np.log10(e1 / e2)))
print("peak excess over floor at peak: %.1f dB" % (10 * np.log10(max(ex[k], 1e-9) / qb[k] + 1)))
