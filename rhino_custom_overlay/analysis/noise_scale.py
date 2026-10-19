import sys
import numpy as np
N = 32768
BW = 552.96e6 / N
t = np.load(sys.argv[1])
b = np.load(sys.argv[2])
kc = int(round(float(sys.argv[3]) * 1e6 / BW))
km = N - kc
Kt = int(t[1] >> np.uint64(32))
Kb = int(b[1] >> np.uint64(32))
ex = t[2:].astype(float) / Kt - b[2:].astype(float) / Kb
def local_sd(c):
    idx = np.r_[c - 300:c - 20, c + 21:c + 301]
    return ex[idx].std(), ex[idx].mean()
sp, mp = local_sd(kc)
sm, mm = local_sd(km)
print(sys.argv[1])
print("noise sd per frame near tone bin %.4f (mean %.4f), near mirror bin %.4f (mean %.4f)" % (sp, mp, sm, mm))
print("excess at tone bin %.4f = %.1f sigma, at mirror bin %.4f = %.1f sigma" % (ex[kc], (ex[kc] - mp) / sp, ex[km], (ex[km] - mm) / sm))
