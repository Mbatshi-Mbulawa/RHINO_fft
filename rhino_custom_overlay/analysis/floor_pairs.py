import sys
import numpy as np
if len(sys.argv) < 2:
    print("usage: python floor_pairs.py capture.npy [baseline.npy]")
    sys.exit(1)
def load(fn):
    a = np.load(fn)
    K = int(a[1]) >> 32
    d = a[2:].astype(np.float64) / K
    med = np.array([np.median(d[b*4096:(b+1)*4096]) for b in range(8)])
    return K, med
def pairs(m):
    return [10*np.log10(m[a]/m[b]) for a, b in ((0,7),(1,6),(2,5),(3,4))]
K, med = load(sys.argv[1])
print(sys.argv[1], "K", K)
print("block medians per frame:", " ".join("%.3f" % m for m in med))
print("mirror pairs 0/7 1/6 2/5 3/4 (dB):", " ".join("%.2f" % p for p in pairs(med)))
if len(sys.argv) > 2:
    Kb, base = load(sys.argv[2])
    r = 10*np.log10(med/base)
    print("baseline", sys.argv[2], "K", Kb)
    print("capture / baseline by block (dB):", " ".join("%+.2f" % x for x in r))
    print("excess over baseline per frame:", " ".join("%.3f" % x for x in (med-base)))
    if np.all(np.abs(r) < 0.1):
        print("RESULT: within 0.1 dB of baseline in every block: floor limited")
    else:
        print("RESULT: some blocks differ from baseline by 0.1 dB or more")
