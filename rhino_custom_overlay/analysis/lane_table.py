import sys
import numpy as np
N = 32768
p = np.load(sys.argv[1])
K = int(p[1] >> np.uint64(32))
q = p[2:].astype(float) / K
k = np.arange(N)
ok = (k >= 100) & (k <= N - 100)
print(sys.argv[1], "K", K)
print("median floor per frame; rows = block 0..7 (4096 bins each), columns = lane (bin mod 8)")
for b in range(8):
    row = []
    for l in range(8):
        m = ok & (k // 4096 == b) & (k % 8 == l)
        row.append(np.median(q[m]))
    print("block %d  " % b + "  ".join("%.3f" % x for x in row))
