import sys
import numpy as np
pk = []
cur = []
for line in open(sys.argv[1]):
    w, t = line.split()
    cur.append(int(w, 16))
    if t == '1':
        pk.append(cur)
        cur = []
print('packets', len(pk), [len(p) for p in pk])
for i, p in enumerate(pk):
    np.save('sim_p%d.npy' % i, np.array(p, dtype=np.uint64))
