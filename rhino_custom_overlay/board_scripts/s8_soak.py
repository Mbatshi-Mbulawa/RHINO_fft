exec(open('s1_load.py').read())
import time
import numpy as np
from pynq import MMIO, allocate
K = 16875
NP = 120
NW = 32770
time.sleep(3)
m = MMIO(0xB0000000, 0x1000)
rc = ol.dma.recvchannel
buf = allocate(shape=(NW,), dtype=np.uint64)
buf[:] = 0
rc.transfer(buf)
m.write(0x04, K)
m.write(0x00, 1)
t_start = time.time()
t_arm = t_start
rows = []
dts = []
for i in range(NP):
    while not rc.idle and time.time() - t_arm < 10:
        time.sleep(0.001)
    ok = rc.idle
    dts.append(time.time() - t_arm)
    rows.append((int(buf[0]), int(buf[1]) & 0xFFFFFFFF, int(buf[1]) >> 32, int(np.count_nonzero(buf)), ok))
    if not ok:
        print('STOP: packet', i, 'not complete', flush=True)
        break
    if i < NP - 1:
        buf[:] = 0
        rc.transfer(buf)
        t_arm = time.time()
m.write(0x00, 0)
n = len(rows)
ts = [r[0] for r in rows]
seq = [r[1] for r in rows]
diffs = [b - a for a, b in zip(ts, ts[1:])]
print('packets received', n, 'of', NP, 'total time (s) %.1f' % (time.time() - t_start))
print('seq first/last', seq[0], seq[-1], 'consecutive', all(b - a == 1 for a, b in zip(seq, seq[1:])))
print('K values', sorted(set(r[2] for r in rows)), 'min nonzero words', min(r[3] for r in rows))
print('timestamp differences: distinct values', sorted(set(diffs)), 'count', len(diffs))
print('host dt (s) min %.3f max %.3f' % (min(dts), max(dts)))
print('SEQ', m.read(0x0C), 'OVERRUN', m.read(0x10), 'STATUS', hex(m.read(0x1C)), 'GAPS', m.read(0x20))
