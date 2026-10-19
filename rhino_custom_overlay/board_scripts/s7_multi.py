exec(open('s1_load.py').read())
import time
import numpy as np
from pynq import MMIO, allocate
K = 16875
NP = 5
NW = 32770
time.sleep(3)
m = MMIO(0xB0000000, 0x1000)
rc = ol.dma.recvchannel
buf = allocate(shape=(NW,), dtype=np.uint64)
pk = []
print('K', K, 'packets', NP, 'expected spacing (clocks):', K * 4096, flush=True)
buf[:] = 0
rc.transfer(buf)
m.write(0x04, K)
m.write(0x00, 1)
t_arm = time.time()
for i in range(NP):
    while not rc.idle and time.time() - t_arm < 10:
        time.sleep(0.001)
    t_done = time.time()
    ok = rc.idle
    d = np.array(buf)
    pk.append(d)
    print('packet', i, 'idle', ok, 'host dt (s) %.3f' % (t_done - t_arm), 'seq', int(d[1]) & 0xFFFFFFFF, 'K', int(d[1]) >> 32, 'ts', int(d[0]), 'nonzero', int(np.count_nonzero(d)), flush=True)
    if not ok:
        break
    if i < NP - 1:
        buf[:] = 0
        rc.transfer(buf)
        t_arm = time.time()
m.write(0x00, 0)
print('SEQ', m.read(0x0C), 'OVERRUN', m.read(0x10), 'STATUS', hex(m.read(0x1C)), 'GAPS', m.read(0x20))
ts = [int(p[0]) for p in pk]
print('timestamp differences (clocks):', [b - a for a, b in zip(ts, ts[1:])])
np.save('multi_k%d_n%d.npy' % (K, NP), np.array(pk))
print('saved multi_k%d_n%d.npy' % (K, NP))
