import time
import numpy as np
from pynq import MMIO, allocate
K = globals().get('K', 1000)
NW = 32770
m = MMIO(0xB0000000, 0x1000)
rc = ol.dma.recvchannel
buf = allocate(shape=(NW,), dtype=np.uint64)
buf[:] = 0
print('K', K, 'integration (s):', round(K * 4096 / 69.12e6, 4), flush=True)
rc.transfer(buf)
m.write(0x04, K)
m.write(0x00, 1)
t0 = time.time()
while not rc.idle and time.time() - t0 < 10:
    time.sleep(0.01)
m.write(0x00, 0)
print('waited (s):', round(time.time() - t0, 3), 'idle', rc.idle, 'error', rc.error)
print('SEQ', m.read(0x0C), 'OVERRUN', m.read(0x10), 'STATUS', hex(m.read(0x1C)), 'GAPS', m.read(0x20))
d = np.array(buf)
np.save('capture_k%d.npy' % K, d)
print('nonzero words', int(np.count_nonzero(d)), 'of', NW)
print('timestamp', int(d[0]), 'K field', int(d[1]) >> 32, 'seq', int(d[1]) & 0xFFFFFFFF)
half = d[2:2 + 16384].astype(np.float64)
pb = int(np.argmax(half))
print('peak bin', pb, 'freq MHz', round(pb * 552.96 / 32768, 4), 'power', int(half[pb]))
print('NEXT: copy capture_k%d.npy to the Mac and send the output to Claude.' % K)
