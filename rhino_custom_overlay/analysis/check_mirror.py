import sys
import numpy as np
d = np.load(sys.argv[1]).ravel()
print('words', d.size, 'dtype', d.dtype)
p = d[2:].astype(np.float64)
N = p.size
print('N', N, 'zero words', int((p == 0).sum()))
for b in range(8):
    s = p[b*4096:(b+1)*4096]
    print('block', b, 'mean dB %.2f' % (10*np.log10(s.mean())), 'zeros', int((s == 0).sum()))
k = np.arange(1, N//2)
a = p[k]
m = p[N - k]
ok = (a > 0) & (m > 0)
r = 10*np.log10(a[ok]/m[ok])
print('mirror pairs', int(ok.sum()), 'of', k.size)
print('mirror ratio dB: mean %.3f std %.3f max abs %.2f' % (r.mean(), r.std(), np.abs(r).max()))
print('corr of log power k vs N-k %.3f' % np.corrcoef(10*np.log10(a[ok]), 10*np.log10(m[ok]))[0,1])
