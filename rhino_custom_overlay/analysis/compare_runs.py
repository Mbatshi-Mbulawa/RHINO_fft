import sys
import numpy as np
a = np.load(sys.argv[1]).ravel()[2:].astype(np.float64)
b = np.load(sys.argv[2]).ravel()[2:].astype(np.float64)
ok = (a > 0) & (b > 0)
r = 10*np.log10(b[ok]/a[ok])
print('bins used', int(ok.sum()), 'of', a.size)
print('second/first ratio dB: mean %.3f std %.3f median %.3f' % (r.mean(), r.std(), np.median(r)))
print('corr of log power %.4f' % np.corrcoef(10*np.log10(a[ok]), 10*np.log10(b[ok]))[0,1])
