import sys
print('python', sys.version.split()[0])
try:
    import pynq
    print('pynq', pynq.__version__)
except Exception as e:
    print('pynq import failed:', e)
for m in ('xrfclk', 'xrfdc'):
    try:
        mod = __import__(m)
        print(m, 'OK', getattr(mod, '__file__', '?'))
    except Exception as e:
        print(m, 'import failed:', e)
try:
    print(open('/etc/os-release').read().split('\n')[0:3])
except Exception as e:
    print('os-release failed:', e)
