def show(label, obj):
    names = [n for n in dir(obj) if not n.startswith('_')]
    print(label, type(obj).__name__, names)
    return names
rfdc = ol.rfdc
show('rfdc:', rfdc)
if hasattr(rfdc, 'IPStatus'):
    try:
        print('IPStatus =', rfdc.IPStatus)
    except Exception as e:
        print('IPStatus read failed:', e)
tiles = getattr(rfdc, 'adc_tiles', None)
if tiles is None:
    tiles = getattr(rfdc, 'adc', None)
print('adc tiles object:', type(tiles).__name__)
t = tiles[2]
tn = show('tile 2:', t)
for n in tn:
    if any(k in n for k in ('Status', 'State', 'Lock', 'Clock')):
        try:
            v = getattr(t, n)
            if not callable(v):
                print('  tile 2', n, '=', v)
        except Exception as e:
            print('  tile 2', n, 'read failed:', e)
print('NEXT: send this output to Claude. Do not call any StartUp or Shutdown method yet.')
