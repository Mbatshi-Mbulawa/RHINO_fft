exec(open('s1_load.py').read())
import xrfdc
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
t = r.adc_tiles[2]
print('blocks', len(t.blocks))
for i, b in enumerate(t.blocks):
    if not (i == 1 or len(t.blocks) == 1):
        continue
    print('--- block', i)
    for n in sorted(dir(b)):
        if n.startswith('_'):
            continue
        try:
            v = getattr(b, n)
        except Exception as e:
            v = 'ERR ' + repr(e)
        if callable(v):
            continue
        print(n, v)
