exec(open('s1_load.py').read())
import xrfdc, pprint, traceback
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
try:
    pprint.pprint(r.IPStatus)
    t = r.adc_tiles[2]
    print([n for n in dir(t) if not n.startswith('_')])
except Exception:
    traceback.print_exc()
