exec(open('s1_load.py').read())
import traceback, xrfdc
try:
    r = xrfdc.RFdc(ol.ip_dict['rfdc'])
    print('RFdc OK', type(r))
    print([n for n in dir(r) if not n.startswith('_')])
except Exception:
    traceback.print_exc()
