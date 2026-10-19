exec(open('s1_load.py').read())
import xrfdc
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
t = r.adc_tiles[2]
for n in ['PLLLockStatus', 'ClockSource', 'FabClkOutDiv', 'PLLConfig']:
    try:
        print(n, getattr(t, n))
    except Exception as e:
        print(n, 'ERR', repr(e))
