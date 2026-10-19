exec(open('s1_load.py').read())
import xrfdc, time
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
time.sleep(3)
b = r.adc_tiles[2].blocks[1]
try:
    v = b.DSA
    print('DSA value:', v, '| type:', type(v).__name__)
    try:
        print('DSA fields:', dict(v))
    except Exception as e:
        print('not a mapping:', e)
except Exception as e:
    print('DSA read failed:', repr(e))
