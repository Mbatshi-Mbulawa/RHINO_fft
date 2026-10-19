exec(open('s1_load.py').read())
import time, xrfdc
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
def st(tag):
    s = r.IPStatus['ADCTileStatus'][2]
    print(tag, 'tile2', s, 'PLLLock', r.adc_tiles[2].PLLLockStatus)
st('before')
time.sleep(3)
st('after 3 s')
exec(open('s4_capture.py').read())
st('after capture')
