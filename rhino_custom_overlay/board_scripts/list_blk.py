exec(open('s1_load.py').read())
import xrfdc
r = xrfdc.RFdc(ol.ip_dict['rfdc'])
b = r.adc_tiles[2].blocks[1]
names = [n for n in dir(b) if not n.startswith('_')]
print('block names:', names)
print('matches:', [n for n in names if any(k in n.lower() for k in ('dsa', 'atten', 'gain'))])
