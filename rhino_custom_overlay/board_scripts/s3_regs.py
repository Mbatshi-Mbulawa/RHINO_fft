import sys
from pynq import MMIO
addr = 0xB0000000
if 'top0' in ol.ip_dict:
    addr = ol.ip_dict['top0']['phys_addr']
print('top0 base address', hex(addr), '(expected 0xb0000000)')
if addr != 0xB0000000:
    print('ADDRESS DIFFERS. STOP.')
else:
    m = MMIO(addr, 0x1000)
    print('reading ID (hangs here if the 69.12 MHz clock is missing)', flush=True)
    idv = m.read(0x24)
    print('ID', hex(idv), '(expected 0x52464654)', flush=True)
    if idv != 0x52464654:
        print('ID WRONG. STOP.')
    else:
        for name, off in (('NPTS', 0x28), ('K', 0x04), ('SCHED', 0x08), ('CTRL', 0x00), ('STATUS', 0x1C), ('SEQ', 0x0C), ('OVERRUN', 0x10), ('GAPS', 0x20)):
            v = m.read(off)
            print(name, v, hex(v))
        print('NEXT: send this output to Claude. Nothing was written.')
