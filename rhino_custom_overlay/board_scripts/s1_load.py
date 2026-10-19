import sys
import xrfclk
from pynq import Overlay
BIT = sys.argv[1] if len(sys.argv) > 1 else 'pfb.bit'
print('step 1: set_ref_clks(lmk_freq=245.76, lmx_freq=491.52)')
xrfclk.set_ref_clks(lmk_freq=245.76, lmx_freq=491.52)
print('step 2: Overlay', BIT)
ol = Overlay(BIT)
print('step 3: IP names in overlay (no driver created):')
print(sorted(ol.ip_dict.keys()))
print('STOP HERE. Do not touch ol.dma or ol.top0 yet.')
