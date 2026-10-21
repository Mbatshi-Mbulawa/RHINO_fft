"""Estimate the total power at the ADC and how much of its range is used, from one capture.
Usage: python total_power.py capture.npy [baseline.npy]
Method: a tone of power P gives C counts/frame in one bin, so for any signal the sum of counts/frame over the
positive-frequency bins is proportional to its total power (Parseval): P = P_CAL + 10 log10(sum / C_CAL).
Assumptions (each uncertain, see notes): calibration point from 19 Oct (0 dBm dial, 30 dB attenuator ->
21,213.575 counts/frame, taken as -29.9 dBm at the ADC); ADC full scale 1 Vppd = +0.97 dBm sine (data sheet typical,
DSA 0 dB; our DSA setting is not known); noise-like signal; window ENBW ignored (+0 to -1.5 dB).
The carrier-off floor (about 0.35 counts/bin) is the design's own numeric floor, NOT ADC input noise: it shows up
here as about -31 dBm, so this estimate only means something once the total is well above about -28 dBm.
With a baseline given, the baseline sum is subtracted first (excess power only)."""
import sys, numpy as np
from math import erfc, sqrt, log2
N=32768; C_CAL=21213.575; P_CAL=-29.9; FS_SINE_DBM=0.97; FS_PEAK_LSB=8192   # 14-bit
def counts(p):
    d=np.load(p); K=int(d[1]>>np.uint64(32)); return d[2:N//2+2].astype(float)/K
c=counts(sys.argv[1]); s=c.sum()
label='total'
if len(sys.argv)>2: s=s-counts(sys.argv[2]).sum(); label='excess over baseline'
if s<=0: sys.exit('sum of counts is not above the baseline; nothing to estimate')
p=P_CAL+10*np.log10(s/C_CAL)
sig=FS_PEAK_LSB/sqrt(2)*sqrt(10**((p-FS_SINE_DBM)/10))
ns=FS_PEAK_LSB/sig
print(f'sum of counts/frame over positive bins ({label}): {s:.1f}')
print(f'estimated power at ADC ({label}): {p:.1f} dBm')
print(f'rms about {sig:.0f} LSB (14-bit), {100*sig/FS_PEAK_LSB:.2f}% of full-scale peak, full scale is {ns:.1f} sigma away')
print(f'approx bits spanned (8 sigma peak to peak): {log2(8*sig):.1f} of 14')
print(f'Gaussian clipping rate at 552.96 MSPS: about {erfc(ns/sqrt(2))*552.96e6:.3g} samples/s')
if p<-28: print('NOTE: near or below the design floor equivalent (about -31 dBm); this number is not reliable here')
if ns<6: print('WARNING: less than 6 sigma to full scale. Reduce the level before going further')
