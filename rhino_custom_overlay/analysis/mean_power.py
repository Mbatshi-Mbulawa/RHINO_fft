"""Mean power spectra (counts/frame, linear units, log y axis) for the 20 Oct 2026 CW test.
Usage: python mean_power.py /path/to/RHINO_fft
Writes mean_power_fullband.png and mean_power_70MHz.png. Each stored bin is a sum over K frames,
so value/K is the mean power per frame in that bin."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N=32768; FS=552.96e6; BW=FS/N
R=(sys.argv[1] if len(sys.argv)>1 else '.').rstrip('/')+'/rhino_custom_overlay/data/'
def get(b,n):
    d=np.load(R+b+'/'+n); K=int(d[1]>>np.uint64(32)); assert d.shape==(N+2,)
    return d[2:N//2+2].astype(float)/K
B1='2026-10-20_board1_direct_arm'; B2='2026-10-20_board2_amp_arm'
f=np.arange(N//2)*BW/1e6
bl,c1,c2,c3,gr='#2a78d6','#eb6834','#1baf7a','#8d8c87','#52514e'
S1=[(B1,'base_b1_direct_off_115643.npy','baseline, carrier off',gr),(B1,'tone_b1_direct_f70_m60_121022.npy','-60 dBm',c1),(B1,'tone_b1_direct_f70_m50_123809.npy','-50 dBm',bl)]
S2=[(B2,'b2_amp_off2_124843.npy','amplifier off, carrier off',gr),(B2,'b2_ampon_off2_132211.npy','amplifier on, carrier off (13:22Z)',c3),(B2,'tone_b2_ampon_f70_m40b_131944.npy','-40 dBm',c1),(B2,'tone_b2_ampon_f70_m30_131659.npy','-30 dBm',bl)]
def fig(name,xlim,w):
    fg,axs=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,S,t in zip(axs,(S1,S2),('Board 1, direct arm','Board 2, amplifier arm')):
        for b,n,l,c in S:
            q=get(b,n); m=(f>=xlim[0])&(f<=xlim[1]); ax.plot(f[m],np.maximum(q[m],1e-3),color=c,lw=w,label=l)
        ax.set_yscale('log'); ax.set_xlim(*xlim); ax.grid(True,color='#e4e3df',lw=0.7)
        ax.set_xlabel('Frequency (MHz)'); ax.set_title(t,loc='left',fontsize=11); ax.legend(frameon=False,fontsize=8.5,loc='upper right')
        for s in ('top','right'): ax.spines[s].set_visible(False)
    axs[0].set_ylabel('Mean power per bin (counts/frame)')
    fg.tight_layout(); fg.savefig(name,dpi=130); plt.close(fg)
fig('mean_power_fullband.png',(0,276.48),0.6)
fig('mean_power_70MHz.png',(69.4975,70.4975),1.3)
