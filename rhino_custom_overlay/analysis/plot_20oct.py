"""Plots for the 20 Oct 2026 RHINO CW test (board 1 direct arm, board 2 amplifier arm).
Usage:  python plot_20oct.py /path/to/RHINO_fft
Needs numpy and matplotlib. Writes board1_spectra.png, board2_spectra.png, tone_vs_level.png
into the current directory. Data format: uint64 array of length 32770; element 1 holds
(K << 32) | seq; elements 2.. are the 32768 power sums; counts/frame = value / K."""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
N=32768; FS=552.96e6; BW=FS/N
REPO=sys.argv[1] if len(sys.argv)>1 else '.'
R=REPO.rstrip('/')+'/rhino_custom_overlay/data/'
def load(path):
    d=np.load(path); K=int(d[1]>>np.uint64(32)); assert d.shape==(N+2,)
    return d[2:].astype(float)/K, K
def get(board,name):
    return load(R+board+'/'+name)
B1='2026-10-20_board1_direct_arm'; B2='2026-10-20_board2_amp_arm'
SURF='#fcfcfb'; TXT='#0b0b0b'; TXT2='#52514e'; GRID='#e4e3df'
C1='#2a78d6'; C2='#eb6834'; C3='#1baf7a'; GR='#8d8c87'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,'text.color':TXT,'axes.labelcolor':TXT2,'xtick.color':TXT2,'ytick.color':TXT2,'axes.edgecolor':GRID,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f=np.arange(N//2)*BW/1e6
def db(q): return 10*np.log10(np.maximum(q[:N//2],1e-3))
def ax_style(ax):
    ax.grid(True,color=GRID,lw=0.7); ax.set_axisbelow(True)
def zoom(ax,series,f0=69.9975,hw=0.5):
    m=(f>f0-hw)&(f<f0+hw)
    for q,c,l in series: ax.plot(f[m],db(q)[m],color=c,lw=1.4,label=l)
    ax.axvline(f0,color=TXT2,lw=0.8,ls=(0,(3,3))); ax_style(ax)
    ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Power per bin (dB re 1 count/frame)')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
# ---- Board 1
b,_=get(B1,'base_b1_direct_off_115643.npy'); t60,_=get(B1,'tone_b1_direct_f70_m60_121022.npy'); t60r,_=get(B1,'tone_b1_direct_f70_m60_repeat_123238.npy'); t50,_=get(B1,'tone_b1_direct_f70_m50_123809.npy')
fig,axs=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.6,1]})
ax=axs[0]; ax.plot(f,db(b),color=GR,lw=0.7,label='baseline, generator carrier off'); ax.plot(f,db(t50),color=C1,lw=0.7,label='carrier on, -50 dBm')
ax_style(ax); ax.set_xlim(0,276.48); ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Power per bin (dB re 1 count/frame)'); ax.legend(frameon=False,fontsize=9,loc='upper right')
ax.set_title('Board 1, direct arm, full band',loc='left',fontsize=11,color=TXT)
zoom(axs[1],[(b,GR,'baseline'),(t60,C2,'-60 dBm'),(t50,C1,'-50 dBm')])
axs[1].set_title('Zoom: 69.9975 MHz ± 0.5 MHz',loc='left',fontsize=11,color=TXT)
fig.suptitle('20 Oct 2026, board 1 ADC_A, 1 s integrations (K = 16875)',x=0.01,ha='left',fontsize=11,color=TXT2)
fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig('board1_spectra.png',dpi=130); plt.close(fig)
# ---- Board 2
u,_=get(B2,'b2_amp_off2_124843.npy'); p1,_=get(B2,'b2_ampon_off_125602.npy'); p2,_=get(B2,'b2_ampon_off2_132211.npy'); t30,_=get(B2,'tone_b2_ampon_f70_m30_131659.npy'); t40,_=get(B2,'tone_b2_ampon_f70_m40b_131944.npy')
fig,axs=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={'width_ratios':[1.6,1]})
ax=axs[0]; ax.plot(f,db(u),color=GR,lw=0.7,label='amplifier off (12:48Z)'); ax.plot(f,db(p1),color=C2,lw=0.7,label='amplifier on, carrier off (12:56Z)'); ax.plot(f,db(p2),color=C1,lw=0.7,label='amplifier on, carrier off (13:22Z)')
ax_style(ax); ax.set_xlim(0,276.48); ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('Power per bin (dB re 1 count/frame)'); ax.legend(frameon=False,fontsize=9,loc='upper right')
ax.set_title('Board 2, amplifier arm, carrier-off baselines',loc='left',fontsize=11,color=TXT)
zoom(axs[1],[(p2,GR,'baseline (13:22Z)'),(t40,C2,'-40 dBm'),(t30,C1,'-30 dBm')])
axs[1].set_title('Tone: 69.9975 MHz ± 0.5 MHz',loc='left',fontsize=11,color=TXT)
fig.suptitle('20 Oct 2026, board 2 ADC_A, 1 s integrations (K = 16875)',x=0.01,ha='left',fontsize=11,color=TXT2)
fig.tight_layout(rect=(0,0,1,0.94)); fig.savefig('board2_spectra.png',dpi=130); plt.close(fig)
# ---- tone vs level
kc=4148
def exc(t,b): return (t-b)[kc]
pts1=[(-60,exc(t60,b)),(-60,exc(t60r,b)),(-50,exc(t50,b))]
pts2=[(-40,exc(t40,p2)),(-30,exc(t30,p2))]
fig,ax=plt.subplots(figsize=(6.6,4.6))
xs=np.array([-65,-25]);
ax.plot(xs,pts1[2][1]*10**((xs+50)/10),color=GR,lw=1,ls=(0,(4,3)))
ax.plot(xs,pts2[1][1]*10**((xs+30)/10),color=GR,lw=1,ls=(0,(4,3)))
ax.plot([p[0] for p in pts1],[p[1] for p in pts1],'o',color=C1,ms=8,mec=SURF,mew=1.5,label='board 1, direct arm')
ax.plot([p[0] for p in pts2],[p[1] for p in pts2],'o',color=C2,ms=8,mec=SURF,mew=1.5,label='board 2, amplifier arm (powered)')
ax.set_yscale('log'); ax_style(ax); ax.set_xlim(-65,-25); ax.set_ylim(2,1500)
ax.set_xlabel('Generator level setting (dBm)'); ax.set_ylabel('Tone excess over baseline (counts/frame)')
ax.legend(frameon=False,fontsize=9,loc='upper left')
ax.set_title('Tone strength at 69.9975 MHz',loc='left',fontsize=11,color=TXT)
ax.text(-64.5,2.4,'dashed lines: 10 dB per decade',color=TXT2,fontsize=8.5)
fig.tight_layout(); fig.savefig('tone_vs_level.png',dpi=130); plt.close(fig)
print('ok', pts1, pts2)
