import uproot
import awkward
import pandas as pd
from array import array
import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import boost_histogram as bh
from plothist import make_hist, plot_hist, plot_error_hist
from fetchscalers_coin import fetchscalers_coin
from fetchnormfac import fetchnormfac
from makehelhistos import makehelhistos

# set up plotting defaults
size=15
params = {'legend.fontsize': size,
          'font.weight':'normal',
          'figure.figsize': (8.5,8.5), #size of figure
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size,
          'font.size':size,
          'axes.titlepad': size,
          'axes.linewidth': 1,
          'lines.linewidth': 1,
          'mathtext.default': 'regular'}
plt.rcParams.update(params)
#IHWP states
#25345->Inf: new Wien angle!
#25325-???? IN
#25062-25324 OUT
#24861-25061 IN


## read in runlist##
filename = "rsidis_runlist.dat"
if (os.path.isfile(filename)):
    target,runtype =np.loadtxt(filename,unpack=True,comments='!',usecols=(5,11),dtype='str')
    runno,ebeam,phms,th_hms,pshms,th_shms = np.loadtxt(filename,unpack=True,comments='!', usecols=(0,3,6,7,8,9),dtype='float')

mytype='PI-SIDIS'
#mymom=6.538 #z=0.9
#mymom=-4.868 #z=0.67
mymom=-3.632 #z=0.5
mytheta=7.51
#mytheta=10.305
mybeam=10.67

mytarg='Carbon'
cut = (mytarg==target) & (mytype==runtype) & (mymom==pshms) & (mytheta==th_shms) & (mybeam==ebeam)
dataruns=runno[cut]

doing_cryo=False
if mytarg=='LH2' or mytarg=='LD2':
    doing_cryo=True

if doing_cryo:
    mytarg='Dummy'
    cut = (mytarg==target) & (mytype==runtype) & (mymom==pshms) & (mytheta==th_shms) & (mybeam==ebeam)
    dummyruns=np.array(runno[cut])

print('Data runs: ',dataruns)
if doing_cryo:
    print('Dummy runs: ',dummyruns)


nvar=17
histos={}
for i in range(nvar):
    histos[i]={}

nhist=13
myPi=3.1415926536
for i in range(nhist):
    # HMS reconstructed quantities
    histos[0,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-8.0, stop=8.0), storage=bh.storage.Weight()) # HMS delta
    histos[1,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.1, stop=0.1), storage=bh.storage.Weight()) # HMS xptar
    histos[2,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.06, stop=0.06), storage=bh.storage.Weight()) # HMS yptar
    histos[3,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-5.0, stop=5.0), storage=bh.storage.Weight()) # HMS ytar
    #SHMS reconstructed quantities
    histos[4,i] = bh.Histogram(bh.axis.Regular(bins=30, start=-10.0, stop=20.0), storage=bh.storage.Weight()) # SHMS delta
    histos[5,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.1, stop=0.1), storage=bh.storage.Weight()) # SHMS xptar
    histos[6,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-0.06, stop=0.06), storage=bh.storage.Weight()) # SHMS yptar
    histos[7,i] = bh.Histogram(bh.axis.Regular(bins=50, start=-5.0, stop=5.0), storage=bh.storage.Weight()) # SHMS ytar
    # Physics things
    histos[8,i] = bh.Histogram(bh.axis.Regular(bins=50, start=2.0, stop=4.0), storage=bh.storage.Weight()) # W
    histos[9,i] = bh.Histogram(bh.axis.Regular(bins=50, start=1.0, stop=4.0), storage=bh.storage.Weight()) # Q2
    histos[10,i] = bh.Histogram(bh.axis.Regular(bins=50, start=0.1, stop=0.4), storage=bh.storage.Weight()) # xbj
    histos[11,i] = bh.Histogram(bh.axis.Regular(bins=45, start=0.1, stop=1.0), storage=bh.storage.Weight()) # z
    histos[12,i] = bh.Histogram(bh.axis.Regular(bins=50, start=0.0, stop=0.5), storage=bh.storage.Weight()) # Pt
    histos[13,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-myPi, stop=myPi), storage=bh.storage.Weight()) # Phi
    histos[14,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-myPi, stop=myPi), storage=bh.storage.Weight()) # Phi h+
    histos[15,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-myPi, stop=myPi), storage=bh.storage.Weight()) # Phi h-
    histos[16,i] = bh.Histogram(bh.axis.Regular(bins=16, start=-myPi, stop=myPi), storage=bh.storage.Weight()) # Asy

#####Data runs ######
qtot_data = 0
#qtot_data = makedathistos(dataruns,1,histos,nvar,1.0)
qtot_data = makehelhistos(dataruns,1,histos,nvar,1.0)

##Dummy runs
#LH2
dumrat=3.550
# LD2
#dumrat=3.7825

if doing_cryo:
    ndummy=len(dummyruns)
    qtot_dummy = 0
    #qtot_dummy = makedathistos(dummyruns,6,histos,nvar,dumrat)
    qtot_dummy = makehelhistos(dummyruns,6,histos,nvar,dumrat)


#subtract dummy from the data
for i in range(nvar-1):
    histos[i,11]=histos[i,3]+histos[i,8]*(-1.0)


#calculate the asymmetry
phi_hplus = histos[14,3].counts()
phi_hplus_err2 = histos[14,3].variances()
phi_hminus = histos[15,3].counts()
phi_hminus_err2 = histos[15,3].variances()



# Access the first (and only) axis
x_axis = histos[15,3].axes[0]
# Get the bin centers as a NumPy array
phicent = x_axis.centers



phi_asy = (phi_hplus-phi_hminus)/(phi_hplus+phi_hminus+0.000001)
phisum = phi_hminus+phi_hplus
ephi_asy2 = 4.0*(phi_hminus_err2*phi_hplus**2 + phi_hplus_err2*phi_hminus**2)/phisum**4
ephi_asy=np.sqrt(ephi_asy2)

print('bin centers', phicent,len(phicent))
print('phy_asy', phi_asy,len(phi_asy))

def A_LT(phi,A):
    return A*np.sin(phi)

x0=np.array([1.0])
popt,pcov=opt.curve_fit(A_LT,phicent,phi_asy,p0=x0,sigma=ephi_asy,absolute_sigma=True)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)

with PdfPages('multiple_figures.pdf') as pdf:

    plt.figure(1)
    ax1=plt.subplot(221)
    ax1.set_xlabel('HMS delta')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[0,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[0,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[0,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
#    plot_hist(histos[0,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('HMS xptar')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[1,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[1,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[1,11], ax=ax2, color='blue', linewidth=1.2)
#    plot_hist(histos[1,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    ax3=plt.subplot(223)
    ax3.set_xlabel('HMS yptar')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[2,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[2,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[2,11], ax=ax3, color='blue', linewidth=1.2)
#    plot_hist(histos[2,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('HMS ytar')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[3,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[3,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[3,11], ax=ax4, color='blue', linewidth=1.2)
#    plot_hist(histos[3,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()
    pdf.savefig()

    plt.figure(2)
    ax1=plt.subplot(221)
    ax1.set_xlabel('SHMS delta')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[4,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[4,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[4,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
#    plot_hist(histos[4,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('SHMS xptar')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[5,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[5,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[5,11], ax=ax2, color='blue', linewidth=1.2)
#    plot_hist(histos[5,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    ax3=plt.subplot(223)
    ax3.set_xlabel('SHMS yptar')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[6,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[6,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[6,11], ax=ax3, color='blue', linewidth=1.2)
#    plot_hist(histos[6,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('SHMS ytar')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[7,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[7,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[7,11], ax=ax4, color='blue', linewidth=1.2)
#    plot_hist(histos[7,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()

    pdf.savefig()


    plt.figure(3)
    ax1=plt.subplot(221)
    ax1.set_xlabel('W')
    ax1.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[8,3], ax=ax1, color='black',linewidth=1.2, label="Data")
    plot_error_hist(histos[8,8], ax=ax1, color='red', linewidth=1.2, label="Dummy")
    plot_error_hist(histos[8,11], ax=ax1, color='blue', linewidth=1.2, label="Data-dummy")
#    plot_hist(histos[8,12], ax=ax1, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax1.set_ylim(bottom=0)
    plt.tight_layout()

    ax2=plt.subplot(222)
    ax2.set_xlabel('Q2')
    ax2.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[9,3], ax=ax2, color='black',linewidth=1.2)
    plot_error_hist(histos[9,8], ax=ax2, color='red', linewidth=1.2)
    plot_error_hist(histos[9,11], ax=ax2, color='blue', linewidth=1.2)
#    plot_hist(histos[9,12], ax=ax2, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax2.set_ylim(bottom=0)
    plt.tight_layout()

    ax3=plt.subplot(223)
    ax3.set_xlabel('xBj')
    ax3.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[10,3], ax=ax3, color='black',linewidth=1.2)
    plot_error_hist(histos[10,8], ax=ax3, color='red', linewidth=1.2)
    plot_error_hist(histos[10,11], ax=ax3, color='blue', linewidth=1.2)
#    plot_hist(histos[10,12], ax=ax3, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax3.set_ylim(bottom=0)
    plt.tight_layout()

    ax4=plt.subplot(224)
    ax4.set_xlabel('Zhad')
    ax4.set_ylabel('Yield (counts/mC)')
    plot_error_hist(histos[11,3], ax=ax4, color='black',linewidth=1.2)
    plot_error_hist(histos[11,8], ax=ax4, color='red', linewidth=1.2)
    plot_error_hist(histos[11,11], ax=ax4, color='blue', linewidth=1.2)
#    plot_hist(histos[11,12], ax=ax4, histtype="step",color='blue', linewidth=1.2, label="SIMC")
    ax4.set_ylim(bottom=0)
    plt.tight_layout()
    pdf.savefig()


    plt.figure(4)
    ax1=plt.subplot(111)
    ax1.set_xlabel('phipq')
    ax1.set_ylabel('asymmetry')
    ax1.errorbar(phicent, phi_asy, yerr=ephi_asy, fmt='o', color='blue', markersize=2)
    phitmp=np.arange(-myPi,myPi,0.1)
    afit=A_LT(phitmp,popt[0])
    plt.plot(phitmp,afit,'r-')

    # phi-dependent asymmetries
    print('asymmetries')
    for i in range(len(phicent)):
        print(phicent[i],phi_asy[i],ephi_asy[i])


    plt.tight_layout()
    pdf.savefig()


plt.show()
