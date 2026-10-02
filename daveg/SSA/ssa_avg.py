import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

import mplhep
plt.style.use(mplhep.style.ROOT)
params = {'figure.figsize': (8.5,6)}
plt.rcParams.update(params)

## read in runlist##
filename = "copper_ssa_z0p5_piminus.txt"
if (os.path.isfile(filename)):
    phia,asya,easya = np.loadtxt(filename,unpack=True,comments='#')

filename = "copper_ssa_z0p5_piminus_thpq0p8.txt"
if (os.path.isfile(filename)):
    phib,asyb,easyb = np.loadtxt(filename,unpack=True,comments='#')

asyavg=(asya/easya**2 + asyb/easyb**2)/(1.0/easya**2+1.0/easyb**2)
easyavg=np.sqrt(1.0/(1.0/easya**2+1.0/easyb**2))

#plot the data
plt.figure()
ax1=plt.subplot(111)
ax1.set_xlabel('$\phi_{pq}$')
ax1.set_ylabel('$A_{LU}$')
#ax1.errorbar(phia, asya, yerr=easya, fmt='o', color='blue', markersize=2,label='thetapq=2 deg')
#ax1.errorbar(phib, asyb, yerr=easyb, fmt='s', color='r', markersize=2,label='thetapq=-0.8 deg')
ax1.errorbar(phib, asyavg, yerr=easyavg, fmt='s', color='blue', markersize=4)

myPi=3.1415926536
ax1.set_ylim(bottom=-0.1,top=0.1)
ax1.set_xlim(left=-myPi,right=myPi)
plt.legend()


#fit stuff

def A_LT(phi,A):
    return A*np.sin(phi)

XL=[-myPi,myPi]
YL=[0.0,0.0]
ax1.plot(XL,YL,color='black',linewidth=0.8,linestyle='--')

x0=np.array([1.0])
popt,pcov=opt.curve_fit(A_LT,phib,asyavg,p0=x0,sigma=easyavg,absolute_sigma=True)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)

phitmp=np.arange(-myPi,myPi,0.1)
afit=A_LT(phitmp,popt[0])
plt.plot(phitmp,afit,'r-')

plt.text(-2.8,0.07,'$A_{LU}=0.0176 +/- 0.0086$')
plt.text(1.5,-0.08,'Copper')

plt.tight_layout()
plt.savefig('ssa_copper.pdf')
plt.show()
