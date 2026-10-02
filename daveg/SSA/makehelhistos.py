import uproot
import awkward
import pandas as pd
from array import array
import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import boost_histogram as bh
from plothist import make_hist, plot_hist, plot_error_hist
from fetchscalers_coin import fetchscalers_coin
from fetchnormfac import fetchnormfac

def makehelhistos(runs,hid,histos,nvar,dumrat):
    #####Data runs ######
    qtot = 0
    realid=hid
    ranid=hid+1
    subid=hid+2

    print('in makedathistos hid:', hid)
    nrun=len(runs)
    #First, loop over the runs to get the total efficiency-corrected charge
    for i in range(nrun):
        run=str(int(runs[i]))
        print('getting efficiency-corrected charge',run)
        (qdum,curdum,hfidtmp,pfidtmp,clttmp,qtmp)=fetchscalers_coin(run)
        qtot=qtot+qtmp

    realwgt=1.0/(dumrat*qtot)
    ranwgt=1.0/(qtot*3.0*dumrat)

    # Loop over the runs again to fill the histograms
    for i in range(nrun):
        run=str(int(runs[i]))
        rootfile='ROOTfiles/coin_replay_production_'+run+'_-1.root:T'

        print('Analyzing run ',run)
        print('Opening root file ', rootfile)

        f=uproot.open(rootfile)

        hsdelta = f['H.gtr.dp'].array(library="np") #open each value we want into a numpy array
        hsyptar = f['H.gtr.ph'].array(library="np")
        hsxptar = f['H.gtr.th'].array(library="np")
        hsytar = f['H.gtr.y'].array(library="np")
        hcer_npe = f['H.cer.npeSum'].array(library="np")
        hsshsum = f['H.cal.etottracknorm'].array(library="np")
        psdelta = f['P.gtr.dp'].array(library="np")
        psyptar = f['P.gtr.ph'].array(library="np")
        psxptar = f['P.gtr.th'].array(library="np")
        psytar = f['P.gtr.y'].array(library="np")
        paero_npe = f['P.aero.npeSum'].array(library="np")
        phgc_npe = f['P.hgcer.npeSum'].array(library="np")
        ppi = f['P.gtr.p'].array(library="np")
        nu = f['H.kin.primary.nu'].array(library="np")
        xbj = f['H.kin.primary.x_bj'].array(library="np")
        W = f['H.kin.primary.W'].array(library="np")
        Q2 = f['H.kin.primary.Q2'].array(library="np")
        phipq = f['P.kin.secondary.ph_xq'].array(library="np")
        thetapq = f['P.kin.secondary.th_xq'].array(library="np")
        ctime = f['CTime.ePiCoinTime_ROC1'].array(library="np")
        hel = f['T.helicity.hel'].array(library="np")

        #combine all of the numpy arrays into a pandas dataframe
        df = pd.DataFrame({'hsdelta': hsdelta, 'hsxptar': hsxptar, 'hsyptar':hsyptar, 'hsytar':hsytar, 'hcer_npe': hcer_npe, 'hsshsum': hsshsum, 'psdelta': psdelta, 'psyptar': psyptar,  'psxptar': psxptar, 'psytar': psytar, 'paero_npe':paero_npe, 'phgc_npe':phgc_npe, 'ppi': ppi, 'nu': nu, 'xbj': xbj, 'W': W, 'Q2': Q2, 'phipq': phipq, 'thetapq': thetapq, 'ctime': ctime, 'hel': hel})
        # this needs tobe done before defining cuts - not sure why
        df['zhad'] = np.sqrt(df['ppi']**2+0.13957**2)/df['nu']
        df['Pt'] = df['ppi']*np.sin(df['thetapq'])

        df_realcut = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-51.2)<2.0)]
        df_realcut_hplus = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-51.2)<2.0) & (hel>0.5)]
        df_realcut_hminus = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-51.2)<2.0) & (hel<-0.5)]

        df_rancut = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-39.2)<6.0)]
        df_rancut_hplus = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-39.2)<6.0) & (hel>0.5)]
        df_rancut_hminus = df[abs(df['hsdelta'] < 8) & (df['hcer_npe'] > 1) & (df['hsshsum'] > 0.7) & (df['psdelta']>-10.0) & (df['psdelta']<20.0) & (df['paero_npe']>2.0) & (df['phgc_npe']>1.0) & (abs(df['ctime']-39.2)<6.0) & (hel<-0.5)]

        #Fill the histograms
        histos[0,1].fill(df_realcut['hsdelta'],weight=realwgt)
        histos[0,2].fill(df_rancut['hsdelta'],weight=ranwgt)

        histos[1,realid].fill((df_realcut['hsxptar']),weight=realwgt)
        histos[1,ranid].fill((df_rancut['hsxptar']),weight=ranwgt)

        histos[2,realid].fill((df_realcut['hsyptar']),weight=realwgt)
        histos[2,ranid].fill((df_rancut['hsyptar']),weight=ranwgt)

        histos[3,realid].fill((df_realcut['hsytar']),weight=realwgt)
        histos[3,ranid].fill((df_rancut['hsytar']),weight=ranwgt)

        histos[4,realid].fill(df_realcut['psdelta'],weight=realwgt)
        histos[4,ranid].fill(df_rancut['psdelta'],weight=ranwgt)

        histos[5,realid].fill((df_realcut['psxptar']),weight=realwgt)
        histos[5,ranid].fill((df_rancut['psxptar']),weight=ranwgt)

        histos[6,realid].fill((df_realcut['psyptar']),weight=realwgt)
        histos[6,ranid].fill((df_rancut['psyptar']),weight=ranwgt)

        histos[7,realid].fill((df_realcut['psytar']),weight=realwgt)
        histos[7,ranid].fill((df_rancut['psytar']),weight=ranwgt)

        histos[8,realid].fill((df_realcut['W']),weight=realwgt)
        histos[8,ranid].fill((df_rancut['W']),weight=ranwgt)

        histos[9,realid].fill((df_realcut['Q2']),weight=realwgt)
        histos[9,ranid].fill((df_rancut['Q2']),weight=ranwgt)

        histos[10,realid].fill((df_realcut['xbj']),weight=realwgt)
        histos[10,ranid].fill((df_rancut['xbj']),weight=ranwgt)

        histos[11,realid].fill((df_realcut['zhad']),weight=realwgt)
        histos[11,ranid].fill((df_rancut['zhad']),weight=ranwgt)

        histos[12,realid].fill((df_realcut['Pt']),weight=realwgt)
        histos[12,ranid].fill((df_rancut['Pt']),weight=ranwgt)

        histos[13,realid].fill((df_realcut['phipq']),weight=realwgt)
        histos[13,ranid].fill((df_rancut['phipq']),weight=ranwgt)

        histos[14,realid].fill((df_realcut_hplus['phipq']),weight=realwgt)
        histos[14,ranid].fill((df_rancut_hplus['phipq']),weight=ranwgt)

        histos[15,realid].fill((df_realcut_hminus['phipq']),weight=realwgt)
        histos[15,ranid].fill((df_rancut_hminus['phipq']),weight=ranwgt)
        print('run,# of real events',run,len(df_realcut['hsdelta']))
    #Subtract randoms
    for i in range(nvar):
        histos[i,subid]=histos[i,realid]+histos[i,ranid]*(-1.0)

    return qtot
