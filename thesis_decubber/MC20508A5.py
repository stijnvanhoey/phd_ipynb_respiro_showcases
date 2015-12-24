# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:08:44 2014

@author: Stijn
"""


import sys
import os
import numpy  
import pandas as pd
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator, LinearLocator, NullLocator
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.pyplot as plt 
sys.path.append('C:\Users\Stijn\Documents\Python Scripts')  
from biointense import *  

#==============================================================================
# DATA INLEZEN
#==============================================================================
ids = "0508A5" #HIGH CASE
#ids = "0508A1" #LOW CASE
data = pd.read_csv(ids+"_pieken.csv")
dataprofile1 = ode_measurements(data["DO"])
time_col = 'NewTime'
lasttime = np.ceil(data[time_col].values[-1])
data = data.set_index(time_col)

#==============================================================================
# AANTAL RUNS EN THRESHOLD SELECTEREN
#==============================================================================
nruns = 1000
treshold = len(data)*(0.1*0.1) #treshold voor WSSE gebaseerd op zie M&M

##==============================================================================
## HIGH CASE WAARDEN
##==============================================================================
ks0 = 1.46
tau0 = 67.8
mumax0 = 3.84
S0 = 69
DOi = 8.907
Xi = 835
Yi = 0.700702
doev = 8.907
klai = 0.0065872140000000001
bh = 0.24

#==============================================================================
# LOW CASE WAARDEN
#==============================================================================

#ks0 = 4.61
#tau0 = 126.7
#mumax0 = 7.37
#S0 = 17.25
#DOi = 8.956
#Xi = 835
#Yi = 0.72045
#doev = 8.956
#klai = 0.0065872140000000001
#bh = 0.24



#==============================================================================
# MC FILTERING
#==============================================================================


#bereik definiëren waaruit parameters gesampled worden 
mumax_low =3.
mumax_high = 5.
ks_low = 0.1
ks_high = 5.
tau_low = 67.78####HIGH CASE
tau_high = 67.82
#tau_low = 126.5
#tau_high =  126.7
#LOW CASE


#model definiëren
Modelname = 'Basicrespiromodel'
Parameters = {'tau':60, 'mumax':6, 'ks':20}
System = {'dSS':'-(1-exp(-t/tau))*((mumax/(3600*24))*XX/('+str(Yi)+')*SS/((SS+ks)))',
          'dDO':'('+str(klai)+'*('+ str(DOi)+'-DO)-(1-exp(-t/tau))*(mumax/(3600*24))*XX/'+str(Yi)+'*(1-'+str(Yi)+')*SS/((SS+ks)))',
          'dXX' :'(1-exp(-t/tau))*(mumax*XX/(3600*24))*SS/(SS+ks)-(0.24/(24*3600))*XX'
          }    

#manueel

M1 = odegenerator(System, Parameters, Modelname = Modelname)
M1.set_measured_states(['DO'])
M1.set_initial_conditions({'S':S0, 'D0': DOi, 'X': Xi})
M1.set_time({'start':0,'end':lasttime,'nsteps':1000})
#

#
##parameters selecteren
mumax=ModPar("mumax", mumax_low, mumax_high, 'randomUniform')
ks=ModPar("ks", ks_low, ks_high, 'randomUniform')
tau=ModPar("tau", tau_low, tau_high, 'randomUniform')


##prepare runs
lh_mumax = mumax.LatinH(nruns)
lh_ks =ks.LatinH(nruns)
lh_tau =tau.LatinH(nruns)
mc_wsse = []

wsse_good = []
wsse_bad = []    
lh_ks_good = []
lh_ks_bad = []
lh_mumax_good = []
lh_mumax_bad = []
lh_tau_good = []
lh_tau_bad = []

#
def calc_sse(mod, meas):
    """
    """
    return ((mod-meas)**2).sum()
Fitprofile1 = ode_optimizer(M1,dataprofile1)
##nruns optimalisaties laten lopen
for run in range(nruns):
    print run
    
    Fitprofile1.set_fitting_parameters({'ks':lh_ks[run],'tau':lh_tau[run],'mumax':lh_mumax[run]})
    mc_wsse.append(Fitprofile1.get_WSSE()[0][0])
    
#    wsse_acid = calc_sse(Fitprofile1.ModMeas["Modelled"]["acid"], 
#                         Fitprofile1.ModMeas["Measured"]["acid"])
    if Fitprofile1.get_WSSE()[0][0] > treshold:
        wsse_bad.append(Fitprofile1.get_WSSE()[0][0]) 
        lh_ks_bad.append(lh_ks[run])
        lh_tau_bad.append(lh_tau[run])
        lh_mumax_bad.append(lh_mumax[run])
    else:
        wsse_good.append(Fitprofile1.get_WSSE()[0][0]) 
        lh_ks_good.append(lh_ks[run])
        lh_tau_good.append(lh_tau[run])
        lh_mumax_good.append(lh_mumax[run]) 
#
#                
colormapgood = cm.summer #colormap van geel naar groen
### punten die 'good' zijn (=wsse lager dan treshold) worden geplot volgens gradient, beste = groen
### punten die 'bad' zijn (=wsse hoger dan treshold) allemaal in rood geplot
#==============================================================================
# SIMULATIES WEGSCHRIJVEN NAAR FILE HIGHCASE
#==============================================================================

##==============================================================================
## HIGH
##==============================================================================
#
np.savetxt('HIGH_wsse_bad.txt', wsse_bad)
np.savetxt('HIGH_ks_bad.txt', lh_ks_bad )
np.savetxt('HIGH_tau_bad.txt', lh_tau_bad )
np.savetxt('HIGH_mumax_bad.txt', lh_mumax_bad)
np.savetxt('HIGH_wsse_good.txt', wsse_good)
np.savetxt('HIGH_ks_good.txt', lh_ks_good)
np.savetxt('HIGH_tau_good.txt', lh_tau_good)
np.savetxt('HIGH_mumax_good.txt', lh_mumax_good)

#==============================================================================
# LOW
#==============================================================================

#np.savetxt('LOW_wsse_bad.txt', wsse_bad)
#np.savetxt('LOW_ks_bad.txt', lh_ks_bad )
#np.savetxt('LOW_tau_bad.txt', lh_tau_bad )
#np.savetxt('LOW_mumax_bad.txt', lh_mumax_bad)
#np.savetxt('LOW_wsse_good.txt', wsse_good)
#np.savetxt('LOW_ks_good.txt', lh_ks_good)
#np.savetxt('LOW_tau_good.txt', lh_tau_good)
#np.savetxt('LOW_mumax_good.txt', lh_mumax_good)



#==============================================================================
# THESISPLOT: DATA INLADEN ZIE APARTE FILE VOOR ZW WIT PLOT
#==============================================================================

#plt.scatter(lh_ks_good, lh_mumax_good, c = wsse_good, marker='o',  edgecolor='none', cmap = colormapgood)
#plt.scatter(lh_ks_bad, lh_mumax_bad, marker ='o', edgecolor = 'none', facecolor ='red')
#plt.grid(linestyle = 'dashed', color = '0.75',linewidth = 1.)
#plt.axis([ks_low, ks_high, mumax_low, mumax_high])
#plt.xlabel(r'K$_S$ (mg/L)',fontsize=16)
#plt.ylabel(r'$\mu_{max}$ (d$^{-1}$)',fontsize=16)
#plt.show()


#plt.scatter(lh_mumax_good, wsse_good)

#



print 'done'