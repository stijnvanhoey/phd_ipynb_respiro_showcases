# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:20:12 2014

@author: stvhoey

Fitten procedure Stijn Decubber
"""

import sys
import os
import numpy  
import matplotlib.pyplot as plt  
import pandas as pd
import pprint
sys.path.append('/media/DATA/Projecten/2013_14_Thesis')  
from biointense import *  
plt.close("all")

from scipy import optimize


#%%============================================================================
# DATA INLEZEN
#==============================================================================

#INFO IVM PIEKEN
#------------------
cols = ["id", "S", "DO","X", "kla", "Y", "ratio", "Ycalc" ]
piekdata = pd.read_table("alles2.lvm", names = cols)
piekdata = piekdata.set_index("id")

#alle namen
allnames = [nm[:-1] for nm in piekdata.index.tolist()]
piekdata["shortname"] = allnames
#unieke namen
names = []    
[names.append(i) for i in allnames if not i in names] 

# TIJDSREEKSEN
#------------------
datapath = "/media/DATA/Projecten/2013_14_Thesis/StijnDecubber/pieken"

#%%============================================================================
# OPTIMALISATIES RUNNEN 1 EXPERIMENT MET 5 PIEKEN
#==============================================================================

#DOELFUNCTIE
def all_peaks_optimize(parameters, name):    
    """
    
    Parameters
    -----------
    name :  char
        id-name of the experiment
    parameters : list
        List with parameter values => [ks rmax]
    """
    
    
    #select the relevant peaks of the experiment
    exp_peaks = piekdata.groupby("shortname").get_group(name)
    
    SSE = 0

    #voor elke piek een model opstellen en SSE optellen
    for index, row in exp_peaks.iterrows():
        #get names of the experiments
        idi = index
        Si = row["S"]
        DOi= row["DO"]
        Xi = row["X"]
        klai = row["kla"]
        Yi = row["Y"]
        ratioi = row["ratio"]
        
        initcond = {'SS':Si, 'DO':DOi, 'XX':Xi}
        
        Modelname = 'Basicrespiromodel'
        #hier de nieuwe pars geven
        Parameters = {'tau':60, 'mumax':parameters[1], 'ks':parameters[0]}
        System = {'dSS':'-(1-exp(-t/tau))*((mumax/(3600*24))*XX/('+str(Yi)+')*SS/((SS+ks)))',
                  'dDO':'('+str(klai)+'*('+ str(DOi)+'-DO)-(1-exp(-t/tau))*(mumax/(3600*24))*XX/'+str(Yi)+'*(1-'+str(Yi)+')*SS/((SS+ks)))',
                  'dXX' :'(1-exp(-t/tau))*(mumax*XX/(3600*24))*SS/(SS+ks)-(0.24/(24*3600))*XX'
                  }                                       
        M1 = odegenerator(System, Parameters,  Modelname = Modelname)
        M1.set_measured_states(['DO'])
        M1.set_initial_conditions(initcond)
        
        #-------------------------------------------------
        #HIER NOG AAN TE PASSEN MET INLEZEN GOEIE FILES
        #-------------------------------------------------
        data = pd.read_csv(os.path.join(datapath,index+"_pieken.csv"))
        #-------------------------------------------------        

        time_col = 'NewTime'
        lasttime = np.ceil(data[time_col].values[-1])
        data = data.set_index(time_col)
        M1.set_time({'start':0,'end':lasttime,'nsteps':20})
        
        dataprofile1 = ode_measurements(data[["DO"]])                         
        Fitprofile1 = ode_optimizer(M1,dataprofile1)
#        initial_parset = {'tau':80,  'mumax':1.0, 'ks' :30}
#        #the bugfix: !!!!
#        initial_parset = collections.OrderedDict(sorted(initial_parset.items(), 
#                                                        key=lambda t: t[0]))#effectieve volgorde is: ks rmax tau
        
        SSEp = Fitprofile1.get_WSSE()[0]
        
        
        SSE +=SSEp
    print "-"*40
    print "SSE of experiment ", name, " is ", SSE
    print "-"*40
    print "Current parameter values are \n", 
    print "ks: ", Fitprofile1.get_all_parameters()['ks']
    print "mumax: ", Fitprofile1.get_all_parameters()['mumax']
    print "-"*40
 
    return SSE

#%%=============================================================================
# def voor confidence intervals
#============================================================================== 

def get_parameter_confidence(FIM, datacount, parvalues, alpha = 0.95):
    '''Calculate confidence intervals for all parameters
    
    Parameters
    -----------
    alpha: float
        confidence level of a two-sided student t-distribution (Do not divide 
        the required alpha value by 2). For example for a confidence level of 
        0.95, the lower and upper values of the interval are calculated at 0.025 
        and 0.975.
    
    Returns
    --------
    CI: pandas DataFrame
        Contains for each parameter the value of the variable, lower and 
        upper value of the interval, the delta value which represents half 
        the interval and the relative uncertainty in percent.
    
    '''
    ECM = np.matrix(FIM).I

    CI = np.zeros([ECM.shape[1],8]) 
    n_p = sum(datacount)-len(parvalues)
    
    CI[:,0] = parvalues
    for i,variance in enumerate(np.array(ECM.diagonal())[0,:]):
        #TODO check whether sum or median or... should be used 
        CI[i,1:3] =  stats.t.interval(alpha, n_p, loc=parvalues[i],scale=np.sqrt(variance))
        #print stats.t.interval(alpha,self._data.Data.count()-len(self.Parameters),scale=np.sqrt(var))[1][0]
        CI[i,3] = stats.t.interval(alpha, n_p, scale=np.sqrt(variance))[1]
    CI[:,4] = abs(CI[:,3]/parvalues)*100
    CI[:,5] = parvalues/np.sqrt(ECM.diagonal())
    CI[:,6] = stats.t.interval(alpha, n_p)[1]
    for i in np.arange(ECM.shape[1]):
        CI[i,7] = 1 if CI[i,5]>=CI[i,6] else 0
    
    if (CI[:,7]==0).any():
        print 'Some of the parameters show a non significant t_value, which\
        suggests that the confidence intervals of that particular parameter\
        include zero and such a situation means that the parameter could be\
        statistically dropped from the model. However it should be noted that\
        sometimes occurs in multi-parameter models because of high correlation\
        between the parameters.'
    else:
        print 'T_values seem ok, all parameters can be regarded as reliable.'
        
    CI = pd.DataFrame(CI,columns=['value','lower','upper','delta','percent','t_value','t_reference','significant'], 
                      index=['ks','mumax'])
    
    parameter_confidence = CI
    
    return CI

def run_all_for_plotting(parameters, name, tauvalue): 
    """
    
    Parameters
    -----------
    name :  char
        id-name of the experiment
    parameters : list
        List with parameter values => [ks rmax]
    """
    
    
    #select the relevant peaks of the experiment
    exp_peaks = piekdata.groupby("shortname").get_group(name)
    
    modelresults = {}
    
    FIM = {}
    FIMsum = np.zeros((2,2))

    #voor elke piek een model opstellen en SSE optellen
    for index, row in exp_peaks.iterrows():
        #get names of the experiments
        idi = index
        Si = row["S"]
        DOi= row["DO"]
        Xi = row["X"]
        klai = row["kla"]
        Yi = row["Y"]
        ratioi = row["ratio"]
        
        initcond = {'SS':Si, 'DO':DOi, 'XX':Xi}
        
        Modelname = 'Basicrespiromodel'
        #hier de nieuwe pars geven
        Parameters = {'mumax':parameters[1], 'ks':parameters[0]}
        System = {'dSS':'-(1-exp(-t/'+str(tauvalue)+'))*((mumax/(3600*24))*XX/('+str(Yi)+')*SS/((SS+ks)))',
                  'dDO':'('+str(klai)+'*('+ str(DOi)+'-DO)-(1-exp(-t/'+str(tauvalue)+'))*(mumax/(3600*24))*XX/'+str(Yi)+'*(1-'+str(Yi)+')*SS/((SS+ks)))',
                  'dXX' :'(1-exp(-t/'+str(tauvalue)+'))*(mumax*XX/(3600*24))*SS/(SS+ks)-(0.24/(24*3600))*XX'
                  }                                       
        M1 = odegenerator(System, Parameters,  Modelname = Modelname)
        M1.set_measured_states(['DO'])
        M1.set_initial_conditions(initcond)
        
        #-------------------------------------------------
        #HIER NOG AAN TE PASSEN MET INLEZEN GOEIE FILES
        #-------------------------------------------------
        data = pd.read_csv(os.path.join(datapath,index+"_pieken.csv"))
        #-------------------------------------------------        

        time_col = 'NewTime'
        lasttime = np.ceil(data[time_col].values[-1])
        data = data.set_index(time_col)
        M1.set_time({'start':0,'end':lasttime,'nsteps':20})
        
        dataprofile1 = ode_measurements(data[["DO"]])                         
        Fitprofile1 = ode_optimizer(M1,dataprofile1)
        modelresults[idi] = Fitprofile1.ModMeas
        FIMobject = ode_FIM(Fitprofile1)#, sensmethod = 'numerical')
        
        FIM[idi] = FIMobject.FIM
        FIMsum += FIMobject.FIM
    
    FIM["sum"] = FIMsum
    return modelresults, FIM

#%%============================================================================
#TAU VAN DE GROOTSTE S-waarde PROFIEL  => COMBINED FOR mumax/ks
#==============================================================================
#ophalen tau-waarden
taupath = "/media/DATA/Dropbox/thesis_Stijn_DC/modelling/resultaten_alle_optimalisaties_eraser"
#first find largest peaks for all experiments
highest_peaks = []
for name in names:
    exp_peaks = piekdata.groupby("shortname").get_group(name)
    highest_peaks.append(exp_peaks['S'].argmax())

#get these tau values from file
f = open(os.path.join(taupath, "optimalpars_alles"))    
allines = f.readlines()
f.close()    
tauvalues = {}
for j,line in enumerate(allines):
    if line[:-1] in highest_peaks:
        tauvalues[line[:-2]] = float(allines[j+3][:-1])
        
        

#%%============================================================================
# SCIPY OPTIMALISATIE UITVOEREN
#==============================================================================
        
#maxiters = 10
#method = 'BFGS'
#optimal_pars = {}
#for name in names:
#for name in ['0508A']:
#    print "*"*40
#    print "working on experiment ", name, "..."
#    print "*"*40
#    initial_parset = [30., 1.0]     #{'ks' :30, 'mumax':1.0}                           
#    res = optimize.minimize(all_peaks_optimize, initial_parset, 
#                                      args=(name,), method= method, jac = False,
#                                      options = {"maxiter": maxiters, "eps": 0.01})
#    print res.message  
#    optimal_pars[name] = res.x
#    print "...done!"
#    print "*"*40  
    

# f = open("2014_optimalpars_allpeaks", "w")
# for j, experiment in enumerate(optpars.keys()):
#     f.write(experiment + "\n")
#     for key, value in parset.items():
#         f.write(str(value)+"\n")
#     f.write(str(wsses[j])+"\n")
# f.close()

#%%=============================================================================
# Inlezen optimale pars na optimalisatie eraser 5pieken teglijk
#============================================================================== 
from itertools import *

optimal_pars = {}
optpath = "/media/DATA/Dropbox/thesis_Stijn_DC/modelling/"

f = open(os.path.join(optpath, "2014_optimalpars_allpeaks_final"))    
allines = f.readlines()   
for i in imap(lambda x:3*x, xrange(len(allines)/3)):
    optimal_pars[allines[i][:-1]] = [float(allines[i+1][:-1]), float(allines[i+2][:-1])]
f.close() 

#0508A
#1.74650157158
#3.84155105518



    
#%%=============================================================================
# DOORREKENEN MODELS MET OPTIMALE WAARDEN EN PLOTTEN
#============================================================================== 

##  PLOTVERSION 1
#============================================================================== 
#for name, pars in optimal_pars.iteritems():
#    
#    modelresults = run_all_for_plotting(pars, name)
#    
#    npieks = len(modelresults)
#    #run each model met laatste info en maak figuur:
#    figure, ax = plt.subplots(npieks, 1, sharex = True, figsize = (8,10)) #, 
#    figure.subplots_adjust(hspace = 0.03)
#    for j,pieknaam in enumerate(modelresults.keys()):
#        ids = int(pieknaam[-1])-1
#        ax[ids].plot(modelresults[pieknaam]["Measured"], color='0.', 
#                 label='Measured DO (mg/L)', linewidth = 2.0)
#        ax[ids].plot(modelresults[pieknaam]["Modelled"]["DO"],'--', 
#                color='0.5', label = 'Modelled DO (mg/L)')
#        
#        ax[ids].text(0.85, 0.8, pieknaam, transform=ax[ids].transAxes)
#        if ids % 2 != 0:
#            ax[ids].yaxis.tick_right()
#            ax[ids].yaxis.set_label_position("right")
#        ax[ids].set_ylabel('DO (mg/L)')   
#        
#        if ax[ids].is_last_row():
#            ax[ids].set_xlabel('Time (s)')
#
#        if ax[ids].is_first_row():
#            ax[ids].legend(loc = 'lower center',bbox_to_anchor = (0.5, 1.05),
#                             bbox_transform= ax[ids].transAxes, ncol = 2)            


##  PLOTVERSION 2
#============================================================================== 
from matplotlib.ticker import LinearLocator


optimal_pars

parconfs = {}
for name, pars in optimal_pars.iteritems():
    
    modelresults, FIM = run_all_for_plotting(pars, name, tauvalues[name])
    npieks = len(modelresults)
    print npieks, ' peaks ', name

    
    #tijd van de experimenten veranderen
    if npieks > 1:
        if not name == "0401A":
            for i in range(2, npieks+1):
                print name + str(i)
                modelresults[name + str(i)].index = modelresults[name + str(i)].index + modelresults[name + str(i-1)].index[-1]
        else:
            modelresults[name + str(3)].index = modelresults[name + str(3)].index + modelresults[name + str(1)].index[-1]            
        
    datasize = 0
    for key, value in modelresults.iteritems():
        datasize += value.index.size
        
    #FIM TO CONFIDENCE
    parconfs[name] = get_parameter_confidence(np.matrix(FIM["sum"]), datasize, pars, alpha = 0.95)
         

#    if npieks == 1:
#        #run each model met laatste info en maak figuur:
#        figure, ax = plt.subplots(1, 1, figsize = (12,6)) #, 
#        for j,pieknaam in enumerate(modelresults.keys()):
#            ax.plot(modelresults[pieknaam].index, modelresults[pieknaam]["Measured"], color='0.', 
#                     label='Measured DO (mg/L)', linewidth = 2.0)
#            ax.plot(modelresults[pieknaam].index, modelresults[pieknaam]["Modelled"]["DO"],'--', 
#                    color='0.5', label = 'Modelled DO (mg/L)')
#            
#            ax.text(0.5, 1.05, pieknaam, transform=ax.transAxes,
#                        verticalalignment='center', horizontalalignment='center')
#            #ax[ids].set_xticks([])
#        
#        ax.set_ylabel('DO (mg/L)') 
#        plt.savefig("2014_" + name + "_5peaksfit.png")
#        
#    else:
#        #run each model met laatste info en maak figuur:
#        figure, ax = plt.subplots(1, npieks,sharey = True, figsize = (12,6)) #, 
#        figure.subplots_adjust(wspace = 0.0001)                 
#
#        for j,pieknaam in enumerate(modelresults.keys()):
#            ids = int(pieknaam[-1])-1
#            print ids          
#            
#            #EXCEPTION FOR ONE OF THE EXPERIMENTS
#            if name == "0401A" and ids == 2:
#                ids = 1
#    
#            ax[ids].plot(modelresults[pieknaam].index, modelresults[pieknaam]["Measured"], color='0.', 
#                     label='Measured DO (mg/L)', linewidth = 2.0)
#            ax[ids].plot(modelresults[pieknaam].index, modelresults[pieknaam]["Modelled"]["DO"],'--', 
#                    color='0.5', label = 'Modelled DO (mg/L)')
#            
#            ax[ids].text(0.5, 1.05, pieknaam, transform=ax[ids].transAxes,
#                        verticalalignment='center', horizontalalignment='center')
#            
#            #ticks van de xas goedzetten
#            if ids == npieks-1:
#                numtick = 2
#            else:
#                numtick = 1
#                
#            if name == "0401A" and ids == 1:
#                numtick = 2
#                
#            majorlocator = LinearLocator(numticks=numtick)
#            ax[ids].xaxis.set_major_locator(majorlocator)
#            
#
#        
#        #common xlabel
#        figure.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
#                       
#            
#        ax[0].set_ylabel('DO (mg/L)') 
#        plt.savefig("2014_" + name + "_5peaksfit.png")
    
#        if ax[ids] == 3:
#            ax[ids].legend(loc = 'lower center',bbox_to_anchor = (0.5, 1.05),
#                             bbox_transform= ax[ids].transAxes, ncol = 2)   



      
      
#%%============================================================================
# QUICK AND DIRTY MC
#==============================================================================       

#nruns = 1000
#name = '0508A'
#
##grenzen
#mumax_low =3.
#mumax_high = 5.
#ks_low = 0.1
#ks_high = 5.
#
##MC-values
#mumax=ModPar("mumax", mumax_low, mumax_high, 'randomUniform')
#ks=ModPar("ks", ks_low, ks_high, 'randomUniform')
#
#lh_mumax = mumax.LatinH(nruns)
#lh_ks =ks.LatinH(nruns)
#
#lh_wsse = []
#for run in range(nruns):
#    print run
#    
#    SSE = all_peaks_optimize([lh_ks[run], lh_mumax[run]], name)
#    lh_wsse.append(SSE)
#
##SELECT BEHAV - TODO
#colormapgood = cm.YlOrRd
#fig, ax1 = plt.subplots()
#ax1.scatter(lh_ks, lh_mumax, c = lh_wsse, marker='o',  edgecolor='none', cmap = colormapgood)
##ax1.scatter(lh_ks_good, lh_mumax_good, c = wsse_good, marker='o',  edgecolor='none', cmap = colormapgood)
#ax1.grid(linestyle = 'dashed', color = '0.75',linewidth = 1.)
#ax1.axis([ks_low, ks_high, mumax_low, mumax_high])
#ax1.set_xlabel(r'K$_S$ (mg/L)',fontsize=16)
#ax1.set_ylabel(r'$\mu_{max}$ (d$^{-1}$)',fontsize=16)
#
#
#sys.path.append('/media/DATA/Githubs/pySTAN')      
#from plot_functions_rev import Scatter_hist_withOF
#
##------------------------------------------------------------------------------
##SELECT BEHAVIOURAL
#Limit_for_good = 100
#colormap2use = cm.RdYlGn_r
##http://www.loria.fr/~rougier/teaching/matplotlib/
##------------------------------------------------------------------------------
#
#lh_wsse = np.array(lh_wsse)
#id_behav = lh_wsse < Limit_for_good
#wsse_behav = lh_wsse[id_behav]
#mumax_behav = lh_mumax[id_behav]
#ks_behav = lh_ks[id_behav]
#
#fig,axScatter,axHistx,axHisty,sc1 = Scatter_hist_withOF(ks_behav, mumax_behav, 
#                                                        data1b=lh_ks, 
#                                                        data2b=lh_mumax, 
#                                                        xbinwidth = 0.1, 
#                                                        ybinwidth=0.1, 
#                                                        SSE=wsse_behav, 
#                                                        SSEb=lh_wsse, 
#                                                        vmin=100, 
#                                                        vmax=110, 
#                                                        colormaps = colormap2use, 
#                                                        cleanstyle = True, 
#                                                        roodlichter=1.)
#
#axScatter.set_xlabel(r'k$_s$',fontsize=16)
#axScatter.set_ylabel(r'r$_{max}$',fontsize=16)
#cbar = fig.colorbar(sc1, ax=axScatter, cmap=colormap2use, 
#                    orientation='vertical', 
#                    ticks=[Limit_for_good,Limit_for_good+Limit_for_good*1.5],shrink=1.)                        
#cbar.ax.set_yticklabels(['<'+str(Limit_for_good),'> '+str(Limit_for_good+Limit_for_good*1.5)])
#



#==============================================================================
# #==============================================================================
# #     PARAMETERS EN FIM WEGSCHRIJVEN
# #==============================================================================
#     f = open("optimalpars_alles", "w")
#     for j, parset in enumerate(optpars):
#         f.write(str(ids[j])+"\n")
#         for key, value in parset.items():
#             f.write(str(value)+"\n")
#         f.write(str(wsses[j])+"\n")
#     f.close()
#     
#   
# 
# #==============================================================================
# #     ############################# PLOT VOOR IN THESIS
# #==============================================================================
#     figure1, ax1 = plt.subplots()
#     ax1.plot(Fitprofile1.ModMeas["Measured"], color='0.',label='Measured DO (mg/L)', linewidth = 2.0)
#     ax1.plot(Fitprofile1.ModMeas["Modelled"]["DO"],'--', color='0.5', label = 'Modelled DO (mg/L)')
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('DO (mg/L)')
#     ax1.legend(loc = 'lower right')
#==============================================================================
#    



    
    
    
    
    


                                

