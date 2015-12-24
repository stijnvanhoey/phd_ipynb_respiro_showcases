# -*- coding: utf-8 -*-
"""
Exercise file for the Biomath seminar on 16/10/2013
@author: Stijn Van Hoey
"""

#%%-------------------------------------------------------------------
#  Import the necessary packages and functions
#---------------------------------------------------------------------
#General stuff²
import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#pySTAN and related stuff
#------------------------
#sys.path.append("D:\Projecten\Githubs") #ADAPT PATH TO GITHUB DIRECTORY!!
#from pySTAN import *

#%%-------------------------------------------------------------------
#  Model implementation
#---------------------------------------------------------------------
def deriv_works(u, t, Pars, Const, Options): #RespiroModel - derivative
    '''
    Differential equations of the respirometric model
    '''
    #Define the parameters
    mumax = np.float64(Pars[0])
    Y = np.float64(Pars[1])
    Ks = np.float64(Pars[2])
    tau = np.float64(Pars[3])
##    print 'mumax is %f ;Y is %f; Ks is %f and tau is %f' %(mumax,Y,Ks,tau)
    b = np.float64(Const[0])
    kla = np.float64(Const[1])
    SOeq = np.float64(Const[2])
##    print ' de kla is %f en b is %f en SOeq is %f' %(kla,b,SOeq)
    Monod=mumax*(u[1])/(u[1]+Ks)    #Monod Kinetic
    Expo=1.0-np.exp(-t/tau)         #Helpfunction

    dXdt = (Expo*Monod-b)*u[0]                         #Biomassa
    dSsdt = -(1.0/Y)*Expo*Monod*u[0]                   #Substrate
    dOdt = kla*(SOeq-u[2])  -((1-Y)/Y)*Expo*Monod*u[0]   #Oxygen

    return np.array([dXdt,dSsdt,dOdt])

def Respiro(Pars, Init, Init_unc, time, Const, Options=[]):
    '''
    Run the respirometric model
    '''
    
    #Define the constants
    ##################################
    b=Const[0]
    kla=Const[1]
    SOeq=Const[2]
    Constt = np.array([b,kla,SOeq])

    #Define the initial conditions (Constants)Ss0
    #########################################
    Ss0=Init[0]
    #Define the initial conditions (Uncertain) -> X0
    #########################################    
    X0=Init_unc[0]
    yinit = np.array([X0,Ss0,SOeq])  

    #Define the necessary parameters
    ##################################
    mumax = np.float64(Pars[0])
    Y = np.float64(Pars[1])
    Ks = np.float64(Pars[2])
    tau = np.float64(Pars[3])

    #Solve with LSODA scheme
    ##################################
    y,infodic=odeint(deriv_works,yinit,time, full_output=True, printmessg=False, args=(Pars,Constt,Options))

    #Get outputs
    ##################################
    X = y[:,0]
    Ss = y[:,1]
    O = y[:,2]

    OUR_ex=((1-np.exp(-time/tau))*mumax*(1-Y)/Y*Ss/(Ss+Ks)*X)/(24*60)

    return [time, X, Ss, O, OUR_ex, infodic]

#%%-------------------------------------------------------------------
# Initialize model
#---------------------------------------------------------------------

#SET TIME OUTPUTS
Mtime_d = np.arange(0.,0.05,0.0001)
#Mtime_d = np.arange(0.,0.05,0.00001) #smaller timesteps

#SET CONSTANT VALUES
b = 0.62
kla = 369.7334962
S0eq = 8.4  #both in init and constants
#SET INITS
Ss0 = 88.7943  #case specific


#%%-------------------------------------------------------------------
# Specify the uncertain inputs to analyse
#for information chekc the parameter.py file in pySTAN package
#---------------------------------------------------------------------
#X0 is considered as uncertain initial condition
X0=ModPar(r'$X_0$',337.,1012.,675., 'randomUniform')

#SET PARS
mumax=ModPar(r'$\mu_{max}$',3.2,4.8,4.0, 'randomUniform')
Y=ModPar(r'$Y$',0.75,0.83,0.79, 'randomUniform')
Ks=ModPar(r'$K_s$',0.2,0.6,0.41,'randomUniform')
tau=ModPar(r'$\tau$',0.000113,0.000339,2.26e-04,'randomUniform')


#%%-------------------------------------------------------------------
# Set up the other model inputs
#---------------------------------------------------------------------
Pars=[mumax.optguess,Y.optguess,Ks.optguess,tau.optguess] #optguess is ModPar-class-property
Init=[Ss0,S0eq]
Init_unc=np.array([X0.optguess])
Const=[b,kla,S0eq]


#%%-------------------------------------------------------------------
# Modelrun and visualisation
#---------------------------------------------------------------------
#outputmodel = Respiro(Pars,Init,Init_unc,Mtime_d,Const)
#
##plot outputs
#fig,ax = plt.subplots(2,2)
#ax[0,0].plot(outputmodel[0],outputmodel[1],'k',label = 'X')
#ax[0,0].set_xticklabels([])
#ax[0,0].legend(loc=0)
#ax[0,1].plot(outputmodel[0],outputmodel[2],'k',label = 'Ss')
#ax[0,1].set_xticklabels([])
#ax[0,1].legend(loc=0)
#ax[1,0].plot(outputmodel[0],outputmodel[3],'k',label = 'D0')
#ax[1,0].legend(loc=0)
#ax[1,0].set_xlabel('Time')
#ax[1,1].plot(outputmodel[0],outputmodel[4],'k',label = 'OUR')
#ax[1,1].legend(loc=0)
#ax[1,1].set_xlabel('Time')
#plt.draw()

##%%-------------------------------------------------------------------
## Run model with multiple parameter sets
##---------------------------------------------------------------------
#MCruns = 10
#fig2,ax2 = plt.subplots()
#for i in range(MCruns):
#    Init_unc=np.array([X0.optguess])
#    Pars=[mumax.avalue(),Y.avalue(),Ks.avalue(),tau.avalue()]
#    test=Respiro(Pars,Init,Init_unc,Mtime_d,Const)
#    ax2.plot(test[0],test[4])
#plt.draw()
#
#
###%%-------------------------------------------------------------------
### Sensitivity analysis
###---------------------------------------------------------------------

#%%-------------------------------------------------------------------
# Morris
#---------------------------------------------------------------------
#tuning factors: intervals, Delta, baseruns


#make your unertain factor set (parameters and initial condition)
Xi = [X0, mumax, Y, Ks, tau] #list of modpar instances is input

#initialize the sensitivity class object
sens = MorrisScreening(Xi, ModelType = 'external')

#prepare your parameter set to calculate
t1,t2 = sens.Optimized_Groups(nbaseruns=100, intervals = 4, noptimized=50, Delta = 'default')

# ONLY FOR MORRIS - Check the quality of the selected trajects                                                                           
#sens.Optimized_diagnostic(width=0.15)

#RUN THE MODEL for each required parameter set (parset2run)
#Interest is DO values
DO_out = np.empty((sens.totalnumberruns,Mtime_d.size))
OUR_out = np.empty((sens.totalnumberruns,Mtime_d.size))

for i in range(sens.totalnumberruns):
    Init_unc=np.array([sens.parset2run[i,0]])
    Pars=sens.parset2run[i,1:]
    modelout=Respiro(Pars,Init,Init_unc,Mtime_d,Const)    
    DO_out[i,:]=modelout[3]
    OUR_out[i,:]=modelout[4]

#Use the outputs to get the sensitivity indices
sens.Morris_Measure_Groups(DO_out)  #OUR_out

#PLOT AND OTHER OUTPUTS
#sens.plotmu(ec='grey',fc='grey', outputid = 10)
sens.plotmustar(ec='grey',fc='grey', outputid = 100)
#sens.plotmustarsigma(outputid=10, zoomperc='none', loc=2)

#%%-------------------------------------------------------------------
# GOAT
#---------------------------------------------------------------------
#tuning factors: samplemethod, numerical approach, CAS/CTRS/PE

#make your unertain factor set (parameters and initial condition)
Xi = [X0, mumax, Y, Ks, tau] #list of modpar instances is input

#initialize the sensitivity class object
sens2 = GlobalOATSensitivity(Xi)

#prepare your parameter set to calculate
sens2.PrepareSample(100, 0.01, samplemethod= 'lh', numerical_approach= 'single')

# ONLY FOR MORRIS - Check the quality of the selected trajects                                                                           
#sens.Optimized_diagnostic(width=0.15)

#RUN THE MODEL for each required parameter set (parset2run)
#Interest is DO values
DO_out = np.empty((sens2.totalnumberruns,Mtime_d.size))
OUR_out = np.empty((sens2.totalnumberruns,Mtime_d.size))
for i in range(sens2.totalnumberruns):
    Init_unc=np.array([sens2.parset2run[i,0]])
    Pars=sens2.parset2run[i,1:]
    modelout=Respiro(Pars,Init,Init_unc,Mtime_d,Const)    
    DO_out[i,:]=modelout[3]
    OUR_out[i,:]=modelout[4]

#Think about which kind of outputs you will use for the sensitivity (timesteps, min,max,...)
DO_summ = np.vstack((DO_out.min(axis=1), DO_out.max(axis=1), DO_out.mean(axis=1))).transpose()

#Use the outputs to get the sensitivity indices
#sens2.Calc_sensitivity(DO_out.min(axis=1))  #OUR_out
sens2.Calc_sensitivity(DO_summ)  #OUR_out

##PLOT AND OTHER OUTPUTS
sens2.plotsens(indice='PE', ec='grey',fc='grey', outputid=2)
sens2.plot_rankmatrix(['min','max','mean'])


#%%-------------------------------------------------------------------
# Regression
#---------------------------------------------------------------------
#tuning factors: number of samples, lh/sobol, rank or not

#make your unertain factor set (parameters and initial condition)
Xi = [X0, mumax, Y, Ks, tau] #list of modpar instances is input

#initialize the sensitivity class object
sens = SRCSensitivity(Xi)

#prepare your parameter set to calculate
sens.PrepareSample(100, samplemethod='Sobol')

# ONLY FOR MORRIS - Check the quality of the selected trajects                                                                           
#sens.Optimized_diagnostic(width=0.15)

#RUN THE MODEL for each required parameter set (parset2run)
#Interest is DO values
DO_out = np.empty((sens.totalnumberruns,Mtime_d.size))
OUR_out = np.empty((sens.totalnumberruns,Mtime_d.size))
for i in range(sens.totalnumberruns):
    Init_unc=np.array([sens.parset2run[i,0]])
    Pars=sens.parset2run[i,1:]
    modelout=Respiro(Pars,Init,Init_unc,Mtime_d,Const)    
    DO_out[i,:]=modelout[3]
    OUR_out[i,:]=modelout[4]

#Think about which kind of outputs you will use for the sensitivity (timesteps, min,max,...)
DO_summ = np.vstack((DO_out.min(axis=1), DO_out[:,5:].max(axis=1), DO_out.mean(axis=1))).transpose()

#Use the outputs to get the sensitivity indices
#sens.Calc_sensitivity(DO_out.min(axis=1))  #OUR_out
sens.Calc_SRC(DO_summ, rankbased=True) #OUR_out

##PLOT AND OTHER OUTPUTS
sens.quickscatter(DO_summ[:,0])
sens.plot_tornado(outputid = 2,gridbins=3, midwidth=0.2, 
                             setequal=True, plotnumb=True, parfontsize=12, 
                             bandwidth=0.75)
sens.plot_SRC(outputid='all', width=0.3, sortit = True, ec='grey',fc='grey')


#%%-------------------------------------------------------------------
# Sobol
#---------------------------------------------------------------------
#tuning factors: repl, baseruns,...

#make your unertain factor set (parameters and initial condition)
Xi = [X0, mumax, Y, Ks, tau] #list of modpar instances is input

#initialize the sensitivity class object
sens = SobolVariance(Xi, ModelType = 'external')

#prepare your parameter set to calculate
sens.SobolVariancePre(500)

# ONLY FOR MORRIS - Check the quality of the selected trajects                                                                           
#sens.Optimized_diagnostic(width=0.15)

#RUN THE MODEL for each required parameter set (parset2run)
#Interest is DO values
DO_out = np.empty((sens.totalnumberruns,Mtime_d.size))
OUR_out = np.empty((sens.totalnumberruns,Mtime_d.size))
for i in range(sens.totalnumberruns):
    Init_unc=np.array([sens.parset2run[i,0]])
    Pars=sens.parset2run[i,1:]
    modelout=Respiro(Pars,Init,Init_unc,Mtime_d,Const)    
    DO_out[i,:]=modelout[3]
    OUR_out[i,:]=modelout[4]

#Think about which kind of outputs you will use for the sensitivity (timesteps, min,max,...)
DO_summ = np.vstack((DO_out.min(axis=1), DO_out[:,5:].max(axis=1), DO_out.mean(axis=1))).transpose()

#Use the outputs to get the sensitivity indices
sens.SobolVariancePost(DO_summ[:,2])


###PLOT AND OTHER OUTPUTS
sens.plotSi(ec='grey',fc='grey')
sens.plotSTi(ec='grey',fc='grey')
sens.plotSTij()
Si_evol, STi_evol = sens.sens_evolution(color='k')







