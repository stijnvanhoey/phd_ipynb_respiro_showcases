# -*- coding: utf-8 -*-
"""
@author: stvhoey

Respirometer python implementation
"""

import numpy as np
from scipy.integrate import odeint

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