from pyomo.environ import *
# from .utils import filter_by_phase_to_dict
import os
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

def initialize_DERload(model, DER_at_bus=None, DER_names=None, DER_phases=None, DER_pi=None, DER_P_per_phase=None, DERP_dict=None, alpha=None, MutDER=None, InfeasibleFlag=False):
    
    model.DERAtBus = Set(model.Buses, initialize=DER_at_bus)
    model.DER = Set(initialize = DER_names)
    model.DERPi =  Param(model.DER, initialize=DER_pi, within=Any)
    model.DERP_Phases = Param(model.Phases, model.DER, initialize=DERP_dict, within=Any)
    model.DERPhases = Param(model.DER, initialize=DER_phases, within=Any)
    model.DERP  = Param(model.DER, initialize=DER_P_per_phase, within=Any)
    
    if alpha == None:
        model.DERAlpha = Var(model.DER, bounds=[0,1], within=Reals)
    else:
        model.DERAlpha = Var(model.DER, bounds=[0,1], within=Reals)
        for d in model.DER:
            model.DERAlpha[d].fixed = True
            if InfeasibleFlag:
                if alpha[d] > 0.0:
                    model.DERAlpha[d].unfix()
                else:
                    model.DERAlpha[d] = alpha[d]
            else:
                model.DERAlpha[d] = alpha[d]

    if MutDER is not None:
        for d in MutDER:
            model.DERAlpha[d].unfix()


