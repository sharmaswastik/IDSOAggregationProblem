import logging

import pandas as pd
import numpy as np
import re
from IDSO import DER_at_bus, DER_bids_at_bus, DER_offers_at_bus, DER_names, DER_bids_names, DER_offers_names, DER_phases, DER_bids_phases, DER_offers_phases, DER_pi, DER_pi_bids, DER_pi_offers, DER_P_per_phase, DER_P_per_phase_bids, DER_P_per_phase_offers, DERP_dict, DERP_offers_dict, DERP_bids_dict, MaxP

from .model import (create_model, initialize_buses,
                initialize_time_periods, initialize_model, thermal_limits, Suffix
                    )
from .network import (initialize_network, derive_network, calculate_network_parameters \
    , create_incidence_matrix)

from .generators import (initialize_generators, maximum_minimum_activepower_output_generators, \
    maximum_minimum_reactivepower_output_generators) 

from .spot_load import (initialize_spotload)
from .DER_load import (initialize_DERload)

from .constraints import (activepower_balance_constraint, reactivepower_balance_constraint, \
    voltage_constraint, objective_function, constraint_net_activepower, constraint_net_reactivepower, constraint_slack_bus, thermal_limit_constraint, substation_power_limit_constraint, mutual_case_constraint, constraint_active_scheduled_interchange, constraint_reactive_scheduled_interchange)

from .global_params import update_func

from .capacitor import (initialize_cap, maximum_output_capacitors, switching_cap_func)

from .utils import (cap_values_per_phase)

from solver import solve_model, PSSTResults

logger = logging.getLogger(__file__)

def build_model(case=None,
                substation_df=None,
                generator_df=None,
                Pload_df=None,
                Qload_df=None,
                branch_df=None,
                bus_df=None,
                config=None,
                SpotLoads=None,
                CapData=None,
                thermal_df=None,
                Inp_thermal=None,
                alpha=None,
                results_cap=None,
                OnlyBids=False,
                OnlyOffers=False,
                BothBidsOffers=False,
                NaiveCase=False,
                FinalCase=False,
                MutualCase=False,
                MutDER=None,
                OnlyBids_MC=False,
                MCData=None,
                SourceActivePower=None,
                SourceReactivePower=None,
                InfeasibleFlag=False):
    
    if config is None:
        config = dict()
    
    generator_df = generator_df #or pd.merge(case.gen, case.gencost, left_index=True, right_index=True)
    Pload_df = Pload_df #or case.load
    Qload_df = Qload_df #or case.load
    branch_df = branch_df #or case.branch
    bus_df = bus_df #or case.bus

    config = config

    branch_df.index = np.arange(1,len(branch_df)+1)
    bus_df.index = np.arange(1,len(bus_df)+1)
    # Pload_df.index = np.arange(1,len(Pload_df)+1)
    # Qload_df.index = np.arange(1,len(Qload_df)+1)
    generator_df.index = np.arange(1,len(generator_df)+1)
    substation_df.index = np.arange(1,len(substation_df)+1)

    
    branch_df.index = branch_df.index.astype(object)
    generator_df.index = generator_df.set_index('Gen_i').index.astype(object)
    bus_df.index = bus_df.set_index('bus_i').index.astype(object)
    Pload_df.index = Pload_df.index.astype(object)
    Qload_df.index = Qload_df.index.astype(object)
    branch_df = branch_df.astype(object)
    generator_df = generator_df.astype(object)
    substation_df = substation_df.astype(object)
    bus_df = bus_df.astype(object)
    Qload_df = Qload_df.astype(object)
    CapData.index = CapData.set_index('Cap_i').index.astype(object)
    CapData = CapData.astype(object)

    model = create_model()

    time_periods = list(Pload_df[Pload_df['Phase'] == 'A'].copy().index)
    initialize_buses(model, bus_names=bus_df.index)
    initialize_time_periods(model, time_periods=time_periods)

    Z_base = (substation_df['V_Base'].unique().item()**2/(substation_df['S_Base'].unique().item()/3000))
    I_base = (substation_df['S_Base'].unique().item())/(substation_df['V_Base'].unique().item()*3)
    HeadBus = substation_df['HeadBus'].values[0]

    initialize_network(model, transmission_lines=list(branch_df.index), leng=branch_df['Length'].to_dict(), bus_from=branch_df['F_BUS'].to_dict(), bus_to=branch_df['T_BUS'].to_dict(), config=branch_df['Config'].to_dict(), Z_base=Z_base, ThermalCap=branch_df['ThermalCap'].to_dict(), branch_phases=branch_df['Phases'].to_dict(), HeadBus=[HeadBus])

    lines_to = {b: list() for b in bus_df.index.unique()}
    lines_from = {b: list() for b in bus_df.index.unique()}
    
    for i, l in branch_df.iterrows():
        lines_from[l['F_BUS']].append(i)
        lines_to[l['T_BUS']].append(i)

    derive_network(model, lines_from=lines_from, lines_to=lines_to)
    A_tilde, A_0, A = create_incidence_matrix(model)
    inv_A = np.linalg.inv(A)
    D_r, D_x = calculate_network_parameters(model, config=config)
    A_dash, R_d, X_d = update_func(A_0, A, D_r, D_x) #I've incorporated that V will account for P_L
    generator_at_bus = {b: list() for b in generator_df['GEN_BUS'].unique()}

    for i, g in generator_df.iterrows():
        generator_at_bus[g['GEN_BUS']].append(i)
   
    initialize_generators(model,
                        generator_names=generator_df.index,
                        generator_at_bus=generator_at_bus)
    
    maximum_minimum_activepower_output_generators(model,
                                        minimum_power_output=generator_df['PMIN'].to_dict(),
                                        maximum_power_output=generator_df['PMAX'].to_dict())

    maximum_minimum_reactivepower_output_generators(model,
                                        minimum_power_output=generator_df['QMIN'].to_dict(),
                                        maximum_power_output=generator_df['QMAX'].to_dict())
    
    
    initialize_spotload(model, SpotLoads = SpotLoads)

    if alpha == None:
        if BothBidsOffers:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, InfeasibleFlag=InfeasibleFlag)
        elif OnlyBids:
            initialize_DERload(model, DER_at_bus=DER_bids_at_bus, DER_names=DER_bids_names, DER_phases=DER_bids_phases, DER_pi=DER_pi_bids, DER_P_per_phase=DER_P_per_phase_bids, DERP_dict=DERP_bids_dict, InfeasibleFlag=InfeasibleFlag)
        elif OnlyOffers:
            initialize_DERload(model, DER_at_bus=DER_offers_at_bus, DER_names=DER_offers_names, DER_phases=DER_offers_phases, DER_pi=DER_pi_offers, DER_P_per_phase=DER_P_per_phase_offers, DERP_dict=DERP_offers_dict, InfeasibleFlag=InfeasibleFlag)
        elif NaiveCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, InfeasibleFlag=InfeasibleFlag)
        elif FinalCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, InfeasibleFlag=InfeasibleFlag)
        elif MutualCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, InfeasibleFlag=InfeasibleFlag)
        elif OnlyBids_MC:
            initialize_DERload(model, DER_at_bus=DER_bids_at_bus, DER_names=DER_bids_names, DER_phases=DER_bids_phases, DER_pi=DER_pi_bids, DER_P_per_phase=DER_P_per_phase_bids, DERP_dict=DERP_bids_dict, InfeasibleFlag=InfeasibleFlag)

    else:
        if BothBidsOffers:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, alpha=alpha, InfeasibleFlag=InfeasibleFlag)
        elif OnlyBids:
            initialize_DERload(model, DER_at_bus=DER_bids_at_bus, DER_names=DER_bids_names, DER_phases=DER_bids_phases, DER_pi=DER_pi_bids, DER_P_per_phase=DER_P_per_phase_bids, DERP_dict=DERP_bids_dict, alpha=alpha, InfeasibleFlag=InfeasibleFlag)
        elif OnlyOffers:
            initialize_DERload(model, DER_at_bus=DER_offers_at_bus, DER_names=DER_offers_names, DER_phases=DER_offers_phases, DER_pi=DER_pi_offers, DER_P_per_phase=DER_P_per_phase_offers, DERP_dict=DERP_offers_dict, alpha=alpha, InfeasibleFlag=InfeasibleFlag)
        elif NaiveCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, alpha=alpha, InfeasibleFlag=InfeasibleFlag)
        elif FinalCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, alpha=alpha, InfeasibleFlag=InfeasibleFlag)
        elif MutualCase:
            initialize_DERload(model, DER_at_bus=DER_at_bus, DER_names=DER_names, DER_phases=DER_phases, DER_pi=DER_pi, DER_P_per_phase=DER_P_per_phase, DERP_dict=DERP_dict, alpha=alpha, MutDER=MutDER, InfeasibleFlag=InfeasibleFlag)
        elif OnlyBids_MC:
            initialize_DERload(model, DER_at_bus=DER_bids_at_bus, DER_names=DER_bids_names, DER_phases=DER_bids_phases, DER_pi=DER_pi_bids, DER_P_per_phase=DER_P_per_phase_bids, DERP_dict=DERP_bids_dict, InfeasibleFlag=InfeasibleFlag)

    cap_at_bus = {b: list() for b in CapData['Node'].unique()}
    
    for i, c in CapData.iterrows():
        cap_at_bus[c['Node']].append(i)
    
    maximum_output = cap_values_per_phase(CapData)
    
    initialize_cap(model, cap_names=CapData.index,
                            cap_at_bus=cap_at_bus)

    maximum_output_capacitors(model,
                            maximum_output = maximum_output,
                            Switching = CapData['Switching'].to_dict())

    switching_cap_func(model, 
                    Switching = CapData['Switching'].to_dict(),
                    SwitchingCost=CapData['Cost'].to_dict())

    if results_cap == None:

        if NaiveCase:
            initialize_model(model, V_max=2, V_min=0, V_base = (substation_df['V_Base'].unique().item()), S_base = (substation_df['S_Base'].unique().item()), I_base = I_base, Sub_S_Max= (substation_df['Ssub_Max'].unique().item()), bus_phases = bus_df['Phases'].to_dict(), BigM=case[0], pf=case[1], MaxPOffer=MaxP)
        else:
            initialize_model(model, V_max=bus_df['Vmax'].unique().item(), V_min=bus_df['Vmin'].unique().item(), V_base = (substation_df['V_Base'].unique().item()), S_base = (substation_df['S_Base'].unique().item()), I_base = I_base, Sub_S_Max= (substation_df['Ssub_Max'].unique().item()), bus_phases = bus_df['Phases'].to_dict(), BigM=case[0], pf=case[1], MaxPOffer=MaxP)
        

    if Inp_thermal == '1' and not NaiveCase:
        thermal_df.index = thermal_df.set_index('c').index.astype(object)
        thermal_limits(model, alpha_c=thermal_df['alpha_c'].to_dict(), beta_c=thermal_df['beta_c'].to_dict(), delta_c=thermal_df['delta_c'].to_dict())
        thermal_limit_constraint(model)
    
    if NaiveCase:
        constraint_slack_bus(model, V_Slack = bus_df['V_0'].iloc[int(HeadBus[-1][-1])])
        voltage_constraint(model, A_dash, R_d, X_d)
        activepower_balance_constraint(model, A.T, A_0.T)
        reactivepower_balance_constraint(model, A.T, A_0.T)
        constraint_net_activepower(model)
        constraint_net_reactivepower(model)
        objective_function(model)
    
    else:
        #active_power_source_bus_constraint(model)
        #reactive_power_source_bus_constraint(model)
        constraint_slack_bus(model, V_Slack = bus_df['V_0'].iloc[int(HeadBus[-1][-1])])
        voltage_constraint(model, A_dash, R_d, X_d)
        activepower_balance_constraint(model, A.T, A_0.T)
        reactivepower_balance_constraint(model, A.T, A_0.T)
        
        if OnlyOffers:
            # constraint_net_activepower_onlyoffers(model)
            # constraint_net_reactivepower_onlyoffers(model)
            constraint_net_activepower(model)
            constraint_net_reactivepower(model)
        else:
            constraint_net_activepower(model)
            constraint_net_reactivepower(model)
        
        if MutualCase:
            mutual_case_constraint(model)
        
        if OnlyBids_MC:
            constraint_active_scheduled_interchange(model, SourceActivePower)
            constraint_reactive_scheduled_interchange(model, SourceReactivePower)


        substation_power_limit_constraint(model)
        objective_function(model)
    return PSSTModel(model)

class PSSTModel(object):

    def __init__(self, model, is_solved=False):
        self._model = model
        self._is_solved = is_solved
        self._status = None
        self._results = None

    def __repr__(self):

        repr_string = 'status={}'.format(self._status)

        string = '<{}.{}({})>'.format(
                    self.__class__.__module__,
                    self.__class__.__name__,
                    repr_string,)


        return string

    def solve(self, solver='glpk', verbose=False, keepfiles=True, **kwargs):
        TC = solve_model(self._model, solver=solver, verbose=verbose, keepfiles=keepfiles, **kwargs)
        self._results = PSSTResults(self._model)
        return TC

    def sort_buses(self):
        self._model.Buses = sorted(self._model.Buses, key=lambda bus: int(re.search(r'\d+', bus).group()))

    @property
    def results(self):
        return self._results