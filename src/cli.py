import os
import sys
import click
import pandas as pd
import random
import numpy as np
from pathlib import Path
from glob import glob
import datetime
import pytz
from tqdm import tqdm
import re
import threading
import pickle
import time
import seaborn as sns
from model import build_model
from model.utils import (create_matrix_dict_pandas, correct_names, data_indexing, phases_from_config, descending_sort_dict, get_duals_in_numpy_vector, custom_sort_key)
from IDSO import count_bids, count_offers, combined_data, bubble_sizes, MaxP, args
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as mticker
plt.rc('font', family='serif', serif=['Times New Roman'])
plt.rc('legend', fontsize=14)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
figures_dir = parent_dir / "Figures"
output_dir = parent_dir / "OutputFiles"
csv_dir = output_dir /"CSVFiles"
np.seterr(all='raise')

SOLVER = os.getenv('PSST_SOLVER')

LMP = args.LMP #for paper
M = args.M
pf = args.pf
C_IDSO = args.C0

def OPF(solver, data=None, OnlyBids=False, OnlyOffers=False, BothBidsOffers=False, NaiveCase=False, FinalCase=False, alpha=None, LMP=LMP, MutualCase=False, MutDER = None, MCData=None, OnlyBids_MC=False, SourceActivePower=None, SourceReactivePower=None, InfeasibleFlag=False):

	if OnlyBids:
		outf = 'OnlyBids'
	elif OnlyOffers:
		outf = 'OnlyOffers'
	elif BothBidsOffers:
		outf = 'BothBidsOffers'
	elif NaiveCase:
		outf = 'NaiveCase'
	elif FinalCase:
		outf = 'FinalCase'
	elif MutualCase:
		outf = 'MutualCase'
	elif OnlyBids_MC:
		outf = 'OnlyBids_MC'
	
	t = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d-%H-%M-%S")
	output = os.path.join(os.getcwd(), "../OutputFiles/T-DOPFFiles/Results_{}_{}.dat").format(t,outf)

	datafile_csv = "OPF_data/"+args.TestCase

	branch_df = pd.read_csv(datafile_csv+'/branch.csv')
	generator_df = pd.read_csv(datafile_csv+'/gen.csv')
	bus_df = pd.read_csv(datafile_csv+'/bus.csv')
	capacitor_df = pd.read_csv(datafile_csv+'/CapData.csv')
	substation_df = pd.read_csv(datafile_csv+'/Substation.csv')
	SpotLoadsP = data_indexing(datafile_csv+'/SpotLoadsP.csv', 'Node', len(bus_df), 'Bus')
	SpotLoadsQ = data_indexing(datafile_csv+'/SpotLoadsQ.csv', 'Node', len(bus_df), 'Bus')
	
	global fixedloadsum
	fixedloadsum = SpotLoadsP.sum().sum()

	if MCData is not None:
		for i in MCData:
			for k, v in i.items():
				for x in v[1]:
					if k[1] in SpotLoadsP.index:
						print(k[1], v[0]*v[-1]/len(v[1]))
						SpotLoadsP.loc[k[1], x] -= v[0]*v[-1]/len(v[1])
						SpotLoadsQ.loc[k[1], x] -= v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1]))
					else:
						if x == 'A':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [-v[0]*v[-1]/len(v[1]), 0, 0]
							SpotLoadsQ.loc[k[1]] =  [-v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1])), 0, 0]
						if x == 'B':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [0, -v[0]*v[-1]/len(v[1]), 0]
							SpotLoadsQ.loc[k[1]] =  [0, -v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1])), 0]
						if x == 'C':
							print(k[1],v[0]*v[-1]/len(v[1]))
							SpotLoadsP.loc[k[1]] =  [0, 0, -v[0]*v[-1]/len(v[1])]
							SpotLoadsQ.loc[k[1]] =  [0, 0, -v[0]*(np.sqrt((1/pf**2)-1))*(v[-1]/len(v[1]))]
			
			SpotLoadsP.sort_index(inplace=True)
			SpotLoadsQ.sort_index(inplace=True)


	if os.path.isfile(datafile_csv+"/thermal_constraint_data.xlsx"): 
		thermal_const_df = pd.read_excel(datafile_csv+"/thermal_constraint_data.xlsx")
		print("\n\nThermal constraints related data found.")
		Inp_thermal = '1'

	else:
		thermal_const_df = None
		Inp_thermal = '0'
		print("Thermal constraint related data file not found, the problem will ignore thermal constraints.")

	branch_df = correct_names(branch_df, 'F_BUS', 'Bus')
	branch_df = correct_names(branch_df, 'T_BUS', 'Bus')
	branch_df = phases_from_config(branch_df, need_phases=True)
	bus_df = correct_names(bus_df, 'bus_i', 'Bus')
	bus_df = phases_from_config(bus_df, need_phases=False)
	generator_df = correct_names(generator_df, 'Gen_i', 'GenCo')
	generator_df = correct_names(generator_df, 'GEN_BUS', 'Bus')
	capacitor_df = correct_names(capacitor_df, 'Cap_i', 'Cap')
	capacitor_df = correct_names(capacitor_df, 'Node', 'Bus')
	substation_df = correct_names(substation_df, 'HeadBus', 'Bus')

	cwd = os.getcwd()
	path = r"OPF_data\load" 
	path = os.path.join(cwd, path)
	Pload_df = []
	Qload_df = []

	csv_files = glob(os.path.join(path, "*.csv"))
	global model
	global SolverOutcomes
	for f in csv_files:
		df = pd.read_csv(f)
		p_df = os.path.basename(f)
		p = p_df.split('-')[-1].split('.')[0]
		p = p.capitalize()
		df['Phase'] = p
		if p_df[0] == 'P':
			Pload_df.append(df)
		else:
			Qload_df.append(df)
	Pload_df = pd.concat(Pload_df, ignore_index=False)
	Qload_df = pd.concat(Qload_df, ignore_index=False)

	HeadBus = substation_df['HeadBus'].values[0]
	config_dict, config_numpy = create_matrix_dict_pandas(datafile_csv+"/config.csv")
	
	def display_loading_bar():
		while not done:
			print("Working on problem...", end="\r")
			time.sleep(0.5)
	
	done = False
	loading_thread = threading.Thread(target=display_loading_bar)
	loading_thread.start()

	try:
		model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, OnlyBids=OnlyBids, OnlyOffers=OnlyOffers, BothBidsOffers=BothBidsOffers, NaiveCase=NaiveCase, FinalCase=FinalCase, MutualCase=MutualCase, OnlyBids_MC=OnlyBids_MC, SourceActivePower=SourceActivePower, SourceReactivePower=SourceReactivePower, InfeasibleFlag=InfeasibleFlag)

		SolverOutcomes = model.solve(solver=solver)
		Status= str(SolverOutcomes[1])
		

	finally:
		done=True
		loading_thread.join()
		datetime_india = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

		if (Status == 'optimal'):
			model.sort_buses()
			instance = model._model
			results_alpha = {}
			

			SolverOutcomes = model.solve(solver=solver)
			instance = model._model	
			result = model.results

			if NaiveCase and alpha==None:
				for bus in instance.Buses:
					if bus in instance.DERAtBus:
						for d in instance.DERAtBus[bus].data():
							if instance.DERP[d] >= 0:
								if instance.DERPi[d] <= LMP:
									results_alpha[d] = instance.DERAlpha[d].value	
								else:
									results_alpha[d] = 0.0
							else:
								if instance.DERPi[d] >= LMP:
									results_alpha[d] = instance.DERAlpha[d].value	
								else:
									results_alpha[d] = 0.0
				
				model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, alpha=results_alpha, OnlyBids=OnlyBids, OnlyOffers=OnlyOffers, BothBidsOffers=BothBidsOffers, NaiveCase=NaiveCase, FinalCase=FinalCase, InfeasibleFlag=InfeasibleFlag)

				SolverOutcomes = model.solve(solver=solver)

				instance = model._model	
				result = model.results

			elif NaiveCase and alpha is not None:
				print("Doing Naive Case without using Bins")
				model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, alpha=alpha, OnlyBids=OnlyBids, OnlyOffers=OnlyOffers, BothBidsOffers=BothBidsOffers, NaiveCase=NaiveCase, FinalCase=FinalCase, InfeasibleFlag=InfeasibleFlag)
				
				SolverOutcomes = model.solve(solver=solver)

				instance = model._model	
				result = model.results
			
			elif FinalCase and alpha is not None:
				print("Doing WPM Case for checking")
				model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, alpha=alpha, OnlyBids=OnlyBids, OnlyOffers=OnlyOffers, BothBidsOffers=BothBidsOffers, NaiveCase=NaiveCase, FinalCase=FinalCase, InfeasibleFlag=InfeasibleFlag)
				
				SolverOutcomes = model.solve(solver=solver)

				instance = model._model	
				result = model.results
			
			elif MutualCase and alpha is not None:
				print("Doing Mututally Contingent Case")
				model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, alpha=alpha, OnlyBids=OnlyBids, OnlyOffers=OnlyOffers, BothBidsOffers=BothBidsOffers, NaiveCase=NaiveCase, FinalCase=FinalCase, MutualCase=MutualCase, MutDER=MutDER, InfeasibleFlag=InfeasibleFlag)
				
				SolverOutcomes = model.solve(solver=solver)

				instance = model._model	
				result = model.results
			
			elif OnlyBids_MC and MCData is not None:
				print("Doing Only Bids Mututally Contingent Case")
				model = build_model(case = (M,pf), substation_df = substation_df, branch_df=branch_df, generator_df=generator_df, bus_df=bus_df, Pload_df=Pload_df, Qload_df=Qload_df, config = (config_dict, config_numpy), SpotLoads = (SpotLoadsP,SpotLoadsQ), CapData=capacitor_df, thermal_df=thermal_const_df, Inp_thermal=Inp_thermal, alpha=alpha, OnlyBids_MC=OnlyBids_MC, MCData=MCData, SourceActivePower=SourceActivePower, SourceReactivePower=SourceReactivePower, InfeasibleFlag=InfeasibleFlag)
				
				SolverOutcomes = model.solve(solver=solver)
				instance = model._model	
				result = model.results

			with open(output.strip("'"), 'w') as f:

				f.write("THE OPF WAS RUN AT : ") 
				f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
				f.write("\n\nSOLUTION_STATUS\n")
				f.write("optimal \t")
				f.write("\nEND_SOLUTION_STATUS\n\n")
				
				f.write("DGResultsforActivePower\n\n")
				
				for g in instance.Generators.data():
					f.write("%s\n" % str(g).ljust(8))
					for t in instance.TimePeriods:
						f.write("Interval: {}\n".format(str(t)))
						for p in instance.Phases:
							if instance.ActivePowerGenerated[p, g, t].value != None:
								f.write("\tActivePowerGenerated: {} kW at Phase {}\n".format(round(instance.ActivePowerGenerated[p, g, t].value * instance.S_Base.value/3,5), p))
							else:
								f.write("\tActivePowerGenerated: {} kW at Phase {}\n".format(0, p))
				f.write("\nEND_DGResultsforActivePower\n\n")

				f.write("DGResultsforReactivePower\n\n")
				for g in instance.Generators.data():
					f.write("%s\n" % str(g).ljust(8))
					for t in instance.TimePeriods:
						f.write("Interval: {}\n".format(str(t)))
						for p in instance.Phases:
							if instance.ReactivePowerGenerated[p, g, t].value != None:
								f.write("\tReactivePowerGenerated: {} kVAr at Phase {}\n".format(round(instance.ReactivePowerGenerated[p, g, t].value * instance.S_Base.value/3,5), p))
							else:
								f.write("\tReactivePowerGenerated: {} kVAr at Phase {}\n".format(0, p))
				f.write("\nEND_DGResultsforReactivePower\n\n")

				f.write("VOLTAGE MAGNITUDES\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases: 
							if p in instance.BusPhase[bus]:
								f.write('Phase: {} Bus: {} Interval: {} : {} p.u.\n'.format(str(p), str(bus), str(t), str(round(np.sqrt(instance.V[p, bus, t].value), 5))))
							else:
								f.write('Phase: {} Bus: {} Interval: {} : {} \n'.format(str(p), str(bus), str(t), 'NaN', 5))
				f.write("\nEND VOLTAGE MAGNITUDES\n\n")

				f.write("Active Power at Each Node [Excludes Source Bus Injection]\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.P[p, bus, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_ACTIVE_POWER_AT_EACH_NODE\n\n")

				f.write("LINE_ACTIVE_POWER_FLOWS\n\n")
				for l in sorted(instance.TransmissionLines):
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kW\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.P_L[p, l, t].value  * instance.S_Base.value/3, 5))))
				f.write("\nEND_LINE_ACTIVE_POWER_FLOWS\n\n")

				f.write("Reactive Power at Each Node [Excludes Source Bus Injection]\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(round(instance.Q[p, bus, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_REACTIVE_POWER_AT_EACH_NODE\n\n")

				f.write("LINE_REACTIVE_POWER_FLOWS\n\n")
				for l in sorted(instance.TransmissionLines):
					for t in instance.TimePeriods:
						for p in instance.Phases:
							f.write('Phase: {} Line Connecting: {} to {} Interval: {} : {} kVAr\n'.format(str(p), str(instance.BusFrom[l]), str(instance.BusTo[l]), str(t), str(round(instance.Q_L[p, l, t].value * instance.S_Base.value/3, 5))))
				f.write("\nEND_LINE_REACTIVE_POWER_FLOWS\n\n")

				f.write("ACTIVE_POWER_AT_SOURCE_BUS\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							if bus == HeadBus:
								if instance.ActivePowerAtSourceBus[p,bus, t].value != None:
									f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(round(instance.ActivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
								else:
									f.write('Phase: {} Bus: {} Interval: {} : {} kW\n'.format(str(p), str(bus), str(t), str(0)))
				f.write("\nEND_ACTIVE_POWER_AT_SOURCE_BUS\n\n")

				f.write("REACTIVE_POWER_AT_SOURCE_BUS\n\n")
				for bus in instance.Buses:
					for t in instance.TimePeriods:
						for p in instance.Phases:
							if bus == HeadBus:
								if instance.ReactivePowerAtSourceBus[p,bus, t].value != None:
									f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(round(instance.ReactivePowerAtSourceBus[p, bus, t].value * instance.S_Base.value/3, 5))))
								else:
									f.write('Phase: {} Bus: {} Interval: {} : {} kVAr\n'.format(str(p), str(bus), str(t), str(0)))
				f.write("\nEND_REACTIVE_POWER_AT_SOURCE_BUS\n\n")


				f.write("DER_States\n\n")
				for d in instance.DER.data():
					f.write("\t{}: at Phases: {} Alpha: {} Pi: {} P: {}\n".format(d, instance.DERPhases[d], round(instance.DERAlpha[d].value,3), round(instance.DERPi[d],3), round(instance.DERP[d] * len(instance.DERPhases[d]),3)))
				f.write("\nEND_DER_States\n\n")



		elif (Status == 'infeasible'):
			with open(output.strip("'"), 'w') as f:
				f.write("THE OPF WAS RUN AT : ") 
				f.write(datetime_india.strftime('%Y:%m:%d %H:%M:%S %Z %z'))
				f.write("\nSOLUTION_STATUS\n")
				f.write("infeasible \t")
				f.write("\nEND_SOLUTION_STATUS\n")

	return result, SolverOutcomes

if __name__ == "__main__":

	CategDuals = {}
	Duals = {}
	Accepted_bids = {}
	Not_accepted_bids = {}
	FinalAlpha = {}
	FinalAlphaBids = {}
	FinalAlphaOffers = {}
	total_bids = count_bids
	SourceActivePower = {}
	SourceReactivePower = {}
	BinIData = {}
	BinIIData = {}

	######Bin I###################

	results, SolverOutcome = OPF(SOLVER, OnlyBids=True)
	instance = model._model

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERAlpha[d].value > 0:
						Accepted_bids[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))
				else:
						Not_accepted_bids[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERPi[d] >= LMP + C_IDSO:
					FinalAlpha[d] = instance.DERAlpha[d].value
					if instance.DERAlpha[d].value != 0:
						FinalAlphaBids[(d,bus)] = [round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d])]	
				else:
					FinalAlpha[d] = 0.0
	
	print("\n\n\nFINAL ALPHA Bids")
	print(FinalAlphaBids)
	# print("\n Min. Bid Value:")
	# minbid = min(Accepted_bids, key=lambda k: Accepted_bids[k][2])
	# print(minbid)
	datalist = []
	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				datalist.append({
						"DER Name": d,
						"Alpha": FinalAlpha[d],
						"Phases": instance.DERPhases[d],
						"Bus": bus[3:],
						"Pi": round(instance.DERPi[d],1),
						"P_inj": FinalAlpha[d]*round(instance.DERP[d] * len(instance.DERPhases[d]),0),
						"Q_inj": FinalAlpha[d]*round(instance.DERP[d] * len(instance.DERPhases[d]),0)*(np.sqrt((1/pf**2)-1)) ,
					})

	Accepted_bids = descending_sort_dict(Accepted_bids)
	Not_accepted_bids = descending_sort_dict(Not_accepted_bids)
	bid_prices = []
	bid_prices2 = []
	bid_quantities = []
	for k,v in Accepted_bids.items():
		price = v[2] - C_IDSO
		price2 = v[2]
		quantity = v[3] * v[0]
		bid_prices.append(price)
		bid_prices2.append(price2)
		bid_quantities.append(abs(quantity))

	maxpibid = max(bid_prices)
	maxpibid += 15
	bid_quantities.append(fixedloadsum)
	bid_prices.append(maxpibid)
	bid_prices2.append(maxpibid)
	bid_data = sorted(zip(bid_prices, bid_quantities), key=lambda x: x[0], reverse=True)
	bid_data2 = sorted(zip(bid_prices2, bid_quantities), key=lambda x: x[0], reverse=True)
	
	sorted_bid_prices, sorted_bid_quantities = zip(*bid_data)
	sorted_bid_prices2, sorted_bid_quantities2 = zip(*bid_data2)
	cumulative_bid_quantities = np.cumsum(sorted_bid_quantities)
	cumulative_bid_quantities2 = np.cumsum(sorted_bid_quantities2)

	fig1_nw, ax1_nw = plt.subplots(figsize=(8,5))
	# ax.set_xticks(range(2500,4750,250))
	ax1_nw.step(cumulative_bid_quantities, sorted_bid_prices, where='post', label='Bids (Network Cost Accounted)', color='mediumseagreen')
	ax1_nw.step(cumulative_bid_quantities2, sorted_bid_prices2, where='post', label='Bids (Network Cost Unaccounted)', color='mediumseagreen', alpha=0.45)

	sorted_bid_prices = sorted(sorted_bid_prices,  reverse=True)
	sorted_bid_prices2 = sorted(sorted_bid_prices2,  reverse=True)
	index = np.searchsorted(-np.asarray(sorted_bid_prices), -LMP, side='right')
	index3 = np.searchsorted(-np.asarray(sorted_bid_prices2), -LMP - C_IDSO, side='right')
	index2 = np.searchsorted(-np.asarray(sorted_bid_prices2), -LMP, side='right')

	if index < len(sorted_bid_prices):
		MCV = cumulative_bid_quantities[index]
	else:
		MCV = None

	if index2 < len(sorted_bid_prices2):
		MCV2 = cumulative_bid_quantities2[index2]
	else:
		MCV2 = None

	ax1_nw.hlines(y=LMP, xmin=1000, xmax=MCV, color='black', linestyle='--', label='Observed LMP')
	ax1_nw.vlines(x=MCV, ymin=0, ymax=LMP, color='grey', linestyle='--', label='Cleared Volume')

	ax1_nw.hlines(y=LMP, xmin=1000, xmax=MCV2, color='black', linestyle='--', alpha=0.45)
	ax1_nw.vlines(x=MCV2, ymin=0, ymax=LMP, color='grey', linestyle='--', alpha=0.45)
	
	ax1_nw.scatter([MCV2], [LMP], marker='x', color='seagreen', zorder=5, alpha=0.45)
	ax1_nw.scatter([MCV], [LMP], marker='x', color='seagreen', zorder=5)
	
	xticks = list(ax1_nw.get_xticks())
	
	if MCV not in xticks:
		xticks.append(MCV)
		xticks.remove(1000)
	if MCV2 not in xticks:
		xticks.append(MCV2)
	# ax.set_xticks(xticks)

	yticks = list(ax1_nw.get_yticks())
	if LMP not in yticks:
		yticks.append(LMP)
		# yticks.remove(0)
	ax1_nw.set_yticks(yticks)

	xticks = ax1_nw.get_xticks()
	if MCV != None:
		xticks = [t for t in xticks if abs(t - MCV) > 350]
		xticks.append(MCV)
	if MCV2 != None:
		xticks = [t for t in xticks if abs(t - MCV2) > 350]
		xticks.append(MCV2)


	xticks = sorted(xticks)
	# ax.spines['left'].set_position(('data', 2500))
	ax1_nw.set_title('(a)', fontsize=16, fontweight='bold',y=-0.2)
	ax1_nw.set_xticks(xticks)
	ax1_nw.set_ylim([0,30])
	ax1_nw.set_xlim([1000,4750])
	ax1_nw.tick_params(axis='both', labelsize=13)
	ax1_nw.set_ylabel('Price (¢/kWh)', fontsize=16, fontweight='bold')
	ax1_nw.set_xlabel('Quantity (kW)', fontsize=16, fontweight='bold')
	ax1_nw.legend()
	ax1_nw.grid(alpha=0.15)
	fig1_nw.savefig(figures_dir / "IDSOBidCurve.pdf", bbox_inches='tight')

	print(f"\nTotal Bid Nos:{total_bids}")
	print(f"\nAccepted Bid Nos:{len(Accepted_bids)}")
	print("\nACCEPTED BIDS\n", Accepted_bids)
	print("\nNOT ACCEPTED BIDS\n", Not_accepted_bids)
	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})
	
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
	
	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')

	# axins = inset_axes(ax, width="60%", height="50%", loc='lower left',
	# 				bbox_to_anchor=(0.55, 0.35, 0.55, 0.35),
	# 				bbox_transform=ax.transAxes)

	# # Define the region of interest (adjust x1, x2, y1, y2 as needed)
	# # Automatically find the x-range where voltage crosses 0.95
	# cross_indices = []
	# for idx in x_values:
	# 	if any(df.iloc[:, idx] <= 0.952):
	# 		cross_indices.append(idx)

	# if cross_indices:
	# 	x1 = max(0, min(cross_indices) - 3)
	# 	x2 = min(n - 1, max(cross_indices) + 2)
	# else:
	# 	x1, x2 = 0, n - 1

	# # Set xlim and ylim for the inset axes
	# axins.set_xlim(x1, x2)
	# axins.set_ylim(0.948, 0.952)

	# # Plot the same data on the inset axes with x-values
	# for phase in df.index:
	# 	if phase == 'Phase A':
	# 		mark ='o'
	# 	if phase == 'Phase B':
	# 		mark ='x'
	# 	if phase == 'Phase C':
	# 		mark ='.'
	# 	axins.plot(x_values, df.loc[phase], marker=mark)
	# axins.axhline(y=0.95, color='purple', linestyle='--', label='Lower Voltage Limit')

	# # Set x-axis labels for the inset
	# axins.set_xticks(range(x1, x2+1))
	# inset_labels = [labels[i][3:] if i%5==0 else "" for i in range(x1, x2+1)]
	# # axins.set_yticklabels(fontsize=6)
	# axins.set_xticklabels(inset_labels, rotation=0, fontsize=8)
	# axins.tick_params(axis='y', labelsize=8)
	# # Ensure the background of the inset is not transparent
	# axins.patch.set_alpha(1.0)

	# # Adjust the rectangle connecting the inset and main plot
	# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
	ax.tick_params(axis='both', labelsize=14)
	ax.legend()
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(BinI).pdf", bbox_inches='tight')

	
	CategDuals['ActivePowerBalanceDuals'] = results.activepowerbalance_dual
	CategDuals['ReactivePowerBalanceDuals'] = results.reactivepowerbalance_dual
	CategDuals['VoltageDuals'] = results.voltage_dual
	CategDuals['IDSONetworkCost'] = results.IDSOCostNetowrk_dual
	CategDuals['ThermalLimitDual'] = results.thermallimit_dual
	CategDuals['SubstationLimitDual'] = results.substationlimit_dual
	Duals['OnlyBids'] = CategDuals
	
	#-----------------A-NQPs
	BusPhasePairs = {(b,p) for b in instance.Buses for p in instance.BusPhase[b]}
	ANQPDuals = CategDuals['ActivePowerBalanceDuals'].reset_index()
	ANQPDuals = ANQPDuals[ANQPDuals.apply(lambda row:(row['Bus'], row['Phase']) in BusPhasePairs, axis=1)]
	labels = ANQPDuals['Bus'].unique().tolist()
	ANQPDuals = ANQPDuals.set_index('Bus')
	val_A = ANQPDuals[ANQPDuals['Phase']=='A']['Dual']#.reset_index(drop=True)
	val_B = ANQPDuals[ANQPDuals['Phase']=='B']['Dual']#.reset_index(drop=True)
	val_C = ANQPDuals[ANQPDuals['Phase']=='C']['Dual']#.reset_index(drop=True)
	
	df = pd.DataFrame({
		'Phase A': val_A/(instance.S_Base.value/3),
		'Phase B': val_B/(instance.S_Base.value/3),
		'Phase C': val_C/(instance.S_Base.value/3)
	})
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)

	fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(12,8), constrained_layout=True, sharex=True)

	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax1.plot(df.loc[phase], marker=mark, label=phase)
	
	n = len(df.columns)
	ax1.set_ylabel('Prices (¢/kWh)', fontsize=16, fontweight='bold')
	# ax1.set_xlabel('Buses', fontsize=12, fontweight='bold')
	ax1.set_title('Active Nodal Qualification Prices (¢/kWh)', fontsize=20, fontweight='bold')
	ax1.set_xticks(range(len(labels)))
	# custom_labels = [labels[i][3:] if i % 5 == 0 else "" for i in range(len(labels))]
	# ax1.set_xticklabels(custom_labels, rotation=0, fontsize=10)
	ax1.tick_params(axis='y', labelsize=16)

	ax1.legend()
	ax1.set_xlim([-1,n])
	ax1.grid(alpha=0.15)
	# fig.savefig(figures_dir/"\A-NQP(OnlyBids).pdf", bbox_inches='tight')

	#------------------------------------------ R-NQPs
	BusPhasePairs = {(b,p) for b in instance.Buses for p in instance.BusPhase[b]}
	ANQPDuals = CategDuals['ReactivePowerBalanceDuals'].reset_index()
	ANQPDuals = ANQPDuals[ANQPDuals.apply(lambda row:(row['Bus'], row['Phase']) in BusPhasePairs, axis=1)]
	labels = ANQPDuals['Bus'].unique().tolist()
	ANQPDuals = ANQPDuals.set_index('Bus')
	val_A = ANQPDuals[ANQPDuals['Phase']=='A']['Dual']#.reset_index(drop=True)
	val_B = ANQPDuals[ANQPDuals['Phase']=='B']['Dual']#.reset_index(drop=True)
	val_C = ANQPDuals[ANQPDuals['Phase']=='C']['Dual']#.reset_index(drop=True)
	
	df = pd.DataFrame({
		'Phase A': val_A/(instance.S_Base.value/3),
		'Phase B': val_B/(instance.S_Base.value/3),
		'Phase C': val_C/(instance.S_Base.value/3)
	})
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
	
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'	
		if phase == 'Phase C':
			mark ='.'
		ax2.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax2.set_ylabel('Prices (¢/kVArh)', fontsize=16, fontweight='bold')
	ax2.set_xlabel('Buses', fontsize=16, fontweight='bold')
	ax2.set_title('Reactive Nodal Qualification Prices (¢/kVArh)', fontsize=20, fontweight='bold')
	ax2.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if i % 5 == 0 else "" for i in range(len(labels))]
	ax2.set_xticklabels(custom_labels, rotation=90, ha='center', fontsize=14)
	ax2.tick_params(axis='both', labelsize=16)
	ax2.legend()
	ax2.set_xlim([-1,n])
	ax2.grid(alpha=0.15)
	
	fig.savefig(figures_dir/"BidNQPs.pdf", bbox_inches='tight')

# ##########Bin II ####################

	results, SolverOutcome = OPF(SOLVER, OnlyOffers=True)
	instance = model._model	
	
	Accepted_offers = {}
	Not_accepted_offers = {}
	CategDuals = {}

	total_offers = count_offers

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERAlpha[d].value > 0:
						Accepted_offers[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))
				else:
						Not_accepted_offers[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERPi[d] <= LMP - C_IDSO:
					FinalAlpha[d] = instance.DERAlpha[d].value
					if instance.DERAlpha[d].value != 0:
						FinalAlphaOffers[(d,bus)] = [round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d])]
				else:
					FinalAlpha[d] = 0.0
	print("\n\n\nFINAL ALPHA OFFERS")
	print(FinalAlphaOffers)

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				datalist.append({
						"DER Name": d,
						"Alpha": FinalAlpha[d],
						"Phases": instance.DERPhases[d],
						"Bus": bus[3:],
						"Pi": round(instance.DERPi[d],1),
						"P_inj": FinalAlpha[d]*round(instance.DERP[d] * len(instance.DERPhases[d]),0),
						"Q_inj": FinalAlpha[d]*round(instance.DERP[d] * len(instance.DERPhases[d]),0)*(np.sqrt((1/pf**2)-1)) ,
					})
		
	df=pd.DataFrame(datalist)
	df = df.sort_values(by=['DER Name'], key=lambda x: x.apply(custom_sort_key))
	df.to_csv(output_dir/'DERInjectionData.csv', index=False)
	minoffer = min(Accepted_offers, key=lambda k: Accepted_offers[k][2])

	Accepted_offers = descending_sort_dict(Accepted_offers)
	Not_accepted_offers = descending_sort_dict(Not_accepted_offers)
	
	bid_prices = []
	bid_prices2 = []
	bid_quantities = []

	for k,v in Accepted_offers.items():
		price = v[2] + C_IDSO
		price2 = v[2]
		quantity = v[3] * v[0]
		bid_prices.append(price)
		bid_prices2.append(price2)
		bid_quantities.append(quantity)
	
	bid_data = sorted(zip(bid_prices, bid_quantities), key=lambda x: x[0])
	bid_data2 = sorted(zip(bid_prices2, bid_quantities), key=lambda x: x[0])
	
	sorted_bid_prices, sorted_bid_quantities = zip(*bid_data)
	sorted_bid_prices2, sorted_bid_quantities2 = zip(*bid_data2)

	cumulative_bid_quantities = np.cumsum(sorted_bid_quantities)
	cumulative_bid_quantities2 = np.cumsum(sorted_bid_quantities2)

	fig2_nw, ax2_nw = plt.subplots(figsize=(8,5))
	ax2_nw.step(cumulative_bid_quantities, sorted_bid_prices, where='post', label='Offers (Network Cost Accounted)', color='indianred')
	ax2_nw.step(cumulative_bid_quantities2, sorted_bid_prices2, where='post', label='Offers (Network Cost Unaccounted)', alpha=0.45, color='indianred')
	
	index = np.searchsorted(sorted_bid_prices, LMP, side='right')
	index2 = np.searchsorted(sorted_bid_prices2, LMP, side='right')
	
	if index < len(sorted_bid_prices):
		MCV = cumulative_bid_quantities[index]
	else:
		MCV = None
	
	if index2 < len(sorted_bid_prices2):
		MCV2 = cumulative_bid_quantities2[index2]
	else:
		MCV2 = None
	

	ax2_nw.hlines(y=LMP, xmin=0, xmax=MCV, color='black', linestyle='--', label='Observed LMP')
	ax2_nw.hlines(y=LMP, xmin=0, xmax=MCV2, color='black', linestyle='--', alpha=0.45)
	
	ax2_nw.vlines(x=MCV, ymin=0, ymax=LMP, color='grey', linestyle='--', label='Cleared Volume')
	ax2_nw.vlines(x=MCV2, ymin=0, ymax=LMP, color='grey', linestyle='--', alpha=0.45)

	ax2_nw.scatter([MCV], [LMP], marker='x', color='firebrick', zorder=5)
	ax2_nw.scatter([MCV2], [LMP], marker='x', color='firebrick', alpha=0.45, zorder=5)

	xticks = list(ax2_nw.get_xticks())
	if MCV not in xticks:
		xticks.append(MCV)
		xticks.remove(0)
	# ax.set_xticks(xticks)
	if MCV2 not in xticks:
		xticks.append(MCV2)

	yticks = list(ax2_nw.get_yticks())
	if LMP not in yticks:
		yticks.append(LMP)
		
	ax2_nw.set_yticks(yticks)

	xticks = ax2_nw.get_xticks()
	if MCV != None:
		xticks = [t for t in xticks if abs(t - MCV) > 50]
		xticks.append(MCV)

	if MCV2 != None:
		xticks = [t for t in xticks if abs(t - MCV2) > 50]
		xticks.append(MCV2)



	xticks.append(MCV)
	xticks.append(MCV2)

	xticks = sorted(xticks)
	ax2_nw.set_xticks(xticks)
	ax2_nw.set_title('(b)', fontsize=16, fontweight='bold',y=-0.2)
	ax2_nw.tick_params(axis='both', labelsize=14)
	ax2_nw.set_ylim([0,30])
	ax2_nw.set_xlim([1, 3500])
	ax2_nw.set_ylabel('Price (¢/kWh)', fontsize=16, fontweight='bold')
	ax2_nw.set_xlabel('Quantity (kW)', fontsize=16, fontweight='bold')
	ax2_nw.legend()
	ax2_nw.grid(alpha=0.15)
	fig2_nw.savefig(figures_dir/"IDSOOfferCurve.pdf", bbox_inches='tight')
	
	
	print(f"\nTotal Offer Nos:{total_offers}")
	print(f"\nAccepted Offer Nos:{len(Accepted_offers)}")
	print("\nACCEPTED OFFERS\n", Accepted_offers)
	print("\nNOT ACCEPTED OFFERS\n", Not_accepted_offers)
	
	
	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})

	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)

	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.tick_params(axis='both', labelsize=14)

	ax.axhline(y=1.05, color='purple', linestyle='--', label='Upper Limit on Voltage Magnitude')
	ax.legend()
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(BinII).pdf", bbox_inches='tight')

	CategDuals['ActivePowerBalanceDuals'] = results.activepowerbalance_dual
	CategDuals['ReactivePowerBalanceDuals'] = results.reactivepowerbalance_dual
	CategDuals['VoltageDuals'] = results.voltage_dual
	CategDuals['IDSONetworkCost'] = results.IDSOCostNetowrk_dual
	CategDuals['ThermalLimitDual'] = results.thermallimit_dual
	CategDuals['SubstationLimitDual'] = results.substationlimit_dual
	Duals['OnlyOffers'] = CategDuals
	#-----------------A-NQPs
	BusPhasePairs = {(b,p) for b in instance.Buses for p in instance.BusPhase[b]}
	ANQPDuals = CategDuals['ActivePowerBalanceDuals'].reset_index()
	ANQPDuals = ANQPDuals[ANQPDuals.apply(lambda row:(row['Bus'], row['Phase']) in BusPhasePairs, axis=1)]
	labels = ANQPDuals['Bus'].unique().tolist()
	ANQPDuals = ANQPDuals.set_index('Bus')
	val_A = ANQPDuals[ANQPDuals['Phase']=='A']['Dual']#.reset_index(drop=True)
	val_B = ANQPDuals[ANQPDuals['Phase']=='B']['Dual']#.reset_index(drop=True)
	val_C = ANQPDuals[ANQPDuals['Phase']=='C']['Dual']#.reset_index(drop=True)
	
	df = pd.DataFrame({
		'Phase A': val_A/(instance.S_Base.value/3),
		'Phase B': val_B/(instance.S_Base.value/3),
		'Phase C': val_C/(instance.S_Base.value/3)
	})
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
	
	fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1,figsize=(12,8), constrained_layout=True, sharex=True)

	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax1.plot(df.loc[phase], marker=mark, label=phase)
	
	n = len(df.columns)
	ax1.set_ylabel('Prices (¢/kWh)', fontsize=16, fontweight='bold')
	# ax1.set_xlabel('Buses', fontsize=12, fontweight='bold')
	ax1.set_title('Active Nodal Qualification Prices (¢/kWh)', fontsize=20, fontweight='bold')
	ax1.set_xticks(range(len(labels)))
	# custom_labels = [labels[i][3:] if i % 5 == 0 else "" for i in range(len(labels))]
	# ax1.set_xticklabels(custom_labels, rotation=0, fontsize=10)
	ax1.tick_params(axis='both', labelsize=16)

	ax1.legend()
	ax1.set_xlim([-1,n])
	ax1.grid(alpha=0.15)
	# fig.savefig(figures_dir/"A-NQP(OnlyBids).pdf", bbox_inches='tight')

	#------------------------------------------ R-NQPs
	BusPhasePairs = {(b,p) for b in instance.Buses for p in instance.BusPhase[b]}
	ANQPDuals = CategDuals['ReactivePowerBalanceDuals'].reset_index()
	ANQPDuals = ANQPDuals[ANQPDuals.apply(lambda row:(row['Bus'], row['Phase']) in BusPhasePairs, axis=1)]
	labels = ANQPDuals['Bus'].unique().tolist()
	ANQPDuals = ANQPDuals.set_index('Bus')
	val_A = ANQPDuals[ANQPDuals['Phase']=='A']['Dual']#.reset_index(drop=True)
	val_B = ANQPDuals[ANQPDuals['Phase']=='B']['Dual']#.reset_index(drop=True)
	val_C = ANQPDuals[ANQPDuals['Phase']=='C']['Dual']#.reset_index(drop=True)
	
	df = pd.DataFrame({
		'Phase A': val_A/(instance.S_Base.value/3),
		'Phase B': val_B/(instance.S_Base.value/3),
		'Phase C': val_C/(instance.S_Base.value/3)
	})
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
	
	
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax2.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax2.set_ylabel('Prices (¢/kVArh)', fontsize=16, fontweight='bold')
	ax2.set_xlabel('Buses', fontsize=16, fontweight='bold')
	ax2.set_title('Reactive Nodal Qualification Prices (¢/kVArh)', fontsize=20, fontweight='bold')
	ax2.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if i % 5 == 0 else "" for i in range(len(labels))]
	ax2.set_xticklabels(custom_labels, rotation=90, ha='center', fontsize=14)
	ax2.tick_params(axis='both', labelsize=16)

	ax2.legend()
	ax2.set_xlim([-1,n])
	ax2.grid(alpha=0.15)
	
	fig.savefig(figures_dir/"OffersNQPs.pdf", bbox_inches='tight')

# #################Bin III

	results, SolverOutcome = OPF(SOLVER, BothBidsOffers=True)
	instance = model._model	

	Accepted_bids_III = {}
	Not_accepted_bids_III = {}
	Accepted_offers_III = {}
	Not_accepted_offers_III = {}
	CategDuals = {}

	total_bids, total_offers = count_bids,count_offers

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERAlpha[d].value > 0:
					if instance.DERP[d] <= 0:
						Accepted_bids_III[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))
					else:
						Accepted_offers_III[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))
				else:
					if instance.DERP[d] <= 0:
						Not_accepted_bids_III[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))
					else:
						Not_accepted_offers_III[(d, bus)] = (round(instance.DERAlpha[d].value,3), instance.DERPhases[d], round(instance.DERPi[d],1), round(instance.DERP[d],0) * len(instance.DERPhases[d]))

	
	Accepted_bids_III = descending_sort_dict(Accepted_bids_III)
	Not_accepted_bids_III = descending_sort_dict(Not_accepted_bids_III)

	Accepted_offers_III = descending_sort_dict(Accepted_offers_III)
	Not_accepted_offers_III = descending_sort_dict(Not_accepted_offers_III)
	
	resultsalpha = {}

	for bus in instance.Buses:
		if bus in instance.DERAtBus:
			for d in instance.DERAtBus[bus].data():
				if instance.DERP[d] >= 0:
					if instance.DERPi[d] <= LMP:
						resultsalpha[d] = instance.DERAlpha[d].value	
					else:
						resultsalpha[d] = 0.0
				else:
					if instance.DERPi[d] >= LMP:
						resultsalpha[d] = instance.DERAlpha[d].value	
					else:
						resultsalpha[d] = 0.0


	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})

	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)

	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.tick_params(axis='both', labelsize=14)

	ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')
	ax.legend()
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(Contingent).pdf", bbox_inches='tight')

	print(f"\nTotal Bid Nos:{total_bids}")
	print(f"\nAccepted Bid Nos:{len(Accepted_bids_III)}")
	print("\nACCEPTED BIDS\n", Accepted_bids_III)
	print("\nNOT ACCEPTED BIDS\n", Not_accepted_bids_III)

	print(f"\nTotal Offer Nos:{total_offers}")
	print(f"\nAccepted Offer Nos:{len(Accepted_offers_III)}")
	print("\nACCEPTED OFFERS\n", Accepted_offers_III)
	print("\nNOT ACCEPTED OFFERS\n\n\n", Not_accepted_offers_III)

#### Mutually Contingent Bids/Offers ######

	MutuallyContingentBids = filter(lambda x: x in Accepted_bids_III, Not_accepted_bids)
	MutuallyContingentOffers = filter(lambda x: x in Accepted_offers_III, Not_accepted_offers)
	MCBids = list(MutuallyContingentBids)
	MCOffers = list(MutuallyContingentOffers)	

	shaded_bids = [i for i in [key[0] for key in MCBids]]
	shaded_offers = [i for i in [key[0] for key in MCOffers]]


	with open(figures_dir/"BidOfferPlot.pkl", "rb") as f:
		fig = pickle.load(f)
	ax = fig.gca()

	shaded_data = combined_data[combined_data['DER'].isin(shaded_bids)]
	ax.scatter(
		shaded_data['Bus_Pos'],
		shaded_data['Price'],
		s=bubble_sizes[shaded_data.index],
		c='darkgreen',
		alpha=1,
		linewidths=1,
		label='Mutually Contigent Bids'
		)
	shaded_data = combined_data[combined_data['DER'].isin(shaded_offers)]
	ax.scatter(
		shaded_data['Bus_Pos'],
		shaded_data['Price'],
		s=bubble_sizes[shaded_data.index],
		c='maroon',
		alpha=1,
		linewidths=1,
		label='Mutually Contigent Offers'
		)
	ax.axhline(y=LMP, color='black', linestyle='--', label='Observed LMP')
	ax.legend(borderpad=1.5)
	fig.savefig(figures_dir/"Contingency.pdf", bbox_inches='tight')

########################Clearing MC Bids and Offers######################################
	MCBidsQual = {}
	MCOffersQual = {}
	MutualDERs = []

	for x in MCBids:
		if Accepted_bids_III[x][2] >= LMP + C_IDSO:
			MCBidsQual[x] = Accepted_bids_III[x]
			MutualDERs.append(x[0])

	for x in MCOffers:
		if Accepted_offers_III[x][2] <= LMP - C_IDSO:
			MCOffersQual[x] = Accepted_offers_III[x]
			MutualDERs.append(x[0])
	
	results, SolverOutcome = OPF(SOLVER, MutualCase=True, MutDER = MutualDERs, alpha=FinalAlpha)

	assert str(SolverOutcome[1]) == 'optimal', "Solver did not return optimal solution for Mutual Contingent Case."
	
	# if str(SolverOutcome[1]) != 'optimal':
	# 	results, SolverOutcome = OPF(SOLVER, MutualCase=True, MutDER = MutualDERs, alpha=FinalAlpha, InfeasibleFlag=True)
		
	instance = model._model

	MCBidsClear = {}
	MCOffersClear = {}

	for k,v in MCBidsQual.items():
		v = list(v)
		v[0] = instance.DERAlpha[k[0]].value
		FinalAlpha[k[0]] = instance.DERAlpha[k[0]].value
		MCBidsQual[k] = v
		if v[0]!=0:
			MCBidsClear[k] = v

	for k,v in MCOffersQual.items():
		v = list(v)
		v[0] = instance.DERAlpha[k[0]].value
		FinalAlpha[k[0]] = instance.DERAlpha[k[0]].value
		MCOffersQual[k] = v
		if v[0]!=0:	
			MCOffersClear[k] = v

	MCOffersQual = dict(sorted(MCOffersQual.items(), key=lambda item: item[1][2]))
	MCBidsQual = dict(sorted(MCBidsQual.items(), key=lambda item: item[1][2], reverse=True))
	print("\n\nMC Bids Qualified for Clearing:\n", MCBidsQual)
	print("\n\nMC Offers Qualified for Clearing:\n", MCOffersQual)
#sjdfwowht############ Don't Uncomment this part unless you want to ###############

	bid_keys = list(MCBidsQual.keys())
	offer_keys = list(MCOffersQual.keys())

	x_labels = []
	bar_values = []
	bar_colors = []
	quantum_vals = []   # (index 0) × (index 3)
	cumulative = []
	cum_val = 0

	for i in range(max(len(bid_keys), len(offer_keys))):
		if i < len(bid_keys):
			key = bid_keys[i]
			der_label = key[0]  
			x_labels.append(der_label)
			bar_values.append(MCBidsQual[key][2])
			bar_colors.append('seagreen')
			quantum = MCBidsQual[key][0] * MCBidsQual[key][3]
			cum_val += quantum
			quantum_vals.append(quantum)
			cumulative.append(cum_val)

		if i < len(offer_keys):
			key = offer_keys[i]
			der_label = key[0]
			x_labels.append(der_label)
			# Offers: index 2 negated so it appears below the x-axis
			bar_values.append(-MCOffersQual[key][2])
			bar_colors.append('firebrick')
			quantum = MCOffersQual[key][0] * MCOffersQual[key][3]
			cum_val += quantum
			quantum_vals.append(quantum)
			cumulative.append(cum_val)

	modified_labels = [lbl.replace("DER_", "DER ") for lbl in x_labels]

	# --- Plot ---
	fig, ax = plt.subplots(figsize=(6.5, 6.5))  # Shrink the figure width
	positions = range(len(x_labels))
	bar_width = 0.7

	# Primary axis: bar chart
	bars = ax.bar(positions, bar_values, width=bar_width, color=bar_colors)
	ax.set_xticks(positions)
	ax.set_xticklabels(modified_labels, rotation=90)
	ax.set_xlabel("Mutually Contingent DERs", fontweight="bold", fontsize=14)
	ax.set_ylabel("Price (¢/kWh)", fontweight="bold", fontsize=14)
	ax.tick_params(labelsize=11)
	# ax.set_title("Interleaved Bids (Seagreen) & Offers (Firebrick)\nwith Quantum & Cumulative Quantum")

	# Threshold lines
	ax.axhline(y=LMP+C_IDSO, color='seagreen', linestyle='--', linewidth=1.0, label='Bid Clearing Price (15.5 ¢/kWh)')
	ax.axhline(y=-LMP+C_IDSO, color='firebrick', linestyle='--', linewidth=1.0, label='Offer Clearing Price (10.5 ¢/kWh)')
	ax.axhline(y=0, color='black', linewidth=1.0)

	# Display negative bar values as positive on the y-axis
	ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{abs(val):.1f}"))

	for bar, quantum in zip(bars, quantum_vals):
		x_pos = bar.get_x() + bar.get_width() / 2
		height = bar.get_height()
		if height >= 0:
			# Bids (above x-axis)
			ax.text(x_pos, height + 0.5, f"{quantum:.1f} kW",
					ha='center', va='bottom', color='black', fontsize=11)
		else:
			# Offers (below x-axis)
			ax.text(x_pos, height - 0.5, f"{quantum:.1f} kW",
					ha='center', va='top', color='black', fontsize=11)

	# Primary legend
	handles, labels = ax.get_legend_handles_labels()

	# Secondary axis for cumulative quantum
	ax2 = ax.twinx()
	ax2.set_ylabel("Cumulative Quantum of Power from MC DERs", color='black', fontweight="bold", fontsize=14)
	ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
	min_cum = min(cumulative)
	max_cum = max(cumulative)
	padding = 20 # Adjust as desired
	ax2.set_ylim(min_cum - padding, max_cum + padding)

	# Reduce the number of major ticks
	ax2.yaxis.set_major_locator(mticker.MaxNLocator(5))
	line, = ax2.plot(positions, cumulative, color='black',
					marker='o', linewidth=2, label='Cumulative Quantum')

	# Combine legends
	handles2, labels2 = [line], ["Cumulative Quantum"]
	ax.legend(handles + handles2, labels + labels2)
	# plt.tight_layout()
	fig.savefig(figures_dir/"MCBidsOffersRet.pdf", bbox_inches='tight')


##################Getting WPM Results Before Moving Further ######################
	results, SolverOutcome = OPF(SOLVER, FinalCase=True, LMP=LMP, alpha=FinalAlpha)
	assert str(SolverOutcome[1]) == 'optimal', "Solver did not return optimal solution for Final Case."
	# if str(SolverOutcome[1]) != 'optimal':
	# 	results, SolverOutcome = OPF(SOLVER, FinalCase=True, LMP=LMP, alpha=FinalAlpha, InfeasibleFlag=True)

	instance = model._model
	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}
	Thermal_margin = {}
	LinePowerP = {}
	LinePowerQ = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	for p in instance.Phases:
		for t in instance.TimePeriods:
			SourceActivePower[p] = instance.ActivePowerAtSourceBus[p,instance.HeadBus.at(1),t].value
			SourceReactivePower[p] = instance.ReactivePowerAtSourceBus[p,instance.HeadBus.at(1),t].value

for t in instance.TimePeriods:	
	for l in sorted(instance.TransmissionLines):
		I= []
		sum_power_P = 0
		sum_power_Q = 0
		for p in instance.Phases:
			I.append(np.sqrt(instance.P_L[p,l,t].value**2 + instance.Q_L[p,l,t].value**2)/instance.V[p, instance.BusFrom[l], t].value)
			sum_power_P += instance.P_L[p,l,t].value*instance.S_Base.value/3
			sum_power_Q += instance.Q_L[p,l,t].value*instance.S_Base.value/3
		Thermal_margin[(instance.BusFrom[l], instance.BusTo[l])] = (1 - max(I)/(instance.ThermalCap[l]/instance.I_Base))*100
		LinePowerP[(instance.BusFrom[l], instance.BusTo[l])] = sum_power_P
		LinePowerQ[(instance.BusFrom[l], instance.BusTo[l])] = sum_power_Q
		# 	sum_power_P += instance.P_L[p,l,t].value
		# 	sum_power_Q += instance.Q_L[p,l,t].value
		# LinePowerMargin[(instance.BusFrom[l], instance.BusTo[l])] = round(np.sqrt(sum_power_P**2 + sum_power_Q**2),4)
			# if p == 'A':
			# 	LineActivePower_A[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
			# 	LineReactivePower_A[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
			# if p == 'B':
			# 	LineActivePower_B[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
			# 	LineReactivePower_B[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
			# if p == 'C':
			# 	LineActivePower_C[(instance.BusFrom[l], instance.BusTo[l])] = instance.P_L[p,l,t].value
			# 	LineReactivePower_C[(instance.BusFrom[l], instance.BusTo[l])] = instance.Q_L[p,l,t].value
	
	df_ThermalMargin = pd.DataFrame({
	'Margin (%)': Thermal_margin
	})
	df_ThermalMargin = df_ThermalMargin.sort_values(by='Margin (%)', ascending=True)
	df_ThermalMargin.to_csv(csv_dir/'FinalLineThermalMargin.csv', index=True)
	
	df_LinePower = pd.DataFrame({
	'Active Power (kW)': LinePowerP,
	'Reactive Power (kVAr)': LinePowerQ
	})
	df_LinePower.to_csv(csv_dir/'FinalLinePowerFlow.csv', index=True)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})
	
	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)
	df2 = df.transpose()
	df2.to_csv(csv_dir/'FinalVoltageProfile.csv', index=True)

	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)
	ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')
	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.tick_params(axis='both', labelsize=14)

	# ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')
	ax.legend()
	ax.set_xlim([-1,n])
	ax.set_ylim([0.94,1.06])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(FinalCase).pdf", bbox_inches='tight')


##########Setting Retail Price################################
	
	pi_ret_bids_notaccepted = {}
	pi_ret_offers_notaccepted = {}
	pi_ret_bids_original =[]
	pi_ret_offers_original =[]

	for k,v in Not_accepted_bids.items():
		if k not in MCBidsClear: 
			pi = round((-sum(Duals['OnlyBids']['ActivePowerBalanceDuals'].loc[(k[1], p)] for p in v[1]).item() - float(np.sqrt((1/pf**2)-1)) * sum(Duals['OnlyBids']['ReactivePowerBalanceDuals'].loc[(k[1],p)] for p in v[1]).item())/ (len(v[1])*(instance.S_Base.value/3)),1)
			pi_ret_bids_notaccepted[k] = max(LMP+C_IDSO, pi)
			pi_ret_bids_original.append(pi)
		
	for k,v in Not_accepted_offers.items():
		if k not in MCOffersClear:
			pi =  round((-sum(Duals['OnlyOffers']['ActivePowerBalanceDuals'].loc[(k[1], p)] for p in v[1]).item() - float(np.sqrt((1/pf**2)-1)) * sum(Duals['OnlyOffers']['ReactivePowerBalanceDuals'].loc[(k[1],p)] for p in v[1]).item())/ (len(v[1])*(instance.S_Base.value/3)) + M/v[-1],1)
			# pi =  round((-sum(Duals['OnlyOffers']['ActivePowerBalanceDuals'].loc[(k[1], p)] for p in v[1]).item() - float(np.sqrt((1/pf**2)-1)) * sum(Duals['OnlyOffers']['ReactivePowerBalanceDuals'].loc[(k[1],p)] for p in v[1]).item())/ (len(v[1])*(instance.S_Base.value/3)) + M/MaxP,1)
			pi_ret_offers_notaccepted[k] = min(LMP-C_IDSO, pi)
			pi_ret_offers_original.append(pi)
	
	

	pi_ret_bids_accepted = {}
	pi_ret_offers_accepted = {}
	pi_lmp = LMP

	for k,v, in Accepted_bids.items():
		pi_ret_bids_accepted[k] = pi_lmp + C_IDSO
	for k,v, in MCBidsClear.items():
		pi_ret_bids_accepted[k] = pi_lmp + C_IDSO
	for k,v, in Accepted_offers.items():
		pi_ret_offers_accepted[k] = pi_lmp - C_IDSO
	for k,v in MCOffersClear.items():
		pi_ret_offers_accepted[k] = pi_lmp - C_IDSO
	
	pi_ret = {}
	pi_der = {}
	pi_ret = pi_ret_bids_notaccepted | pi_ret_bids_accepted | pi_ret_offers_notaccepted | pi_ret_offers_accepted
	

	pi_ret_bids2 = pi_ret_bids_notaccepted 
	pi_ret_offers2 = pi_ret_offers_notaccepted 

	P_ret = {}
	
	pi_der = Accepted_bids | Accepted_offers | Not_accepted_bids | Not_accepted_offers
	P_der = Accepted_bids | Accepted_offers | Not_accepted_bids | Not_accepted_offers
	
	pi_ret = sorted(
		pi_ret.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	pi_der_bids2 = {}
	pi_der_offers2 = {}
	
	for k,v in pi_der.items():
		val = pi_der[k][2]
		pi_der[k] = val
	
	for k,v in pi_ret_bids2.items():
		pi_der_bids2[k] = pi_der[k]

	for k,v in pi_ret_offers2.items():
		pi_der_offers2[k] = pi_der[k]
	
	pi_der = sorted(
		pi_der.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	pi_ret_bids2 = sorted(
		pi_ret_bids2.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	pi_der_bids2 = sorted(
		pi_der_bids2.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	pi_ret_offers2 = sorted(
		pi_ret_offers2.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	pi_der_offers2 = sorted(
		pi_der_offers2.items(), key=lambda item: int(item[0][0].split('_')[1])
	)

	
# Plotting just retail prices for not qualified bids
	labels = [item[0][0] for item in pi_ret_bids2]
	val_ret = [item[1] for item in pi_ret_bids2]
	val_der = [item[1] for item in pi_der_bids2]
	
	n = len(labels)
	x = np.arange(len(labels))

	fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
	width = 0.7
	bars = ax.bar(x,val_ret,width, label='Retail Price (¢/kWh)', color='mediumseagreen', alpha=0.8)
	ax.bar(x,val_der,width, label='Bid Price (¢/kWh)', color= 'seagreen', alpha=1)
	ax.set_ylabel('Prices (¢/kWh)', fontsize=16, fontweight='bold')
	ax.set_xlabel('DER', fontsize=16, fontweight='bold')

	# ax.set_title('Retail vs. Bid Price of DERs (¢/kWh)', fontsize=20, fontweight='bold')
	ax.set_xticks(x)
	custom_labels = [labels[i][4:] if i % 1 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=90, ha='center', fontsize=13)
	ax.tick_params(axis='both', labelsize=16)

	for bar, price in zip(bars,pi_ret_bids_original) :
		x_left = bar.get_x()
		width = bar.get_width()
		bar_center = x_left + width/2
		bar_height = price
		line = ax.hlines(
			y= bar_height,
			xmin = x_left,
			xmax = x_left + width,
			color= 'forestgreen',
			linewidth = '2',
			zorder=3,
		)
	handles, labels = ax.get_legend_handles_labels()
	handles2, labels2 = [line], ["Qualification Price (¢/kWh)"]
	ax.legend(handles+handles2, labels+labels2)
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"BidRet.pdf", bbox_inches='tight')

	labels = [item[0][0] for item in pi_ret_offers2]
	val_ret = [item[1] for item in pi_ret_offers2]
	val_der = [item[1] for item in pi_der_offers2]
	n = len(labels)
	x = np.arange(len(labels))

	fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
	width = 0.7
	ax.bar(x,val_der,width, label='Offer Price (¢/kWh)', color='indianred', alpha=0.8)
	bars = ax.bar(x,val_ret,width, label='Retail Price (¢/kWh)', color='firebrick', alpha=1)
	ax.set_ylabel('Prices (¢/kWh)', fontsize=16, fontweight='bold')
	ax.set_xlabel('DER', fontsize=16, fontweight='bold')
	for bar, price in zip(bars,pi_ret_offers_original) :
		x_left = bar.get_x()
		width = bar.get_width()
		bar_center = x_left + width/2
		bar_height = price
		line = ax.hlines(
			y= bar_height,
			xmin = x_left,
			xmax = x_left + width,
			color= 'darkred',
			linewidth = '2',
			zorder=3,
		)
	# ax.set_title('Retail vs. Offer Price of DERs (¢/kWh)', fontsize=20, fontweight='bold')
	ax.set_xticks(x)
	custom_labels = [labels[i][4:] if i % 1 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=90, ha='center', fontsize=10)
	ax.tick_params(axis='both', labelsize=16)
	handles, labels = ax.get_legend_handles_labels()
	handles2, labels2 = [line], ["Qualification Price (¢/kWh)"]
	ax.legend(handles+handles2, labels+labels2)
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"OfferRet.pdf", bbox_inches='tight')

	for k,v in P_der.items():
		val = P_der[k][3]
		P_der[k] = val
	
	for k,v in P_der.items():
		if k in (Accepted_bids | Accepted_offers).keys():
			P_ret[k] = P_der[k] * (Accepted_bids | Accepted_offers)[k][0]

		else:
			P_ret[k] = 0.0

	P_der = sorted(
		P_der.items(), key=lambda item: int(item[0][0].split('_')[1])
	)

	P_ret = sorted(
		P_ret.items(), key=lambda item: int(item[0][0].split('_')[1])
	)
	
	#pi_ret and pi_der plot
	labels = [item [0][0] for item in pi_ret]
	val_ret = [item[1] for item in pi_ret]
	val_der = [item[1] for item in pi_der]
	n = len(labels)
	x = np.arange(len(labels))
	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True, constrained_layout=True)
	
	width = 0.7
	ax1.bar(x, val_ret, width, label='Retail Price (¢/kWh)', color='darkslategrey', alpha=0.8)
	ax1.bar(x, val_der, width, label='Bid/Offer Price (¢/kWh)', color= 'cadetblue', alpha=0.5)
	
	ax1.set_ylabel('Prices (¢/kWh)', fontsize=16, fontweight='bold')
	ax1.set_title('Retail vs. Bid/Offer Price of DERs (¢/kWh)', fontsize=16, fontweight='bold')
	ax1.set_xticks(x)
	ax1.set_xticklabels([])
	ax1.tick_params(axis='y', labelsize=13)

	ax1.legend()
	# ax1.set_xlim([-1,n])
	ax1.grid(alpha=0.15)

    #P_ret and P_der plot

	labels = [item [0][0] for item in P_ret]
	val_ret = [item[1] for item in P_ret]
	val_der = [item[1] for item in P_der]
	n = len(labels)
	x = np.arange(len(labels))
	max_val_der = max(val_der)

	width = 0.45
	ax2.bar(x, val_ret, width, label='Retail Power (kW)', color='darkslategray', alpha=0.8)
	ax2.bar(x, val_der, width, label='Bid/Offer Quantity (kW)', color= 'cadetblue', alpha=0.5)
	
	ax2.set_ylabel('Power (kW)', fontsize=16, fontweight='bold')
	ax2.set_xlabel('Buses', fontsize=16, fontweight='bold')
	ax2.set_title('Retail vs. Bid/Offer Power of DERs (kW)', fontsize=16, fontweight='bold')
	ax2.set_xticks(x)
	custom_labels = [labels[i] if i % 5 == 0 else "" for i in range(len(labels))]
	ax2.set_xticklabels(custom_labels, rotation=60, ha='right', fontsize=13)
	ax2.legend()
	ax2.set_xlim([-1,n])
	ax2.grid(alpha=0.15)
	ax2.set_yticks(np.arange(0, max_val_der + 20, 20))
	ax2.set_ylim(0, max_val_der + 10)
	ax2.tick_params(axis='both', labelsize=13)

	fig.savefig(figures_dir/"Full.pdf", bbox_inches='tight')

##################Naive Continget Case Voltage Profile ######################
	results, SolverOutcome = OPF(SOLVER, NaiveCase=True, LMP=LMP, alpha=resultsalpha)

	instance = model._model
	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})

	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)

	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.tick_params(axis='both', labelsize=14)

	ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')
	ax.legend()
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(NaiveContingent).pdf", bbox_inches='tight')
	

# #### Naive Case

	results, SolverOutcome = OPF(SOLVER, NaiveCase=True, LMP=LMP)
	##### Voltage Profile #########
	instance = model._model


	labels = list(instance.Buses)
	Voltage_A = {}
	Voltage_B = {}
	Voltage_C = {}

	for b in instance.Buses:
		for t in instance.TimePeriods:
			for p in instance.Phases: 
				if p in instance.BusPhase[b]:
					if p == 'A':
						Voltage_A[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='B':
						Voltage_B[b] = round(np.sqrt(instance.V[p, b, t].value), 5)
					if p=='C':
						Voltage_C[b] = round(np.sqrt(instance.V[p, b, t].value), 5)

	df = pd.DataFrame({
		'Phase A': Voltage_A,
		'Phase B': Voltage_B,
		'Phase C': Voltage_C
	})

	df = df.transpose()
	df = df.reindex(sorted(df.columns, key=lambda x: int(x[3:])), axis=1)

	fig, ax = plt.subplots(figsize=(12,5))
	x_values = range(len(df.columns))
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':
			mark ='.'
		ax.plot(df.loc[phase], marker=mark, label=phase)

	n = len(df.columns)
	ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=16, fontweight='bold')
	ax.set_xlabel('Buses', fontsize=16, fontweight='bold')
	# ax.set_title('Bus & Phase Voltage Magnitudes in p.u.', fontsize=16, fontweight='bold')
	ax.set_xticks(range(len(labels)))
	custom_labels = [labels[i][3:] if (i-1) % 5 == 0 else "" for i in range(len(labels))]
	ax.set_xticklabels(custom_labels, rotation=0, fontsize=14)
	ax.tick_params(axis='both', labelsize=14)

	ax.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')
	# ax.axhline(y=1.05, color='purple', linestyle='--', label='Upper Magnitude')

	axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
					bbox_to_anchor=(0.25, 0.25, 0.4, 0.3),
					bbox_transform=ax.transAxes)

	# Define the region of interest (adjust x1, x2, y1, y2 as needed)
	# Automatically find the x-range where voltage crosses 0.95
	cross_indices = []
	for idx in x_values:
		if any(df.iloc[:, idx] < 0.95):
			cross_indices.append(idx)

	if cross_indices:
		x1 = max(0, min(cross_indices) - 3)
		x2 = min(n - 1, max(cross_indices) + 2)
	else:
		x1, x2 = 0, n - 1

	# Set xlim and ylim for the inset axes
	axins.set_xlim(x1, x2)
	axins.set_ylim(0.935, 0.955)

	# Plot the same data on the inset axes with x-values
	for phase in df.index:
		if phase == 'Phase A':
			mark ='o'
		if phase == 'Phase B':
			mark ='x'
		if phase == 'Phase C':	
			mark ='.'
		axins.plot(x_values, df.loc[phase], marker=mark)
	axins.axhline(y=0.95, color='purple', linestyle='--', label='Lower Limit on Voltage Magnitude')

	# Set x-axis labels for the inset
	axins.set_xticks(range(x1, x2+1))
	
	axins.grid(alpha=0.15)
	inset_labels = [labels[i][3:] if i%3==0 else "" for i in range(x1, x2+1)]
	# axins.set_yticklabels(fontsize=6)
	axins.set_xticklabels(inset_labels, rotation=0, fontsize=12)
	axins.tick_params(axis='y', labelsize=13)
	# Ensure the background of the inset is not transparent
	axins.patch.set_alpha(1.0)

	# Adjust the rectangle connecting the inset and main plot
	mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
	ax.legend()
	ax.set_xlim([-1,n])
	ax.grid(alpha=0.15)
	fig.savefig(figures_dir/"Voltage(Naive).pdf", bbox_inches='tight')


