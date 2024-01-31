import numpy as np
import bb_minlp_solve as bb
import math as mt
from pyomo.environ import *
import pandas as pd
import argparse

def load_translate(state_array, N, M, J, count):
	# recover the channel
	G = np.empty((N,M), dtype = np.complex128)
	G.real = state_array[count, :N*M].reshape(N,M)
	G.imag = state_array[count, N*M: 2*N*M].reshape(N,M)

	F = np.empty((N,J), dtype = np.complex128)
	F.real = state_array[count, 2*N*M:2*N*M+N*J].reshape(N,J)
	F.imag = state_array[count, 2*N*M+N*J: 2*N*(M+J)].reshape(N,J)
	
	signal_RIS_PU_1 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_PU_1.real = state_array[count, 2*N*(M+J):2*N*(M+J) + N].reshape(N,1)
	signal_RIS_PU_1.imag = state_array[count, 2*N*(M+J) + N: 2*N*(M+J) + 2*N].reshape(N,1)

	signal_RIS_PU_2 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_PU_2.real = state_array[count, 2*N*(M+J) + 2*N:2*N*(M+J) + 3*N].reshape(N,1)
	signal_RIS_PU_2.imag = state_array[count, 2*N*(M+J) + 3*N: 2*N*(M+J) + 4*N].reshape(N,1)

	signal_RIS_PU_3 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_PU_3.real = state_array[count, 2*N*(M+J) + 4*N:2*N*(M+J) + 5*N].reshape(N,1)
	signal_RIS_PU_3.imag = state_array[count, 2*N*(M+J) + 5*N: 2*N*(M+J) + 6*N].reshape(N,1)

	signal_RIS_SU_1 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_SU_1.real = state_array[count, 2*N*(M+J) + 6*N:2*N*(M+J) + 7*N].reshape(N,1)
	signal_RIS_SU_1.imag = state_array[count, 2*N*(M+J) + 7*N: 2*N*(M+J) + 8*N].reshape(N,1)

	signal_RIS_SU_2 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_SU_2.real = state_array[count, 2*N*(M+J) + 8*N:2*N*(M+J) + 9*N].reshape(N,1)
	signal_RIS_SU_2.imag = state_array[count, 2*N*(M+J) + 9*N: 2*N*(M+J) + 10*N].reshape(N,1)

	return G, F, signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--RIS_N", default=8, type=int, help='CRIS_reflective_element')
	args = parser.parse_args()

	# PTx antenna
	M = 3
	# STx antenna
	J = 2
	# RIS elements
	N = args.RIS_N
	# number of primary user (PU)
	K = 3
	# number of secondary user (SU)
	L = 2
	# number of subchannels
	num_subchannel = 3

	#----CSI parameters----#
	noise =  mt.pow(10, (-147/10))/1e3

	# min SINR of PU is 5 (7 dB)
	var_pi = 5

	# num of data
	num_data = 10
	
	maxStep = 41

	results = np.empty((num_data, maxStep, 9))

	state_array = np.load(f"channel_data/evaluation_state_{N}.npy")

	avg_SE = 0.
	num_infeasible = 0

	for i in range(num_data):
		all_SE = 0
		for num_step in range(maxStep):
			# channel
			G, F, signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2 = load_translate(state_array[i,:,:], N, M , J, num_step)
			H = np.concatenate((signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2), axis=1)
			
			# primary message
			pu_power = state_array[i, num_step,-2*K:-K]
			pu_freq_usage = state_array[i, num_step,-K:]

			spectrum_usage = np.diag(pu_freq_usage)

			# bb slove the optimization problem
			objective, power, theta, usage, solver_state, used_time = bb.minlp_solve(
			K, L, N, M, J, var_pi, num_subchannel, pu_power, spectrum_usage , noise, G, F, H)

			alpha_1 = usage[:num_subchannel]
			alpha_2 = usage[-num_subchannel:]

			# recrod the optimization result
			if solver_state != "infeasible":

				total_SE = objective
				best_params = np.array([pu_freq_usage[0], pu_freq_usage[1], pu_freq_usage[2], power[0], power[1], np.where(alpha_1 == 1)[0][0], 
							np.where(alpha_2 == 1)[0][0], used_time, total_SE]) 
			else:
				num_infeasible += 1
				total_SE = 0
				best_params = np.array([0, 0, 0, 0, 0, 0, 0, used_time, 0]) 

			# put the parameters into results
			results[i, num_step, :] = best_params

			all_SE += total_SE
			
			print("Step： " + str(num_step) + " is done. The reward is " + str(total_SE))

		avg_SE += all_SE / (maxStep - num_infeasible)

	# average the results
	results = np.mean(results, axis = 0)
	df = pd.DataFrame(results)

	# save DataFrame
	df.to_csv(f"BB_result_{N}.csv", index=False, header=False)

	avg_SE /= num_data

	print("rewards：" + str(avg_SE))

if __name__== "__main__" :
	main()


