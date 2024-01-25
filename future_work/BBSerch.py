import numpy as np
import bb_minlp_solve as bb
import math as mt
from pyomo.environ import *
import test as test

# restore the csv file to original channel
def load_translate(state_array, N, M, J, count):

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

	signal_RIS_SU_3 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_SU_3.real = state_array[count, 2*N*(M+J) + 10*N:2*N*(M+J) + 11*N].reshape(N,1)
	signal_RIS_SU_3.imag = state_array[count, 2*N*(M+J) + 11*N: 2*N*(M+J) + 12*N].reshape(N,1)

	signal_RIS_SU_4 = np.empty((N,1), dtype = np.complex128)
	signal_RIS_SU_4.real = state_array[count, 2*N*(M+J) + 12*N:2*N*(M+J) + 13*N].reshape(N,1)
	signal_RIS_SU_4.imag = state_array[count, 2*N*(M+J) + 13*N: 2*N*(M+J) + 14*N].reshape(N,1)


	return G, F, signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2, signal_RIS_SU_3, signal_RIS_SU_4

def main():
	
	# PTx antenna
	M = 3
	# STx antenna
	J = 2
	# RIS elements
	N = 12
	# number of secondary user (SU)
	L = 4
	# number of subchannels
	num_subchannel = 6

	#----CSI parameters----#
	bandwidth = 1e6
	data_num = 10
	noise =  mt.pow(10, (-147/10))/1e3
	

	state_array = np.load(f"4SU_evaluation_state_{N}.npy")

	result = 0
	num_infeasible = 0

	for i in range(data_num):
		# Use first step to evaluate
		# recover thr channel
		_, F, _, _, _, signal_RIS_SU_1, signal_RIS_SU_2, signal_RIS_SU_3,signal_RIS_SU_4= load_translate(state_array[i,:,:], N, M , J, 0)
		
		# combine the H channel
		H = np.concatenate((signal_RIS_SU_1, signal_RIS_SU_2, signal_RIS_SU_3,signal_RIS_SU_4), axis=1)
		# H = signal_RIS_SU_1

		# branch and bound slove the optimization problem
		objective, power, theta, usage, solver_state, used_time = bb.minlp_solve(L, N, J, num_subchannel, noise, F, H, bandwidth)
		

		# if problem is feasible
		if solver_state != "infeasible":

			total_capacity = objective
		
		# if problem is infeasible
		else:
			num_infeasible += 1
			total_capacity = 0
		
		result += total_capacity

		print("Result： " + str(total_capacity))

	# Average the result
	print(result/(data_num - num_infeasible))


if __name__ == '__main__':
	np.random.seed(100)
	main()
