import numpy as np
import time
import math as mt
import pandas as pd

class CommunicationProblemAnnealer:
	def __init__(self, state, K, L, N, M, J, R_min_p, num_subchannel, pu_power, spectrum_usage, noise, G, H, F):

		# parameter setting
		self.state = state
		self.alpha_1 = np.eye(num_subchannel)[int(state[0])]
		self.alpha_2 = np.eye(num_subchannel)[int(state[1])]
		self.power_1 = state[2]
		self.power_2 = state[3]
		self.Theta_R = state[4:]
		self.K = K
		self.L = L
		self.N = N
		self.M = M
		self.J = J
		self.R_min_p = R_min_p
		self.num_subchannel = num_subchannel
		self.noise = noise
		self.PU_1_power = pu_power[0]
		self.PU_2_power = pu_power[1]
		self.PU_3_power = pu_power[2]
		self.PU_1_spec = spectrum_usage[0]
		self.PU_2_spec = spectrum_usage[1]
		self.PU_3_spec = spectrum_usage[2]
		self.G = G
		self.F = F
		self.signal_RIS_PU_1 = H[:, 0]
		self.signal_RIS_PU_2 = H[:, 1]
		self.signal_RIS_PU_3 = H[:, 2]
		self.signal_RIS_SU_1 = H[:, 3]
		self.signal_RIS_SU_2 = H[:, 4]

	def move(self):
		# choose the change parameter randomly
		index = np.random.randint(0, len(self.state) - 1)

		# spectrum allocation
		if index < self.L:
			spec = (self.state[index] + 1) % (self.L + 1)
			self.state[index] = spec
			if index == 0:
				self.alpha_1 = np.eye(self.num_subchannel)[int(spec)]
			else:
				self.alpha_2 = np.eye(self.num_subchannel)[int(spec)]

		# power control
		elif index >= self.L and index < 2 * self.L:
			power = np.random.uniform(0.01, 1.0)
			self.state[index] = power
			if index == self.L:
				self.power_1 = power
			else:
				self.power_2 = power
		
		# phase shift adjust
		else:
			phase_shift = np.random.uniform(0, 1) * 2 * np.pi
			self.state[index] = phase_shift
			self.Theta_R[index - 2 * self.L] = phase_shift

	def calculate_SE(self):
		coefficients = np.diag(np.exp(1j*self.Theta_R))

		# received signal for SU 1
		h_s_1 = self.signal_RIS_SU_1
		SU_link_1 = np.conj(h_s_1).T @ coefficients @ self.F
		channel_SU_1 = np.dot(SU_link_1, np.conj(SU_link_1).T)
		interference = np.conj(h_s_1).T @ coefficients @ self.G
		inter_SU_1_from_PU = np.dot(interference, np.conj(interference).T)
		signal_SU_1 = channel_SU_1 * self.power_1
		interference_SU_1 = 0

		for i in range(self.num_subchannel):
			if self.alpha_1[i] == 1 :
				if self.PU_1_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * self.PU_1_power
				if self.PU_2_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * self.PU_2_power
				if self.PU_3_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * self.PU_3_power
				if self.alpha_2[i] == 1:
					interference_SU_1 += channel_SU_1 * self.power_2

		SINR_1 = signal_SU_1/(interference_SU_1 + self.noise)
		
		# received signal for SU 2
		h_s_2 = self.signal_RIS_SU_2
		SU_link_2 = np.conj(h_s_2).T @ coefficients @ self.F
		channel_SU_2 = np.dot(SU_link_2, np.conj(SU_link_2).T)
		interference = np.conj(h_s_2).T @ coefficients @ self.G
		inter_SU_2_from_PU = np.dot(interference, np.conj(interference).T)
		signal_SU_2 = channel_SU_2 * self.power_2
		interference_SU_2 = 0
		for i in range(self.num_subchannel):
			if self.alpha_2[i] == 1 :
				if self.PU_1_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * self.PU_1_power
				if self.PU_2_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * self.PU_2_power
				if self.PU_3_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * self.PU_3_power
				if self.alpha_1[i] == 1:
					interference_SU_2 += channel_SU_2 * self.power_1

		SINR_2 = signal_SU_2/(interference_SU_2 + self.noise)

		if SINR_1.real > -1:
			Aver_SE_1 = mt.log((1 + SINR_1.real), 2)
		else:
			Aver_SE_1 = 0
		
		if SINR_2.real > -1:
			Aver_SE_2 = mt.log((1 + SINR_2.real), 2)
		else:
			Aver_SE_2 = 0

		total_SE = Aver_SE_1 + Aver_SE_2

		return total_SE

	def PU_constraint(self):
		coefficients = np.diag(np.exp(1j*self.Theta_R))

		# received signal for PU 1
		h_p_1 = self.signal_RIS_PU_1   
		PU_link_1 = np.conj(h_p_1).T @ coefficients @ self.G
		channel_PU_1 = np.dot(PU_link_1, np.conj(PU_link_1).T)
		interference = np.conj(h_p_1).T @ coefficients @ self.F
		inter_PU_1_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_1 = channel_PU_1 * self.PU_1_power
		interference_PU_1 = 0
		for i in range(self.num_subchannel):
			if self.PU_1_spec[i] == 1 :
				if self.alpha_1[i] == 1:
					interference_PU_1 += inter_PU_1_from_SU * self.power_1
				if self.alpha_2[i] == 1:
					interference_PU_1 += inter_PU_1_from_SU * self.power_2
		SINR_1 = signal_PU_1/(interference_PU_1 + self.noise)

		# received signal for PU 2
		h_p_2 = self.signal_RIS_PU_2   
		PU_link_2 = np.conj(h_p_2).T @ coefficients @ self.G
		channel_PU_2 = np.dot(PU_link_2, np.conj(PU_link_2).T)
		interference = np.conj(h_p_2).T @ coefficients @ self.F
		inter_PU_2_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_2 = channel_PU_2 * self.PU_2_power
		interference_PU_2 = 0
		for i in range(self.num_subchannel):
			if self.PU_2_spec[i] == 1 :
				if self.alpha_1[i] == 1:
					interference_PU_2 += inter_PU_2_from_SU  * self.power_1
				if self.alpha_2[i] == 1:
					interference_PU_2 += inter_PU_2_from_SU  * self.power_2

		SINR_2 = signal_PU_2/(interference_PU_2 + self.noise)

		# received signal for PU 3
		h_p_3 = self.signal_RIS_PU_3  
		PU_link_3 = np.conj(h_p_3).T @ coefficients @ self.G
		channel_PU_3 = np.dot(PU_link_3, np.conj(PU_link_3).T)
		interference = np.conj(h_p_3).T @ coefficients @ self.F
		inter_PU_3_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_3 = channel_PU_3 * self.PU_3_power
		interference_PU_3 = 0
		for i in range(self.num_subchannel):
			if self.PU_3_spec[i] == 1 :
				if self.alpha_1[i] == 1:
					interference_PU_3 += inter_PU_3_from_SU * self.power_1
				if self.alpha_2[i] == 1:
					interference_PU_3 += inter_PU_3_from_SU * self.power_2

		SINR_3 = signal_PU_3/(interference_PU_3 + self.noise)

		return [SINR_1.real, SINR_2.real, SINR_3.real]

	def energy(self):
		capacity = self.calculate_SE()
		Aver_SE = self.PU_constraint()
		penalty = 0
		for i in range(len(Aver_SE)):
			# check if the minimum SINR is satisfied
			if Aver_SE[i] < self.R_min_p and Aver_SE[i] != 0:
				penalty += 50 * (self.R_min_p - Aver_SE[i])
				
		if penalty == 0:	
			return -capacity
		else:	
			return penalty

def simulated_annealing(initial_state, K, L, N, M, J, R_min_p, num_subchannel, pu_power, spectrum_usage, noise, G, H, F):
	# init the sa solver
	annealer = CommunicationProblemAnnealer(initial_state, K, L, N, M, J, R_min_p, num_subchannel, pu_power, spectrum_usage, noise, G, H, F)
	
	# parameter setting
	temperature = 15.0
	min_temperature = 0.1
	cooling_rate = 0.5
	iterations = 150

	current_state = annealer.state
	best_state = current_state.copy()
	current_energy = annealer.energy()
	best_energy = current_energy

	# sa algorithm
	while temperature > min_temperature:
		# move times each temperature
		for _ in range(iterations):

			annealer.move()
			new_energy = annealer.energy()
			delta_energy = new_energy - current_energy

			# better solution after move (delta_energy < 0)
			# accept the new solution with probability (np.random.rand() < np.exp(-delta_energy / temperature)) 
			if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
				current_state = annealer.state
				current_energy = new_energy

				# find out the better solution
				if current_energy < best_energy:
					best_state = current_state
					best_energy = current_energy

		# update temerature
		temperature *= cooling_rate

	return best_state, -best_energy

def load(state_array, N, M, J, count):
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
	# PTx antenna
	M = 3
	# STx antenna
	J = 2
	# RIS elements
	N = 8
	# number of primary user (PU)
	K = 3
	# number of secondary user (SU)
	L = 2
	# number of subchannels
	num_subchannel = 3

	#----CSI parameters----#
	noise =  mt.pow(10, (-147/10))/1e3

	# min SINR of PU is 5 (7dB)
	R_min_p = 5

	maxStep = 41
	num_data = 10

	results = np.empty((num_data, maxStep, 9))

	state_array = np.load(f"channel_data/evaluation_state_{N}.npy")
	avg_SE = 0.
	for i in range(num_data):
		for num_step in range(maxStep):
			# channel
			G, F, signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2 = load(state_array[i,:,:], N, M , J, num_step)
			H = np.concatenate((signal_RIS_PU_1, signal_RIS_PU_2, signal_RIS_PU_3, signal_RIS_SU_1, signal_RIS_SU_2), axis=1)
			
			# primary message
			pu_power = state_array[i, num_step,-6:-3]
			pu_freq_usage = state_array[i, num_step,-3:]
			
			spectrum_usage = np.diag(pu_freq_usage)

			# create initial solution(initial status)
			initial_state = np.zeros(2 * L + N)

			start_time = time.time()
			# create sa solver
			best_state, best_energy = simulated_annealing(initial_state, K, L, N, M, J, R_min_p, num_subchannel, pu_power, spectrum_usage, noise, G, H, F)
			end_time = time.time()
			
			# computing time
			used_time = end_time - start_time

			total_SE = best_energy
			best_params = np.array([pu_freq_usage[0], pu_freq_usage[1], pu_freq_usage[2],best_state[2],best_state[3],best_state[0],best_state[1],used_time, total_SE])
			results[i,num_step,:] = best_params
			avg_SE += total_SE

			print("Step： " + str(num_step) + " is done. The reward is " + str(total_SE) + ". The used time is " + str(used_time))

	results = np.mean(results, axis = 0)

	# save the csv file
	df = pd.DataFrame(results)
	df.to_csv(f"sa_result_{N}.csv", index=False, header=False)

	avg_SE /= num_data

	print("rewards：" + str(avg_SE))

if __name__ == '__main__':
	np.random.seed(0)
	main()

