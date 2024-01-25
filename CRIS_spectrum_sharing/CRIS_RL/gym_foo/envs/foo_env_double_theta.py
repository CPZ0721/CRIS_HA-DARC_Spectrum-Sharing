import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
import globe
import numpy as np
import math as mt
import spectrum_sensing_MDP as ss
import torch

class FooEnv(gym.Env):
	# metadata = {'render.modes': ['human']}
	def __init__(self, LoadData = True, Train = True, MaxStep = 41):
		globe._init()
		# the location of RIS
		globe.set_value('RIS_loc', [0, 0, 20]) #[x, y, z]
		# the location of PTx
		globe.set_value('PTx_loc', [-50, -50, 10])
		# the location of STx
		globe.set_value('STx_loc', [50, -50, 5])
		# PTx antenna
		globe.set_value('PTx_M', 3)
		# STx antenna
		globe.set_value('STx_J', 2)
		# RIS elements
		globe.set_value('RIS_N', 8)
		# number of primary user (PU)
		globe.set_value('PU_K', 3)
		# number of secondary user (SU)
		globe.set_value('SU_L', 2)
		# number of subchannels
		globe.set_value('B', 3)

		#----CSI parameters----#

		# σ2 = -147dBm
		globe.set_value('AWGN', mt.pow(10, (-147/10))/1e3)
		# max transmit Power from BS is 5W
		globe.set_value('P_max', 2)
		# speed of light
		globe.set_value('c', 3 * mt.pow(10, 8))
		# the number of total time slots
		globe.set_value('t', int(MaxStep))
		# current time slot
		globe.set_value('step', 0)


		if LoadData == True:
			if Train == True:
				# Training Data
				self.PU_1_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP0.csv", delimiter=",")
				self.PU_2_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP1.csv", delimiter=",")
				self.PU_3_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP2.csv", delimiter=",")
				self.SU_1_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP3.csv", delimiter=",")
				self.SU_2_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP4.csv", delimiter=",")

				self.PU_spectrum_all = np.loadtxt("../CreateData/Train_PU_Spectrum_MDP.csv", delimiter=",")

			else:
				# Test Data
				self.PU_1_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP0.csv", delimiter=",")
				self.PU_2_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP1.csv", delimiter=",")
				self.PU_3_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP2.csv", delimiter=",")
				self.SU_1_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP3.csv", delimiter=",")
				self.SU_2_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP4.csv", delimiter=",")


				self.PU_spectrum_all = np.loadtxt("../CreateData/Test_PU_Spectrum_MDP.csv", delimiter=",")

		self.RIS_element_loc = self.create_RIS_element_location(globe.get_value('RIS_loc'), globe.get_value('RIS_N'))
		
		self.con_action_space = np.zeros((2 + globe.get_value("RIS_N") * 2,))
		self.dis_action_space = np.zeros((9,))
		self.observation_space = np.zeros((171,))
		self.max_action = 1
		self.min_action = -1

		self.max_episode_steps = 41

	def normalize(self, state):
		return (state - np.mean(state)) / np.std(state)

	def step(self, actions):
		t = globe.get_value('t')
				
		discrete_logit = actions[:self.dis_action_space.shape[0]]
		
		#prob = torch.softmax(torch.tensor(discrete_logit), dim=-1)
		
		discrete_action = torch.argmax(torch.tensor(discrete_logit), dim=-1).item()
		
		continue_action = actions[-self.con_action_space.shape[0]:]

		dis_a = int(discrete_action)

		# SU spectrum allocation (action range conversion [-1,1] -> [0,2])
		alpha_1 = np.eye(globe.get_value('B'))[dis_a // globe.get_value('B')]
		alpha_2 = np.eye(globe.get_value('B'))[dis_a % globe.get_value('B')]

		# SU transmit power (action range conversion [-1,1] -> [0,1], it means watt)
		power_1 = 0.495 * continue_action[0] + 0.505
		power_2 = 0.495 * continue_action[1] + 0.505
		# power_1 = mt.pow(10, continue_action[0]-1) # power for SU 1
		# power_2 = mt.pow(10, continue_action[1]-1) # power for SU 2

		# RIS phase shift (action range conversion [-1,1] -> [0,1])
		Theta_R_real = continue_action[2: 2 + globe.get_value('RIS_N')] 
		Theta_R_imag = continue_action[-globe.get_value('RIS_N'):] 
		step = globe.get_value('step')
		#print(14 * continue_action[0]+16, 14*continue_action[1]+16, power_1, power_2)
		reward, radio_state, total_SE, Aver_SE = self.env_state(step, alpha_1, alpha_2, power_1, power_2, Theta_R_real, Theta_R_imag)

		#if power_1 == 0 or power_2 == 0:
		#	reward = 0

		done = False
		if step == t - 1:
			done = True
   
		globe.set_value('step', int(step+1))

		return radio_state, reward, done, (total_SE, Aver_SE)

	def reset(self):
		globe.set_value('step', 0)
		
		step = globe.get_value('step')
		L_RIS = globe.get_value('RIS_loc')
		RIS_N = globe.get_value('RIS_N')
		
		# User Position
		PU_1 = self.PU_1_all[step]
		PU_2 = self.PU_2_all[step]
		PU_3 = self.PU_3_all[step]
		SU_1 = self.SU_1_all[step]
		SU_2 = self.SU_2_all[step]
		
		# PU spectrum usage
		pu_power = self.PU_spectrum_all[step][:3]
		PU_spec = self.PU_spectrum_all[step][3:]

		spectrum_usage = ss.spectrum_sensing(PU_spec)
		PU_1_spec = spectrum_usage[0][0]
		PU_2_spec = spectrum_usage[1][1]
		PU_3_spec = spectrum_usage[2][2]
		
		pu_spec = np.array([PU_1_spec, PU_2_spec, PU_3_spec])
		
		# Calculate distance between users and RIS
		distance_RIS_PU_1 = mt.sqrt(mt.pow((L_RIS[0] - PU_1[0]), 2) + mt.pow((L_RIS[1] - PU_1[1]), 2) + mt.pow((L_RIS[2] - PU_1[2]), 2))
		distance_RIS_PU_2 = mt.sqrt(mt.pow((L_RIS[0] - PU_2[0]), 2) + mt.pow((L_RIS[1] - PU_2[1]), 2) + mt.pow((L_RIS[2] - PU_2[2]), 2))
		distance_RIS_PU_3 = mt.sqrt(mt.pow((L_RIS[0] - PU_3[0]), 2) + mt.pow((L_RIS[1] - PU_3[1]), 2) + mt.pow((L_RIS[2] - PU_3[2]), 2))
		distance_RIS_SU_1 = mt.sqrt(mt.pow((L_RIS[0] - SU_1[0]), 2) + mt.pow((L_RIS[1] - SU_1[1]), 2) + mt.pow((L_RIS[2] - SU_1[2]), 2))
		distance_RIS_SU_2 = mt.sqrt(mt.pow((L_RIS[0] - SU_2[0]), 2) + mt.pow((L_RIS[1] - SU_2[1]), 2) + mt.pow((L_RIS[2] - SU_2[2]), 2))
		# scale
		location_state = np.array([distance_RIS_PU_1, distance_RIS_PU_2, distance_RIS_PU_3, distance_RIS_SU_1, distance_RIS_SU_2])
		# location_state_scale = self.normalize(location_state)
		location_state_scale = location_state/np.sum(location_state)

		# Calculate the communication channel
		self.G = self.calc_G_channel(globe.get_value('PTx_loc'), self.RIS_element_loc, globe.get_value('PTx_M'), globe.get_value('RIS_N'))
		self.G_flatten = self.G.reshape(-1,1)
		self.F = self.calc_F_channel(globe.get_value('STx_loc'), self.RIS_element_loc, globe.get_value('STx_J'), globe.get_value('RIS_N'))
		self.F_flatten = self.F.reshape(-1,1)
		self.signal_RIS_PU_1 = self.calc_H_channel(PU_1, RIS_N)
		self.signal_RIS_PU_2 = self.calc_H_channel(PU_2, RIS_N)
		self.signal_RIS_PU_3 = self.calc_H_channel(PU_3, RIS_N)
		self.signal_RIS_SU_1 = self.calc_H_channel(SU_1, RIS_N)
		self.signal_RIS_SU_2 = self.calc_H_channel(SU_2, RIS_N)

		radio_state = np.concatenate((self.G_flatten.real, self.G_flatten.imag, self.F_flatten.real, self.F_flatten.imag, self.signal_RIS_PU_1.real, self.signal_RIS_PU_1.imag, self.signal_RIS_PU_2.real, self.signal_RIS_PU_2.imag,self.signal_RIS_PU_3.real, self.signal_RIS_PU_3.imag, self.signal_RIS_SU_1.real, self.signal_RIS_SU_1.imag, self.signal_RIS_SU_2.real, self.signal_RIS_SU_2.imag), axis = 0)
		radio_state = np.squeeze(radio_state)
		
		# radio_state_scaled = self.normalize(radio_state)
		radio_state_scaled = (radio_state - np.min(radio_state)) / (np.max(radio_state) - np.min(radio_state))
		# radio_state_scaled = radio_state * 1e3

		# concatenate distance, channel, PU spectrum usage
		next_state = np.hstack((radio_state_scaled, location_state_scale))
		next_state = np.hstack((next_state, pu_power))
		next_state = np.hstack((next_state, pu_spec))

		return next_state  

	# def render(self, mode='human', close=False):
	#     pass

	def create_RIS_element_location(self, RIS_position, N):
		# RIS element space
		space= globe.get_value('c')/(1.5e9*2)    # space = lamda/2
		RIS_element_position = np.zeros(shape=(N, 3))
		for i in range(N):
			RIS_element_position[i, 0] = RIS_position[0] + i * space
			RIS_element_position[i, 1] = RIS_position[1]
			RIS_element_position[i, 2] = RIS_position[2]

		return RIS_element_position
	
	def calc_F_channel(self, L_STx, L_RIS, J, N):
		# large-scale path loss
		distance = np.linalg.norm((L_RIS.reshape(-1, 3) - L_STx),
								axis=1, keepdims=True).reshape(N, 1)   # SIZE = (N, 1)
		path_loss = np.zeros(shape=distance.shape, dtype='complex_')
		
		for i in range(distance.shape[0]):
			path_loss[i, 0] = np.sqrt(
				10**(-30/10)*np.power(distance[i, 0], -2.2))

		path_loss = np.tile(path_loss, (1, J)) # repeat the matrix => SIZE = (N, M)

		# small-scale
		small_scale = 1/np.sqrt(2)*(np.random.randn(N, J) + 1j * np.random.randn(N, J))  # SIZE = (N, M)

		F = path_loss * small_scale

		return F
	
	def calc_G_channel(self, L_PTx, L_RIS, M, N):
		# large-scale path loss
		distance = np.linalg.norm((L_RIS.reshape(-1, 3) - L_PTx),
								axis=1, keepdims=True).reshape(N, 1)   # SIZE = (N, 1)
		path_loss = np.zeros(shape=distance.shape, dtype='complex_')
		
		for i in range(distance.shape[0]):
			path_loss[i, 0] = np.sqrt(
				10**(-30/10)*np.power(distance[i, 0], -2.2))

		path_loss = np.tile(path_loss, (1, M)) # repeat the matrix => SIZE = (N, M)

		# small-scale
		small_scale = 1/np.sqrt(2)*(np.random.randn(N, M) + 1j * np.random.randn(N, M))  # SIZE = (N, M)

		G = path_loss * small_scale

		return G

	def calc_H_channel(self, users_position, N):
		RIS_position = self.RIS_element_loc
		all_path_loss = np.zeros(shape=(N, 1), dtype='complex_')

		# large-scale path loss
		distance = np.linalg.norm((RIS_position.reshape(-1, 3) - users_position),
								axis=1, keepdims=True).reshape(N, 1)   # SIZE = (N, 1)

		path_loss = np.zeros(shape=distance.shape)
		for j in range(distance.shape[0]):
			path_loss[j, 0] = np.sqrt(10**(-30/10)*np.power(distance[j, 0], -2.2))

		all_path_loss[:, 0] = path_loss.squeeze()

		# small-scale
		small_scale = 1/np.sqrt(2)*(np.random.randn(N, 1) + 1j * np.random.randn(N, 1))  # SIZE = (N, user_num)

		H = path_loss * small_scale

		return H

	def SU_SE(self, PU_1_spec, PU_2_spec, PU_3_spec, alpha_1, alpha_2, power_1, power_2, PU_1_power, PU_2_power, PU_3_power, Theta_R_real, Theta_R_imag):
		AWGN = globe.get_value('AWGN')
		RIS_N = globe.get_value('RIS_N')

		num_subchannel = globe.get_value('B')
		coefficients = np.eye(RIS_N, dtype=complex) * (Theta_R_real + 1j * Theta_R_imag)

		# received signal for SU 1
		h_s_1 = self.signal_RIS_SU_1
		SU_link_1 = np.conj(h_s_1).T @ coefficients @ self.F
		channel_SU_1 = np.dot(SU_link_1, np.conj(SU_link_1).T)
		interference = np.conj(h_s_1).T @ coefficients @ self.G
		inter_SU_1_from_PU = np.dot(interference, np.conj(interference).T)
		signal_SU_1 = channel_SU_1 * power_1
		interference_SU_1 = 0

		for i in range(num_subchannel):
			if alpha_1[i] == 1 :
				if PU_1_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * PU_1_power
				if PU_2_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * PU_2_power
				if PU_3_spec[i] == 1:
					interference_SU_1 += inter_SU_1_from_PU * PU_3_power
				if alpha_2[i] == 1:
					interference_SU_1 += channel_SU_1 * power_2

		SINR_1 = signal_SU_1/(interference_SU_1 + AWGN)

		# received signal for SU 2
		h_s_2 = self.signal_RIS_SU_2
		SU_link_2 = np.conj(h_s_2).T @ coefficients @ self.F
		channel_SU_2 = np.dot(SU_link_2, np.conj(SU_link_2).T)
		interference = np.conj(h_s_2).T @ coefficients @ self.G
		inter_SU_2_from_PU = np.dot(interference, np.conj(interference).T)
		signal_SU_2 = channel_SU_2 * power_2
		interference_SU_2 = 0
		for i in range(num_subchannel):
			if alpha_2[i] == 1 :
				if PU_1_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * PU_1_power
				if PU_2_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * PU_2_power
				if PU_3_spec[i] == 1:
					interference_SU_2 += inter_SU_2_from_PU * PU_3_power
				if alpha_1[i] == 1:
					interference_SU_2 += channel_SU_2 * power_1

		SINR_2 = signal_SU_2/(interference_SU_2 + AWGN)

		if SINR_1 > -1:
			 Aver_SE_1 = mt.log((1 + SINR_1.real), 2)
		else:
			 Aver_SE_1 = 0
		
		if SINR_2 > -1:
			 Aver_SE_2 = mt.log((1 + SINR_2.real), 2)
		else:
			 Aver_SE_2 = 0

		total_SE = Aver_SE_1 + Aver_SE_2

		return total_SE


	def capacity (self, PU_1_spec, PU_2_spec, PU_3_spec, alpha_1, alpha_2, power_1, power_2, PU_1_power, PU_2_power, PU_3_power, Theta_R_real, Theta_R_imag):

		AWGN = globe.get_value('AWGN')
		RIS_N = globe.get_value('RIS_N')
		num_subchannel = globe.get_value('B')

		coefficients = np.eye(RIS_N, dtype=complex) * (Theta_R_real + 1j * Theta_R_imag)

		# received signal for PU 1
		h_p_1 = self.signal_RIS_PU_1   
		PU_link_1 = np.conj(h_p_1).T @ coefficients @ self.G
		channel_PU_1 = np.dot(PU_link_1, np.conj(PU_link_1).T)
		interference = np.conj(h_p_1).T @ coefficients @ self.F
		inter_PU_1_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_1 = channel_PU_1 * PU_1_power
		interference_PU_1 = 0
		for i in range(num_subchannel):
			if PU_1_spec[i] == 1 :
				if alpha_1[i] == 1:
					interference_PU_1 += inter_PU_1_from_SU * power_1
				if alpha_2[i] == 1:
					interference_PU_1 += inter_PU_1_from_SU * power_2
		SINR_1 = signal_PU_1/(interference_PU_1 + AWGN)

		# received signal for PU 2
		h_p_2 = self.signal_RIS_PU_2   
		PU_link_2 = np.conj(h_p_2).T @ coefficients @ self.G
		channel_PU_2 = np.dot(PU_link_2, np.conj(PU_link_2).T)
		interference = np.conj(h_p_2).T @ coefficients @ self.F
		inter_PU_2_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_2 = channel_PU_2 * PU_2_power
		interference_PU_2 = 0
		for i in range(num_subchannel):
			if PU_2_spec[i] == 1 :
				if alpha_1[i] == 1:
					interference_PU_2 += inter_PU_2_from_SU * power_1
				if alpha_2[i] == 1:
					interference_PU_2 += inter_PU_2_from_SU * power_2

		SINR_2 = signal_PU_2/(interference_PU_2 + AWGN)

		# received signal for PU 3
		h_p_3 = self.signal_RIS_PU_3  
		PU_link_3 = np.conj(h_p_3).T @ coefficients @ self.G
		channel_PU_3 = np.dot(PU_link_3, np.conj(PU_link_3).T)
		interference = np.conj(h_p_3).T @ coefficients @ self.F
		inter_PU_3_from_SU = np.dot(interference, np.conj(interference).T)
		signal_PU_3 = channel_PU_3 * PU_3_power
		interference_PU_3 = 0
		for i in range(num_subchannel):
			if PU_3_spec[i] == 1 :
				if alpha_1[i] == 1:
					interference_PU_3 += inter_PU_3_from_SU * power_1
				if alpha_2[i] == 1:
					interference_PU_3 += inter_PU_3_from_SU * power_2

		SINR_3 = signal_PU_3/(interference_PU_3 + AWGN)

		return [SINR_1.real[0][0], SINR_2.real[0][0], SINR_3.real[0][0]]

	def env_state(self, step, alpha_1, alpha_2, power_1, power_2, Theta_R_real, Theta_R_imag):

		RIS_N = globe.get_value('RIS_N')
		L_RIS = globe.get_value('RIS_loc')

		PU_1 = self.PU_1_all[step]
		PU_2 = self.PU_2_all[step]
		PU_3 = self.PU_3_all[step]
		SU_1 = self.SU_1_all[step]
		SU_2 = self.SU_2_all[step]
		PU_1_power = self.PU_spectrum_all[step][0]
		PU_2_power = self.PU_spectrum_all[step][1]
		PU_3_power = self.PU_spectrum_all[step][2]
		PU_spec = self.PU_spectrum_all[step][3:]
		
		# spectrum sensing to get spectrum usage of t time step
		spectrum_usage = ss.spectrum_sensing(PU_spec)
		PU_1_spec = spectrum_usage[0]
		PU_2_spec = spectrum_usage[1]
		PU_3_spec = spectrum_usage[2]
		
		reward = self.SU_SE(PU_1_spec, PU_2_spec, PU_3_spec, alpha_1, alpha_2, power_1, power_2, PU_1_power, PU_2_power, PU_3_power, Theta_R_real, Theta_R_imag)
		Aver_SE= self.capacity(PU_1_spec, PU_2_spec, PU_3_spec, alpha_1, alpha_2, power_1, power_2, PU_1_power, PU_2_power, PU_3_power, Theta_R_real, Theta_R_imag)

		total_SE = reward

		reward_penalty = 1
		PU = np.array([PU_1_spec[0], PU_2_spec[1], PU_3_spec[2]])
		result = [1 if (pu == 1 and (su_1 == 1 or su_2 ==1)) or (pu == 0 and (su_1 ==1 and su_2 == 1 )) else 0 for su_1, su_2, pu in zip(alpha_1, alpha_2, PU)]
		indices = [index for index, value in enumerate(result) if value == 1]
		flag = 0
		penalty = 0
		for i in range(len(Aver_SE)):
			# deviation = abs(5 - Aver_SE[i])
			if Aver_SE[i] < 5 and Aver_SE[i] != 0:
				flag = 1
				penalty += -(5-Aver_SE[i])
		if flag == 1:
			reward += penalty

		#if np.sum(PU) == 0 or np.sum(PU) == 1:
		#	normalize_reward = reward / 40
		#elif np.sum(PU) == 2 :
		#	normalize_reward = reward / 25
		#else:
		#	normalize_reward = reward / 5
    
		#reward = normalize_reward * reward_penalty
 
 
		if step < globe.get_value('t')-1:
			L_RIS = globe.get_value('RIS_loc')

			PU_1 = self.PU_1_all[step+1]
			PU_2 = self.PU_2_all[step+1]
			PU_3 = self.PU_3_all[step+1]
			SU_1 = self.SU_1_all[step+1]
			SU_2 = self.SU_2_all[step+1]
			PU_1_power = self.PU_spectrum_all[step+1][0]
			PU_2_power = self.PU_spectrum_all[step+1][1]
			PU_3_power = self.PU_spectrum_all[step+1][2]
			PU_spec = self.PU_spectrum_all[step+1][3:]
		else:
			L_RIS = globe.get_value('RIS_loc')

			PU_1 = self.PU_1_all[step]
			PU_2 = self.PU_2_all[step]
			PU_3 = self.PU_3_all[step]
			SU_1 = self.SU_1_all[step]
			SU_2 = self.SU_2_all[step]
			PU_1_power = self.PU_spectrum_all[step][0]
			PU_2_power = self.PU_spectrum_all[step][1]
			PU_3_power = self.PU_spectrum_all[step][2]
			PU_spec = self.PU_spectrum_all[step][3:]

		# spectrum sensing, get the new sensing result
		spectrum_usage = ss.spectrum_sensing(PU_spec)
		PU_1_spec = spectrum_usage[0][0]
		PU_2_spec = spectrum_usage[1][1]
		PU_3_spec = spectrum_usage[2][2]

		pu_spec = np.array([PU_1_spec, PU_2_spec, PU_3_spec])

		distance_RIS_PU_1 = mt.sqrt(mt.pow((L_RIS[0] - PU_1[0]), 2) + mt.pow((L_RIS[1] - PU_1[1]), 2) + mt.pow((L_RIS[2] - PU_1[2]), 2))
		distance_RIS_PU_2 = mt.sqrt(mt.pow((L_RIS[0] - PU_2[0]), 2) + mt.pow((L_RIS[1] - PU_2[1]), 2) + mt.pow((L_RIS[2] - PU_2[2]), 2))
		distance_RIS_PU_3 = mt.sqrt(mt.pow((L_RIS[0] - PU_3[0]), 2) + mt.pow((L_RIS[1] - PU_3[1]), 2) + mt.pow((L_RIS[2] - PU_3[2]), 2))
		distance_RIS_SU_1 = mt.sqrt(mt.pow((L_RIS[0] - SU_1[0]), 2) + mt.pow((L_RIS[1] - SU_1[1]), 2) + mt.pow((L_RIS[2] - SU_1[2]), 2))
		distance_RIS_SU_2 = mt.sqrt(mt.pow((L_RIS[0] - SU_2[0]), 2) + mt.pow((L_RIS[1] - SU_2[1]), 2) + mt.pow((L_RIS[2] - SU_2[2]), 2))
		
		# scale
		location_state = np.array([distance_RIS_PU_1, distance_RIS_PU_2, distance_RIS_PU_3, distance_RIS_SU_1, distance_RIS_SU_2])
		# location_state_scale = self.normalize(location_state)
		location_state_scale = location_state/np.sum(location_state)

		self.G = self.calc_G_channel(globe.get_value('PTx_loc'), self.RIS_element_loc, globe.get_value('PTx_M'), globe.get_value('RIS_N'))
		G = self.G.reshape(-1,1)
		# print("G", G)
		self.F = self.calc_F_channel(globe.get_value('STx_loc'), self.RIS_element_loc, globe.get_value('STx_J'), globe.get_value('RIS_N'))
		F = self.F.reshape(-1,1)
		# print("F", F)
		self.signal_RIS_PU_1 = self.calc_H_channel(PU_1, RIS_N)
		self.signal_RIS_PU_2 = self.calc_H_channel(PU_2, RIS_N)
		self.signal_RIS_PU_3 = self.calc_H_channel(PU_3, RIS_N)
		self.signal_RIS_SU_1 = self.calc_H_channel(SU_1, RIS_N)
		self.signal_RIS_SU_2 = self.calc_H_channel(SU_2, RIS_N)

		radio_state = np.concatenate((G.real, G.imag, F.real, F.imag, self.signal_RIS_PU_1.real, self.signal_RIS_PU_1.imag, self.signal_RIS_PU_2.real, self.signal_RIS_PU_2.imag,self.signal_RIS_PU_3.real, self.signal_RIS_PU_3.imag, self.signal_RIS_SU_1.real, self.signal_RIS_SU_1.imag, self.signal_RIS_SU_2.real, self.signal_RIS_SU_2.imag), axis = 0)
		radio_state = np.squeeze(radio_state)
		# scale
		# radio_state_scaled = self.normalize(radio_state)
		radio_state_scaled = (radio_state - np.min(radio_state)) / (np.max(radio_state) - np.min(radio_state))
		# radio_state_scaled = radio_state * 1e3
						
		pu_power = np.hstack((PU_1_power, PU_2_power, PU_3_power))

		next_state = np.hstack((radio_state_scaled, location_state_scale))
		next_state = np.hstack((next_state, pu_power))
		next_state = np.hstack((next_state, pu_spec))


		return reward, next_state, total_SE, Aver_SE

