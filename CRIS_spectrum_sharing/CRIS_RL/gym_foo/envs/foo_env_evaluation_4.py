import gym
from gym import error, spaces, utils
from gym.utils import seeding
import globe
import numpy as np
import math as mt
import spectrum_sensing_MDP as ss

class FooEnv(gym.Env):
	metadata = {'render.modes': ['human']}
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
		globe.set_value('RIS_N', 12)
		# number of primary user (PU)
		globe.set_value('PU_K', 3)
		# number of secondary user (SU)
		globe.set_value('SU_L', 4)
		# number of subchannels
		globe.set_value('B', 3)

		#----CSI parameters----#

		# σ2 = -147dBm
		globe.set_value('AWGN', mt.pow(10, (-147/10))/1e3)
		# max transmit Power from BS is 5W
		globe.set_value('P_max', 5)
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
				self.SU_3_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP1.csv", delimiter=",")
				self.SU_4_all = np.loadtxt("../CreateData/Train_Trajectory_User_MDP2.csv", delimiter=",")

				self.PU_spectrum_all = np.loadtxt("../CreateData/Train_PU_Spectrum_MDP.csv", delimiter=",")

			else:
				# Test Data
				self.PU_1_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP0.csv", delimiter=",")
				self.PU_2_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP1.csv", delimiter=",")
				self.PU_3_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP2.csv", delimiter=",")
				self.SU_1_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP3.csv", delimiter=",")
				self.SU_2_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP4.csv", delimiter=",")
				self.SU_3_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP1.csv", delimiter=",")
				self.SU_4_all = np.loadtxt("../CreateData/Test_Trajectory_User_MDP2.csv", delimiter=",")


				self.PU_spectrum_all = np.loadtxt("../CreateData/Test_PU_Spectrum_MDP.csv", delimiter=",")

		self.RIS_element_loc = self.create_RIS_element_location(globe.get_value('RIS_loc'), globe.get_value('RIS_N'))
		self.observation_space = np.zeros((312,))
		self.max_episode_steps = 41
		self.steps = 0

	def exhaustive_step(self, steps):
	   
		next_state, next_state_scaled = self.env_state(steps)

		return next_state, next_state_scaled 

	def reset(self):
		
		step = globe.get_value('step')
		L_RIS = globe.get_value('RIS_loc')
		RIS_N = globe.get_value('RIS_N')
		
		# User Position
		PU_1 = self.PU_1_all[step]
		PU_2 = self.PU_2_all[step]
		PU_3 = self.PU_3_all[step]
		SU_1 = self.SU_1_all[step]
		SU_2 = self.SU_2_all[step]
		SU_3 = self.SU_3_all[step]
		SU_4 = self.SU_4_all[step]
		
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
		distance_RIS_SU_3 = mt.sqrt(mt.pow((L_RIS[0] - SU_3[0]), 2) + mt.pow((L_RIS[1] - SU_3[1]), 2) + mt.pow((L_RIS[2] - SU_3[2]), 2))
		distance_RIS_SU_4 = mt.sqrt(mt.pow((L_RIS[0] - SU_4[0]), 2) + mt.pow((L_RIS[1] - SU_4[1]), 2) + mt.pow((L_RIS[2] - SU_4[2]), 2))
		# scale
		location_state = np.array([distance_RIS_PU_1, distance_RIS_PU_2, distance_RIS_PU_3, distance_RIS_SU_1, distance_RIS_SU_2,distance_RIS_SU_3,distance_RIS_SU_4])
		location_state_scale = location_state/np.sum(location_state)

		# Calculate the communication channel
		self.G = self.calc_G_channel(globe.get_value('PTx_loc'), self.RIS_element_loc, globe.get_value('PTx_M'), globe.get_value('RIS_N'))
		self.G_flatten = self.G.reshape(-1,1)
		print("G",self.G)
		print(self.G.shape)
		self.F = self.calc_F_channel(globe.get_value('STx_loc'), self.RIS_element_loc, globe.get_value('STx_J'), globe.get_value('RIS_N'))
		self.F_flatten = self.F.reshape(-1,1)
		print("F", self.F)
		print(self.F.shape)
		print("==================")
		self.signal_RIS_PU_1 = self.calc_H_channel(PU_1, RIS_N)
		self.signal_RIS_PU_2 = self.calc_H_channel(PU_2, RIS_N)
		self.signal_RIS_PU_3 = self.calc_H_channel(PU_3, RIS_N)
		self.signal_RIS_SU_1 = self.calc_H_channel(SU_1, RIS_N)
		self.signal_RIS_SU_2 = self.calc_H_channel(SU_2, RIS_N)
		self.signal_RIS_SU_3 = self.calc_H_channel(SU_3, RIS_N)
		self.signal_RIS_SU_4 = self.calc_H_channel(SU_4, RIS_N)
		
		radio_state = np.concatenate((self.G_flatten.real, self.G_flatten.imag, self.F_flatten.real, self.F_flatten.imag, self.signal_RIS_PU_1.real, self.signal_RIS_PU_1.imag, self.signal_RIS_PU_2.real, self.signal_RIS_PU_2.imag,self.signal_RIS_PU_3.real, self.signal_RIS_PU_3.imag, self.signal_RIS_SU_1.real, self.signal_RIS_SU_1.imag, self.signal_RIS_SU_2.real, self.signal_RIS_SU_2.imag, self.signal_RIS_SU_3.real, self.signal_RIS_SU_3.imag, self.signal_RIS_SU_4.real, self.signal_RIS_SU_4.imag), axis = 0)
		radio_state = np.squeeze(radio_state)
		radio_state_scaled = (radio_state - np.min(radio_state)) / (np.max(radio_state) - np.min(radio_state))

		next_state = np.hstack((radio_state, location_state))
		print(next_state.shape)
		next_state = np.hstack((next_state, pu_power))
		print(next_state.shape)
		next_state = np.hstack((next_state, pu_spec))
		print(next_state.shape)

		next_state_scaled = np.hstack((radio_state_scaled, location_state_scale))
		next_state_scaled = np.hstack((next_state_scaled, pu_power))
		next_state_scaled = np.hstack((next_state_scaled, pu_spec))

		return next_state, next_state_scaled


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
	
	def calc_G_channel(self, L_BS, L_RIS, M, N):
		# large-scale path loss
		distance = np.linalg.norm((L_RIS.reshape(-1, 3) - L_BS),
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


	def env_state(self, step):
		RIS_N = globe.get_value('RIS_N')
		L_RIS = globe.get_value('RIS_loc')
		if step < globe.get_value('t')-1:

			PU_1 = self.PU_1_all[step+1]
			PU_2 = self.PU_2_all[step+1]
			PU_3 = self.PU_3_all[step+1]
			SU_1 = self.SU_1_all[step+1]
			SU_2 = self.SU_2_all[step+1]
			SU_3 = self.SU_3_all[step+1]
			SU_4 = self.SU_4_all[step+1]
			PU_1_power = self.PU_spectrum_all[step+1][0]
			PU_2_power = self.PU_spectrum_all[step+1][1]
			PU_3_power = self.PU_spectrum_all[step+1][2]
			PU_spec = self.PU_spectrum_all[step+1][3:]
		else:

			PU_1 = self.PU_1_all[step]
			PU_2 = self.PU_2_all[step]
			PU_3 = self.PU_3_all[step]
			SU_1 = self.SU_1_all[step]
			SU_2 = self.SU_2_all[step]
			SU_3 = self.SU_3_all[step]
			SU_4 = self.SU_4_all[step]
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
		distance_RIS_SU_3 = mt.sqrt(mt.pow((L_RIS[0] - SU_3[0]), 2) + mt.pow((L_RIS[1] - SU_3[1]), 2) + mt.pow((L_RIS[2] - SU_3[2]), 2))
		distance_RIS_SU_4 = mt.sqrt(mt.pow((L_RIS[0] - SU_4[0]), 2) + mt.pow((L_RIS[1] - SU_4[1]), 2) + mt.pow((L_RIS[2] - SU_4[2]), 2))
		
		# scale
		location_state = np.array([distance_RIS_PU_1, distance_RIS_PU_2, distance_RIS_PU_3, distance_RIS_SU_1, distance_RIS_SU_2, distance_RIS_SU_3, distance_RIS_SU_4])
		location_state_scale = location_state/np.sum(location_state)
		
		self.G = self.calc_G_channel(globe.get_value('PTx_loc'), self.RIS_element_loc, globe.get_value('PTx_M'), globe.get_value('RIS_N'))
		G = self.G.reshape(-1,1)
		# print("G", G)
		self.F = self.calc_F_channel(globe.get_value('STx_loc'), self.RIS_element_loc, globe.get_value('STx_J'), globe.get_value('RIS_N'))
		F = self.F.reshape(-1,1)
				
		self.signal_RIS_PU_1 = self.calc_H_channel(PU_1, RIS_N)
		self.signal_RIS_PU_2 = self.calc_H_channel(PU_2, RIS_N)
		self.signal_RIS_PU_3 = self.calc_H_channel(PU_3, RIS_N)
		self.signal_RIS_SU_1 = self.calc_H_channel(SU_1, RIS_N)
		self.signal_RIS_SU_2 = self.calc_H_channel(SU_2, RIS_N)
		self.signal_RIS_SU_3 = self.calc_H_channel(SU_3, RIS_N)
		self.signal_RIS_SU_4 = self.calc_H_channel(SU_4, RIS_N)

		radio_state = np.concatenate((G.real, G.imag, F.real, F.imag, self.signal_RIS_PU_1.real, self.signal_RIS_PU_1.imag, self.signal_RIS_PU_2.real, self.signal_RIS_PU_2.imag,self.signal_RIS_PU_3.real, self.signal_RIS_PU_3.imag, self.signal_RIS_SU_1.real, self.signal_RIS_SU_1.imag, self.signal_RIS_SU_2.real, self.signal_RIS_SU_2.imag, self.signal_RIS_SU_3.real, self.signal_RIS_SU_3.imag, self.signal_RIS_SU_4.real, self.signal_RIS_SU_4.imag), axis = 0)
		radio_state = np.squeeze(radio_state)

		radio_state_scaled = (radio_state - np.min(radio_state)) / (np.max(radio_state) - np.min(radio_state))

		pu_power = np.hstack((PU_1_power, PU_2_power, PU_3_power))

		next_state = np.hstack((radio_state, location_state))
		next_state = np.hstack((next_state, pu_power))
		next_state = np.hstack((next_state, pu_spec))

		next_state_scaled = np.hstack((radio_state_scaled, location_state_scale))
		next_state_scaled = np.hstack((next_state_scaled, pu_power))
		next_state_scaled = np.hstack((next_state_scaled, pu_spec))
		
		return next_state, next_state_scaled

