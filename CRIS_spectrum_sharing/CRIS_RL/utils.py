import numpy as np
import torch

# replay buffer
class ReplayBuffer(object):
	def __init__(self, state_dim, discrete_action, continue_action, device, max_size=int(1e7)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.discrete_action = np.zeros((max_size, discrete_action))
		self.continue_action = np.zeros((max_size, continue_action))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device

	def add(self, state, discrete_action, continue_action, next_state, reward, done):
		self.state[self.ptr] = state
		self.discrete_action[self.ptr] = discrete_action
		self.continue_action[self.ptr] = continue_action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	# random choose the `batch_size` data
	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.discrete_action[ind]).to(self.device),
			torch.FloatTensor(self.continue_action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
