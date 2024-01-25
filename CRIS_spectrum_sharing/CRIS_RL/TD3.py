import copy
import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))

		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l6 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		min_action,
		max_action,
		device,
		discount,
		tau,
		policy_noise,
		noise_clip,
		policy_freq,
		actor_lr,
		critic_lr,
		hidden_sizes,
	):
		self.device = device

		self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

		self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

		self.min_action = min_action
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		
		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % self.policy_freq == 0:
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic.pt", _use_new_zipfile_serialization = False)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor.state_dict(), filename + "_actor.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt", _use_new_zipfile_serialization = False)

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
		self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
