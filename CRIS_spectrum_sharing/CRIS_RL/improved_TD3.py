import copy
import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, num_discrete_actions, num_continuous_actions, max_action, hidden_sizes=[400, 300]):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3_discrete = nn.Linear(hidden_sizes[1], num_discrete_actions)
		self.l3_continuous = nn.Linear(hidden_sizes[1], num_continuous_actions)
	
		self.max_action = max_action

	def forward(self, state):

		a = F.relu(self.l1(state))
		#a = F.relu(self.l2(a))

		discrete_logit = F.softmax(self.l3_discrete(a), dim=-1)
		continuous_actions = torch.tanh(self.l3_continuous(a))


		return  discrete_logit, continuous_actions


class Critic(nn.Module):
	def __init__(self, state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim +  num_discrete_actions + num_continuous_actions, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim +  num_discrete_actions + num_continuous_actions, hidden_sizes[0])
		self.l5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l6 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, discrete_action, continuous_action):

		sa = torch.cat([state, discrete_action, continuous_action], 1)

		q1 = F.relu(self.l1(sa))
		#q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		#q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, discrete_action, continuous_action):
		sa = torch.cat([state, discrete_action, continuous_action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		num_discrete_actions,
		num_continuous_actions,
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

		self.actor = Actor(state_dim,  num_discrete_actions, num_continuous_actions,  max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)
		

		self.critic = Critic(state_dim,  num_discrete_actions, num_continuous_actions, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

		self.min_action = min_action
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.weight = 0.01

		self.total_it = 0

		self.num_discrete_actions = num_discrete_actions
		self.num_continuous_actions = num_continuous_actions


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		with torch.no_grad():
			discrete_logit, continuous_actions = self.actor(state)

		discrete_logit = discrete_logit.cpu().numpy()[0]
		continuous_actions = continuous_actions.cpu().numpy()[0]

		return discrete_logit, continuous_actions


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, discrete_logit, continuous_action, next_state, reward, not_done = replay_buffer.sample(batch_size)


		with torch.no_grad():
			# use actor_target to predict next action
			next_discrete_logit, next_continuous_actions = self.actor_target(next_state)

			noise = torch.randn(
				(discrete_logit.shape[0], discrete_logit.shape[1]), 
				dtype=discrete_logit.dtype, layout=discrete_logit.layout, device=discrete_logit.device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_discrete_logit = (next_discrete_logit + noise).clamp(self.min_action, self.max_action)

			noise = torch.randn(
				(continuous_action.shape[0], continuous_action.shape[1]), 
				dtype=continuous_action.dtype, layout=continuous_action.layout, device=continuous_action.device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_continuous_actions = (next_continuous_actions + noise).clamp(self.min_action, self.max_action)
			 
			target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_logit, next_continuous_actions)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, discrete_logit, continuous_action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % self.policy_freq == 0:

			actor_loss = -self.critic.Q1(state, self.actor(state)[0], self.actor(state)[1]).mean()

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
		torch.save(self.actor_optimizer.state_dict(), filename + "_dis_actor_optimizer.pt", _use_new_zipfile_serialization = False)

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
		self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_dis_actor_optimizer.pt"))
