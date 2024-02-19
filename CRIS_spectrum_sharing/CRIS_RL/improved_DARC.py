import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create Actor Model
class Actor(nn.Module):
	def __init__(self, state_dim, num_discrete_actions, num_continuous_actions, max_action, hidden_sizes=[400, 300]):
		super(Actor, self).__init__()

		# linear layer and layer norm
		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.ln1 = nn.LayerNorm(hidden_sizes[0])

		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.ln2 = nn.LayerNorm(hidden_sizes[1])

		# parallel layer with continous and discrete layers
		self.l3_discrete = nn.Linear(hidden_sizes[1], num_discrete_actions)
		self.l3_continuous = nn.Linear(hidden_sizes[1], num_continuous_actions)
		self.ln3_d = nn.LayerNorm(num_discrete_actions)
		self.ln3_c = nn.LayerNorm(num_continuous_actions)

		# weight and bias setting
		self.l1.weight.data.normal_(0, 0.1)
		self.l1.bias.data.normal_(0, 0.1)
		self.l2.weight.data.normal_(0, 0.1)
		self.l2.bias.data.normal_(0, 0.1)
		self.l3_continuous.weight.data.normal_(0, 0.1)
		self.l3_continuous.bias.data.normal_(0, 0.1)
		self.l3_discrete.weight.data.normal_(0, 0.1)
		self.l3_discrete.bias.data.normal_(0, 0.1)

		self.max_action = max_action


	def forward(self, state):
		
		# activation function

		a = F.relu(self.ln1(self.l1(state)))
		# a = F.relu(self.ln2(self.l2(a)))
		discrete_logit = F.softmax(self.ln3_d(self.l3_discrete(a)), dim=-1)
		continuous_actions = torch.tanh(self.ln3_c(self.l3_continuous(a)))

		return discrete_logit, continuous_actions


class Critic(nn.Module):
	def __init__(self, state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()

		# linear layer and layer norm layer

		# first layer is concentrate layer
		self.l1 = nn.Linear(state_dim + num_discrete_actions + num_continuous_actions, hidden_sizes[0])
		self.ln1 = nn.LayerNorm(hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.ln2 = nn.LayerNorm(hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)

		# weight and bias setting
		self.l1.weight.data.normal_(0, 0.1)
		self.l1.bias.data.normal_(0, 0.1)
		self.l2.weight.data.normal_(0, 0.1)
		self.l2.bias.data.normal_(0, 0.1)
		self.l3.weight.data.normal_(0, 0.1)
		self.l3.bias.data.normal_(0, 0.1)


	def forward(self, state, discrete_action, continuous_action):

		# activation function
		if len(state.shape) == 3:
			sa = torch.cat([state, discrete_action, continuous_action], 2)
		else:
			sa = torch.cat([state, discrete_action, continuous_action], 1)

		q = F.relu(self.ln1(self.l1(sa)))
		#q = F.relu(self.ln2(self.l2(q)))
		q = self.l3(q)

		return q


class DARC(object):
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
		actor_lr,
		critic_lr,
		hidden_sizes,
		q_weight,
		regularization_weight,
	):
		self.device = device

		# actor/critic and target networks
		self.actor1 = Actor(state_dim, num_discrete_actions, num_continuous_actions, max_action, hidden_sizes).to(self.device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr, weight_decay=1e-5)

		self.actor2 = Actor(state_dim, num_discrete_actions, num_continuous_actions, max_action, hidden_sizes).to(self.device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr, weight_decay=1e-5)

		self.critic1 = Critic(state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=1e-5)

		self.critic2 = Critic(state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=1e-5)
		self.min_action = min_action

		# global parameter setting
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.q_weight = q_weight
		self.regularization_weight = regularization_weight

		self.num_discrete_actions = num_discrete_actions
		self.num_continuous_actions = num_continuous_actions

	# choose the action which is output of actor models
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		with torch.no_grad():
			discrete_logit_1, continuous_actions_1 = self.actor1(state)
			discrete_logit_2, continuous_actions_2 = self.actor2(state)
			q1 = self.critic1(state, discrete_logit_1, continuous_actions_1)
			q2 = self.critic2(state, discrete_logit_2, continuous_actions_2)

		discrete_logit = discrete_logit_1 if q1 >= q2 else discrete_logit_2
		continuous_actions = continuous_actions_1 if q1 >= q2 else continuous_actions_2

		discrete_logit = discrete_logit.cpu().numpy()[0]
		continuous_actions = continuous_actions.cpu().numpy()[0]

		return discrete_logit, continuous_actions


	def train(self, replay_buffer, batch_size=100):
		## cross-update scheme
		self.train_one_q_and_pi(replay_buffer, True, batch_size=batch_size)
		self.train_one_q_and_pi(replay_buffer, False, batch_size=batch_size)

	def train_one_q_and_pi(self, replay_buffer, update_a1 = True, batch_size=100):
		# smaple a batch size data from replay buffer
		state, discrete_logit, continuous_action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# back propogation
		with torch.no_grad():
			next_discrete_logit_1, next_continuous_actions_1 = self.actor1_target(next_state)
			next_discrete_logit_2, next_continuous_actions_2 = self.actor2_target(next_state)

			noise = torch.randn(
				(discrete_logit.shape[0], discrete_logit.shape[1]),
				dtype=discrete_logit.dtype, layout=discrete_logit.layout, device=discrete_logit.device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_discrete_logit_1 = (next_discrete_logit_1 + noise).clamp(self.min_action, self.max_action)
			next_discrete_logit_2 = (next_discrete_logit_2 + noise).clamp(self.min_action, self.max_action)

			noise = torch.randn(
				(continuous_action.shape[0], continuous_action.shape[1]),
				dtype=continuous_action.dtype, layout=continuous_action.layout, device=continuous_action.device
			) * self.policy_noise
			noise = noise.clamp(-self.noise_clip, self.noise_clip)

			next_continuous_actions_1 = (next_continuous_actions_1 + noise).clamp(self.min_action, self.max_action)
			next_continuous_actions_2 = (next_continuous_actions_2 + noise).clamp(self.min_action, self.max_action)

			next_Q1_a1 = self.critic1_target(next_state, next_discrete_logit_1, next_continuous_actions_1)
			next_Q2_a1 = self.critic2_target(next_state, next_discrete_logit_1, next_continuous_actions_1)

			next_Q1_a2 = self.critic1_target(next_state, next_discrete_logit_2, next_continuous_actions_2)
			next_Q2_a2 = self.critic2_target(next_state, next_discrete_logit_2, next_continuous_actions_2)

			## min first, max afterward to avoid underestimation bias
			next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
			next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)

			## soft q update
			next_Q = self.q_weight * torch.min(next_Q1, next_Q2) + (1-self.q_weight) * torch.max(next_Q1, next_Q2)

			target_Q = reward + not_done * self.discount * next_Q

		# update actor 1
		if update_a1:
			current_Q1 = self.critic1(state, discrete_logit, continuous_action)
			current_Q2 = self.critic2(state, discrete_logit, continuous_action)

			# critic regularization
			critic1_loss = F.mse_loss(current_Q1, target_Q) + self.regularization_weight * F.mse_loss(current_Q1, current_Q2)

			self.critic1_optimizer.zero_grad()
			critic1_loss.backward()
			self.critic1_optimizer.step()

			actor1_loss = -self.critic1(state, self.actor1(state)[0], self.actor1(state)[1]).mean()

			self.actor1_optimizer.zero_grad()
			actor1_loss.backward()
			self.actor1_optimizer.step()

			# copy parameters from model 1 to target 1
			for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# update actor 2
		else:
			current_Q1 = self.critic1(state, discrete_logit, continuous_action)
			current_Q2 = self.critic2(state, discrete_logit, continuous_action)

			# critic regularization
			critic2_loss = F.mse_loss(current_Q2, target_Q) + self.regularization_weight * F.mse_loss(current_Q2, current_Q1)

			self.critic2_optimizer.zero_grad()
			critic2_loss.backward()
			self.critic2_optimizer.step()

			actor2_loss = -self.critic2(state, self.actor2(state)[0], self.actor2(state)[1]).mean()

			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()

			# copy parameters from model 2 to target 2
			for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# save model
	def save(self, filename):
		torch.save(self.critic1.state_dict(), filename + "_critic1.pt", _use_new_zipfile_serialization = False)
		torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor1.state_dict(), filename + "_actor1.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer.pt", _use_new_zipfile_serialization = False)

		torch.save(self.critic2.state_dict(), filename + "_critic2.pt", _use_new_zipfile_serialization = False)
		torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor2.state_dict(), filename + "_actor2.pt", _use_new_zipfile_serialization = False)
		torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer.pt", _use_new_zipfile_serialization = False)

	# load the model
	def load(self, filename):
		self.critic1.load_state_dict(torch.load(filename + "_critic1.pt"))
		self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer.pt"))
		self.actor1.load_state_dict(torch.load(filename + "_actor1.pt"))
		self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer.pt"))

		self.critic2.load_state_dict(torch.load(filename + "_critic2.pt"))
		self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer.pt"))
		self.actor2.load_state_dict(torch.load(filename + "_actor2.pt"))
		self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer.pt"))
