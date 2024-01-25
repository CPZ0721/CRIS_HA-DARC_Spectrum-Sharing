import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


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

		return discrete_logit, continuous_actions


class Critic(nn.Module):
	def __init__(self, state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + num_discrete_actions + num_continuous_actions, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, discrete_action, continuous_action):
		q = F.relu(self.l1(torch.cat([state, discrete_action, continuous_action], dim=1)))
		#q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):
	def __init__(self, device, state_dim, num_discrete_actions, num_continuous_actions, min_action, max_action, 
				discount=0.99, 
				tau=0.005, 
				hidden_sizes=[400, 300],
				actor_lr = 3e-4,
				critic_lr = 3e-4):
		self.device = device
		self.actor = Actor(state_dim, num_discrete_actions, num_continuous_actions, max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay = 1e-5)

		self.critic = Critic(state_dim, num_discrete_actions, num_continuous_actions, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay = 1e-5)

		self.discount = discount
		self.num_discrete_actions = num_discrete_actions
		self.num_continuous_actions = num_continuous_actions
		self.tau = tau

		self.min_action = min_action
		self.max_action = max_action

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		with torch.no_grad():
			discrete_logit, continuous_actions = self.actor(state)
			
		discrete_logit = discrete_logit.cpu().numpy()[0]
		continuous_actions = continuous_actions.cpu().numpy()[0]

		return discrete_logit, continuous_actions

	def train(self, replay_buffer, batch_size=100):
		# Sample replay buffer 
		state, discrete_logit, continuous_action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state)[0], self.actor_target(next_state)[1])
			self.bias_dis = torch.mean(target_Q)
			target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, discrete_logit, continuous_action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		# Compute actor loss
		q_loss = -self.critic(state, self.actor(state)[0], self.actor(state)[1]).mean()

		actor_loss = q_loss
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
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
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
		self.actor_target = copy.deepcopy(self.actor)
