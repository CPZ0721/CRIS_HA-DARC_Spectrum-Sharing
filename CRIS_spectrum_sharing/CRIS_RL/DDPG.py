import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[400, 300]):
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
	def __init__(self, state_dim, action_dim, hidden_sizes=[400, 300]):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		
		return self.l3(q)


class DDPG(object):
	def __init__(self, device, state_dim, action_dim, min_action, max_action, 
				discount=0.99, 
				tau=0.005, 
				hidden_sizes=[400, 300],
				actor_lr = 3e-4,
				critic_lr = 3e-4):
		self.device = device
		self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay = 1e-5)

		self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay = 1e-5)

		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=100):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		self.bias_dis = torch.mean(target_Q)
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
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
