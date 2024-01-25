import numpy as np
import torch
import gym
import gym_foo
import argparse
import os
import utils
import random
import improved_TD3 as TD3
import improved_DDPG as DDPG ## baselines
import improved_DARC as DARC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# evaluation
def eval_policy(policy, env_name, seed, eval_episodes=10, eval_cnt=None, args = None):

	eval_env = gym.make(env_name, RIS_N = args.RIS_N, Train= False )
	eval_env.seed(seed + 100)
	avg_reward = 0.
	avg_SE = 0.

	for epi in range(eval_episodes):
		state, done = eval_env.reset(), False
		count = 0
		while not done:
			discrete_action, continue_action = policy.select_action(np.array(state))
			actions = np.hstack((discrete_action, continue_action))
			next_state, reward, done, (total_SE, _) = eval_env.step(actions)

			dis_a = np.argmax(discrete_action)

			# record result into `args.dir` path
			with open(args.dir, 'a') as f:
				f.write('epi: ' + str(eval_cnt * eval_episodes + epi) + '|step: ' + str(count) + ' |env: '+ str(state[-3:])+'| Act_alpha1: '+ str(np.eye(3)[dis_a // 3])
						+ ' | Act_alpha2: '+ str(np.eye(3)[dis_a % 3]) + ' | Act_SU_power1: ' + str(0.495 * continue_action[0] + 0.505) + ' | Act_SU_power2: ' + str(0.495 * continue_action[1] + 0.505)
						+ ' | Phase shift: ' + str((continue_action[2:2+args.RIS_N]+1)/2*2*np.pi) + ' | Reward: ' + str(reward) + ' | SE: ' + str(total_SE) + '\n')

			count += 1
			avg_SE += total_SE
			avg_reward += reward
			state = next_state

		with open(args.dir, 'a') as f:
			f.write('====================================='+ '\n')

	state, done = env.reset(), False
	avg_SE /= eval_episodes
	avg_reward /= eval_episodes

	print("[{}] Evaluation over {} episodes, Avg_Rewards: {}, Avg_SE: {}".format(eval_cnt, eval_episodes, avg_reward, avg_SE))
	
	return avg_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", default="result.txt")
	
	# Communication parameters
	parser.add_argument("--RIS_N", default=8, type=int, help='CRIS_reflective_element')

	# Training parameters
	parser.add_argument("--policy", default="DDPG")
	parser.add_argument("--env", default='foo-v0')
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--steps", default=41, type=int)
	parser.add_argument("--start-episode", default=100, type=int, help='Number of episodes for the warm-up stage using random policy')
	parser.add_argument("--eval-episode", default=100, type=int, help='Number of episodes per evaluation')
	parser.add_argument("--episode", default=30000, type=int, help='Maximum number of episodes')

	parser.add_argument("--actor-lr", default=5e-4, type=float)     
	parser.add_argument("--critic-lr", default=5e-4, type=float)    
	parser.add_argument("--hidden-sizes", default='512,512', type=str)  
	parser.add_argument("--batch-size", default=pow(2, 10), type=int)      # Batch size for both actor and critic
	parser.add_argument("--save-model", action="store_true", default=True)        # Save model and optimizer parameters
	parser.add_argument("--load-model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	# Algo parameter
	parser.add_argument("--discount", default=0.99, help='Discount factor')
	parser.add_argument("--tau", default=0.005, help='Target network update rate')                    
	
	parser.add_argument("--qweight", default=0.4, type=float, help='The weighting coefficient that correlates value estimation from double actors')
	parser.add_argument("--reg", default=0.005, type=float, help='The regularization parameter for DARC')   

	parser.add_argument("--expl-noise", default=0.1, type=float)                # Std of Gaussian exploration noise
	parser.add_argument("--policy-noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise-clip", default=0.5, type=float)                # Range to clip target policy noise
	parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
	
	args = parser.parse_args()

	print("---------------------------------------")
	print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
	print("---------------------------------------")
	outdir = "checkpoints"
	file_name = f"greedy_{args.RIS_N}_{args.policy}_{args.actor_lr}_{args.critic_lr}_{args.qweight}_{args.reg}"

	if args.save_model and not os.path.exists("{}/models".format(outdir)):
		os.makedirs("{}/models".format(outdir))

	Rewards = []
	History_Reward= []

	env = gym.make(args.env, RIS_N = args.RIS_N)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.cuda.manual_seed_all(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	con_action_dim = env.con_action_space.shape[0]
	dis_action_dim = env.dis_action_space.shape[0]
	max_action = float(env.max_action)
	min_action = float(env.min_action) 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	kwargs = {
		"state_dim": state_dim,
		"num_continuous_actions": con_action_dim,
		"num_discrete_actions": dis_action_dim,
		"min_action": min_action,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
		"actor_lr": args.actor_lr,
		"critic_lr": args.critic_lr,
		"device": device,
	}

	if args.policy == "TD3":

		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq

		policy = TD3.TD3(**kwargs)

	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	
	elif args.policy == "DARC":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["q_weight"] = args.qweight
		kwargs["regularization_weight"] = args.reg

		policy = DARC.DARC(**kwargs)
	
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load("{}/models/{}".format(outdir, policy_file))

		print("==== load success ====")

	replay_buffer = utils.ReplayBuffer(state_dim, dis_action_dim, con_action_dim, device)

	eval_cnt = 0

	# recreation empty file
	if os.path.exists(args.dir):
		os.remove(args.dir)

	eval_return = eval_policy(policy, args.env, args.seed, eval_cnt=eval_cnt, args = args)
	eval_cnt += 1
	Rewards.append(eval_return)

	state, done = env.reset(), False
	episode_reward = 0
	episode_SE = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.steps * args.steps)):
		episode_timesteps += 1
		
		if t < args.steps * args.start_steps:
			# random
			discrete_action = ((max_action - min_action) * np.random.random(dis_action_dim) + min_action).clip(-max_action, max_action)
			continue_action = ((max_action - min_action) * np.random.random(con_action_dim) + min_action).clip(-max_action, max_action)

		else:
			# model output
			discrete_action, continue_action = policy.select_action(np.array(state))
			discrete_action = (discrete_action +  np.random.normal(0, max_action * args.expl_noise, size=dis_action_dim)).clip(-max_action, max_action)
			continue_action = (continue_action +  np.random.normal(0, max_action * args.expl_noise, size=con_action_dim)).clip(-max_action, max_action)
			
		actions = np.hstack((discrete_action, continue_action))
		next_state, reward, done, (total_SE, PU_SINR) = env.step(actions)
		done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

		# put tuple into replay buffer
		replay_buffer.add(state, discrete_action, continue_action, next_state, reward, done_bool)

		state = next_state
		episode_SE += total_SE
		episode_reward += reward

		# start training after the number of data is larger than bach size
		if t >= (args.steps * args.start_steps):
			# train agent
			policy.train(replay_buffer, args.batch_size)
		
		# the final step
		if done: 
			print("Total T: {} Episode Num: {} Episode T: {} Reward: {} Total SE: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward, episode_SE))
			History_Reward.append(episode_reward)
			state, done = env.reset(), False
			episode_reward = 0
			episode_SE = 0
			episode_timesteps = 0
			episode_num += 1
		
		# every `args.eval_freq` execute evaluation
		if (t + 1) % (args.steps * args.eval_freq) == 0:
			eval_return = eval_policy(policy, args.env, args.seed, eval_cnt=eval_cnt, args = args)
			eval_cnt += 1
			Rewards.append(eval_return)
			state, done = env.reset(), False

			if args.save_model:
				policy.save('{}/models/{}'.format(outdir, file_name))

	np.savetxt(f"greedy_{args.policy}_Training_Rewards_{args.RIS_N}_{args.qweight}_{args.reg}.csv", Rewards, delimiter=',')
	np.savetxt(f"greedy_{args.policy}_Training_History_Reward_{args.RIS_N}_{args.qweight}_{args.reg}.csv", History_Reward, delimiter=',')
