import numpy as np
import torch
import gym
import gym_foo
import argparse
import os
import random
import improved_TD3 as TD3
import improved_DDPG as DDPG ## baselines
import improved_DARC as DARC
import time
import pandas as pd

##### Notice: change the environment parameter to test the model
#####         (ex: policy, RIS_N(reflective elements))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_policy(policy, env_name, seed, eval_episodes=10, args=None):

	# init the environment
	eval_env = gym.make(env_name, RIS_N = args.RIS_N, Train= False)
	eval_env.seed(seed + 100)
	np.random.seed(100)
	avg_reward = 0.
	avg_SE = 0.
	count = 0

	# load the channel data
	state_scaled_array = np.load(f"channel_data/evaluation_scaled_state_{args.RIS_N}.npy")

	results = np.empty((state_scaled_array.shape[0], 41, 9))

	for episode_idx in range(state_scaled_array.shape[0]):
		state, done = eval_env.reset(), False
		count = 0
		while not done:
			# get the state
			state = state_scaled_array[episode_idx, count, :]
			
			# use model to output the action
			start = time.time()
			discrete_action, continue_action = policy.select_action(np.array(state))
			end = time.time()

			# computing time
			used_time = end - start

			action = np.hstack((discrete_action, continue_action))

			# get the reward
			_, reward, done, (total_SE, _) = eval_env.step(action)
		    
			dis_a = np.argmax(discrete_action)
		
			best_params = np.array([state[-3], state[-2], state[-1], 0.4995 * continue_action[0] + 0.505, 0.4995 * continue_action[1] + 0.505, dis_a % 5 // 3, dis_a % 5 % 3 , used_time , total_SE]) 

			avg_SE += total_SE

			avg_reward += reward
			results[episode_idx, count, :] = best_params
			count += 1

	# average the results
	results = np.mean(results, axis = 0)
	df = pd.DataFrame(results)

	# save ro csv file
	df.to_csv(f"{args.policy}_result_{args.RIS_N}.csv", index=False, header=False)

	state, done = env.reset(), False
	avg_SE /= eval_episodes
	avg_reward /= eval_episodes

	print("Evaluation over {} episodes, Avg_Rewards: {}, Avg_SE: {}".format(eval_episodes, avg_reward, avg_SE))
	
	return avg_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", default="result.txt")
	
	# Communication parameters
	parser.add_argument("--RIS_N", default=8, type=int, help='CRIS_reflective_element')

	# Training parameters
	parser.add_argument("--policy", default="DARC")
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
	parser.add_argument("--load-model", default="default")                 # Model load file name, "" doesn't load, "default" uses file_name

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

	env = gym.make(args.env, RIS_N = args.RIS_N)

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
	
	# load the model (default: checkpoints/models/)
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load("{}/models/{}".format(outdir, policy_file))
		print("==== load success ====")

	eval_return = eval_policy(policy, args.env, args.seed, args=args)
