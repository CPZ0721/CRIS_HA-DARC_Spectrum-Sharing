import numpy as np
import gym
import gym_foo
import os
import argparse

##### Notice: change the Env to "foo_env_evaluation.py"
#####         and adjust the corresponding number of RIS element

def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	parser = argparse.ArgumentParser()
	parser.add_argument("--RIS_N", default=8, type=int, help='CRIS_reflective_element')
	args = parser.parse_args()

	N = args.RIS_N

	maxStep = 41
	
	# number of data
	data_num = 10

	env = gym.make('foo-v0', RIS_N = N, Train = False)
	env.seed(100)

	state_dim = env.observation_space.shape[0]
	state_array = np.empty((data_num, maxStep, state_dim))
	state_scaled_array = np.empty((data_num, maxStep, state_dim))

	# generate the number of `data_num` data
	for i in range(data_num):
		state, state_scaled = env.reset()
		# each episode has `maxStep` steps
		for num_step in range(maxStep):
			state_array[i,num_step,:] = state
			state_scaled_array[i,num_step,:] = state_scaled
			state, state_scaled = env.evaluation_step(num_step)

		print("Data Step： " + str(i+1) + " is done.")
	np.save(f"evaluation_state_{N}.npy", state_array)
	np.save(f"evaluation_scaled_state_{N}.npy", state_scaled_array)

if __name__ == '__main__':
	main()
