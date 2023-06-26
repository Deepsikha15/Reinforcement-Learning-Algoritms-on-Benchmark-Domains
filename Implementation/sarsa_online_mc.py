import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tiles3 import IHT, tiles
from collections import defaultdict


class Agent:
	def __init__(self, mdp, fn, alpha=0.1, gamma=1, epsilon=0):
		self.mdp, self.run_episode, self.gamma, self.epsilon, self.alpha = mdp, lambda: fn(mdp,
		                                                                                   self), gamma, epsilon, alpha
		self.reset()

	def q_value(self, state, action):
		return self.q_values[(state, action)]

	def select_action(self, state):
		actions = self.mdp.get_actions(state)
		if np.random.uniform(0, 1) < self.epsilon:
			return actions[np.random.choice(len(actions))]
		else:
			possible_actions = self.mdp.get_actions(state)
			q_values = [self.q_value(state, action) for action in possible_actions]
			best_actions = [a for i, a in enumerate(possible_actions) if
			                np.round(q_values[i], 4) == np.round(np.max(q_values), 4)]
			return best_actions[np.random.choice(np.array(best_actions).shape[0])]

	def reset(self):
		self.q_values = defaultdict(float)
		self.num_updates = 0


class MountainCar:
	def __init__(self):
		self.x_min, self.x_max = (-1.2, 0.5)
		self.v_min, self.v_max = (-0.07, 0.07)

	def initial_state(self):
		x = np.random.uniform(-0.6,-0.4)
		v = 0
		self.state = (x, v)
		return self.state

	def get_reward(self, state, action, next_state):
		next_x, next_v = next_state
		if next_x >= self.x_max:return 0
		else:return -1

	def get_actions(self, state):
		return [-1, 0, 1]

	def next_state_reward(self, state, action):
		x, v = state
		next_v = v + 0.001 * action - 0.0025 * np.cos(3 * x)
		next_x = x + next_v
		next_v = np.clip(next_v, self.v_min, self.v_max)
		next_x = np.clip(next_x, self.x_min, self.x_max)
		if next_x == self.x_min:
			next_v = 0
		next_state = next_x, next_v
		reward = self.get_reward(state, action, next_state)

		return next_state, reward

	def is_terminal(self, state):
		return np.round(state[0], 6) >= np.round(self.x_max, 6)


def true_online_sarsa_lambda(mountainCar, ag):
	state = mountainCar.initial_state()
	action = ag.select_action(state)
	x = ag.features(state, action)
	z = np.zeros_like(ag.w)
	Q_old = 0
	timesteps = 0
	done = False
	ag.episode_reward = 0

	while not mountainCar.is_terminal(state):
		ag.action_count += 1
		next_state, reward = mountainCar.next_state_reward(state, action)
		ag.cumulative_rewards += reward
		ag.episode_reward += reward
		next_action = ag.select_action(next_state)
		Q = ag.q_value(None, None, x)
		Q_prime = ag.q_value(next_state, next_action)
		z *= ag.gamma * ag.lam
		z[x] += 1 - ag.alpha * ag.gamma * ag.lam * (np.sum(z[x]))
		delta = reward + ag.gamma * Q_prime - Q
		delta += Q - Q_old
		delta_tiles = Q - Q_old
		ag.update(delta, delta_tiles, z, x)
		Q_old = Q_prime
		x = ag.features(next_state, next_action)
		state = next_state
		action = next_action
		timesteps += 1

		if timesteps >= 1000:
			break


class TileDiscretization(Agent):
	def __init__(self, tilings=8, tile=8, max_size=4096, **kwargs):
		self.max_size = max_size
		self.tilings = tilings
		self.tile = tile

		super().__init__(**kwargs)

	def reset(self):
		self.iht = IHT(self.max_size)
		self.state_scale_factor = [self.tile / abs(self.mdp.x_max - self.mdp.x_min),
		                           self.tile / abs(self.mdp.v_max - self.mdp.v_min)]

		self.features = lambda state, action: tiles(self.iht,
		                                         self.tilings,
		                                         [state[0] * self.state_scale_factor[0],
		                                          state[1] * self.state_scale_factor[1]],
		                                         [action])

		self.w = np.zeros(self.max_size)
		self.cumulative_rewards = 0
		self.episode_reward = 0
		self.action_count = 0

	def q_value(self, state, action, x=None):
		if not x:
			x = self.features(state, action)
		return np.sum(self.w[x])


class TrueOnlineSarsaLambda(TileDiscretization):
	def __init__(self, lam, fn=true_online_sarsa_lambda, **kwargs):
		self.lam = lam
		super().__init__(fn=true_online_sarsa_lambda, **kwargs)

	def update(self, d_trace, d_active, z, x):
		self.w += self.alpha * d_trace * z
		self.w[x] -= self.alpha * d_active
		self.w = np.clip(self.w, -50, 50)


def mountainCar_trueOnlineSarsaLambda():
	run_count = 20
	episode_count = 100
	alphas = np.arange(0.2, 2, 0.2) / 8
	lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
	epsilon = 0

	mdp = MountainCar()
	ags = {'lambda=0.1': TrueOnlineSarsaLambda(mdp=mdp, lam=lam1, epsilon=epsilon),
	          'lambda=0.2': TrueOnlineSarsaLambda(mdp=mdp, lam=lam2, epsilon=epsilon),
	          'lambda=0.3': TrueOnlineSarsaLambda(mdp=mdp, lam=lam3, epsilon=epsilon),
	          'lambda=0.4': TrueOnlineSarsaLambda(mdp=mdp, lam=lam4, epsilon=epsilon),
	          'lambda=0.5': TrueOnlineSarsaLambda(mdp=mdp, lam=lam5, epsilon=epsilon),
	          'lambda=0.6': TrueOnlineSarsaLambda(mdp=mdp, lam=lam6, epsilon=epsilon),
	          'lambda=0.7': TrueOnlineSarsaLambda(mdp=mdp, lam=lam7, epsilon=epsilon),
	          'lambda=0.8': TrueOnlineSarsaLambda(mdp=mdp, lam=lam8, epsilon=epsilon),
	          'lambda=0.9': TrueOnlineSarsaLambda(mdp=mdp, lam=lam9, epsilon=epsilon)
	          }

	# best hyperparams
	# run_count = 20
	# episode_count = 100
	best_lambda = 0.9
	# best_alpha = 0.1
	# alphas = np.array([best_alpha])
	#
	# # best agent
	ags = {'lambda='+str(best_lambda): TrueOnlineSarsaLambda(mdp=mdp, lam=best_lambda, epsilon=epsilon)}

	action_lambdas = []
	action_alphas = []
	episode_rewards = []

	for ag_name, ag in tqdm(ags.items(), desc='agents'):

		with tqdm(total=alphas.shape[0] * run_count * episode_count, leave=False) as pbar:

			avg_reward_per_episode = np.zeros(alphas.shape[0])
			avg_action_count_per_episode = np.zeros(episode_count)

			for i, alpha in enumerate(alphas):
				ag.alpha = alpha
				avg_action_count_per_episode_alpha = np.zeros(episode_count)
				reward_learning_curve = np.zeros(episode_count)
				std = np.zeros((episode_count, run_count))

				for run in range(run_count):
					ag.reset()

					for e in range(episode_count):
						ag.run_episode()
						avg_action_count_per_episode[e] += ag.action_count
						std[e][run] += ag.episode_reward

						if ag_name == 'lambda='+str(best_lambda):
							avg_action_count_per_episode_alpha[e] += ag.action_count
							reward_learning_curve[e] += ag.episode_reward
							std[e][run] += ag.episode_reward
					avg_reward_per_episode[i] += ag.cumulative_rewards / episode_count
				avg_reward_per_episode[i] /= run_count
				avg_action_count_per_episode /= run_count
				avg_action_count_per_episode_alpha /= run_count
				reward_learning_curve /= run_count
				mean = np.mean(std, axis=1)
				std = np.std(std, axis=1)

				if ag_name == 'lambda='+str(best_lambda):
					action_alphas.append(avg_action_count_per_episode_alpha)
					episode_rewards.append(reward_learning_curve)
			action_lambdas.append(avg_action_count_per_episode)

	for i in range(len(action_alphas)):
		plt.plot(action_alphas[i], np.arange(episode_count), label=r'' + '$\\alpha$=' + str(alphas[i]))
	plt.xlabel('Average actions taken per episode')
	plt.ylabel('Episodes')
	plt.legend()
	plt.title('Action count for each alpha for lambda=0.9')
	plt.savefig('figures/MountainCar-avgActionCount-alpha.png')
	plt.close()

	for i in range(len(action_alphas)):
		plt.plot(np.arange(episode_count), -episode_rewards[i], label=r'' + '$\\alpha$=' + str(alphas[i]))
	plt.xlabel('Episode')
	plt.ylabel('Average reward per episode')
	plt.legend()
	plt.title('Avg timestep per episode for each alpha for lambda=0.9')
	plt.savefig('figures/MountainCar-avgRewards-alpha.png')
	plt.close()


# final graph
# 	for i in range(len(action_alphas)):
# 		mean *= -1
# 		x1 = mean - std
# 		x2 = mean + std
# 		# plt.plot(np.arange(episode_count), episode_rewards[i], label=r'' + '$\\alpha$=' + str(alphas[i]))
# 		plt.plot(np.arange(episode_count), mean, label=r'' + '$\\alpha$=' + str(alphas[i]))
# 		plt.fill_between(np.arange(episode_count), x1, x2, color='lightsteelblue')
# 	plt.xlabel('Episode')
# 	plt.ylabel('Average timesteps per episode')
# 	plt.legend()
# 	plt.title('Avg timesteps per episode for alpha='+str(best_alpha) + ' for lambda='+str(best_lambda))
# 	plt.savefig('figures/mountainCar-final-avgRewards-alpha.png')
# 	plt.close()


if __name__ == '__main__':
	np.random.seed(687)
	mountainCar_trueOnlineSarsaLambda()
