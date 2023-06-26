import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import math
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


class CartPole:

	def __init__(self):
		self.env = gym.make('CartPole-v1')
		self.max_x1, self.max_x2, self.max_x3, self.max_x4 = [self.env.observation_space.high[0], 0.5,
		                                                      self.env.observation_space.high[2], math.radians(50) / 1.]
		self.min_x1, self.min_x2, self.min_x3, self.min_x4 = [self.env.observation_space.low[0], -0.5,
		                                                      self.env.observation_space.low[2], -math.radians(50) / 1.]

	def initial_state(self):
		self.state = self.env.reset()[0]
		return self.state

	def get_actions(self, state):
		return [0, 1]

	def next_state_reward(self, state, action):
		next_state, reward, done, truncated, _ = self.env.step(action)
		return next_state, reward, done or truncated


def true_online_sarsa_lambda(cartPole, ag):
	state = cartPole.initial_state()
	action = ag.select_action(state)
	x = ag.features(state, action)
	z = np.zeros_like(ag.w)
	Q_old = 0
	timesteps = 0
	done = False
	ag.episode_reward = 0

	while not done:
		ag.action_count += 1
		next_state, reward, done = cartPole.next_state_reward(state, action)
		ag.total_rewards += reward
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
	def __init__(self, tilings=8, tile=8, max_size=8192, **kwargs):
		self.max_size = max_size
		self.tilings = tilings
		self.tile = tile

		super().__init__(**kwargs)

	def reset(self):
		self.iht = IHT(self.max_size)
		self.state_scale_factor = [self.tile / abs(self.mdp.max_x1 - self.mdp.min_x1),
		                           self.tile / abs(self.mdp.max_x2 - self.mdp.min_x2),
		                           self.tile / abs(self.mdp.max_x3 - self.mdp.min_x3),
		                           self.tile / abs(self.mdp.max_x4 - self.mdp.min_x4)]

		self.features = lambda state, action: tiles(self.iht, self.tilings,
		                                            [state[0] * self.state_scale_factor[0],
		                                             state[1] * self.state_scale_factor[1],
		                                             state[2] * self.state_scale_factor[2],
		                                             state[3] * self.state_scale_factor[3]],
		                                            [action])

		self.w = np.zeros(self.max_size)
		self.total_rewards = 0
		self.action_count = 0
		self.episode_reward = 0

	def q_value(self, state, action, x=None):
		if not x:
			x = self.features(state, action)
		return np.sum(self.w[x])


class TrueOnlineSarsaLambda(TileDiscretization):
	def __init__(self, lam, fn=true_online_sarsa_lambda, **kwargs):
		self.lam = lam
		self.gamma = 1
		super().__init__(fn=true_online_sarsa_lambda, **kwargs)

	def update(self, d_trace, d_active, z, x):
		self.w += self.alpha * d_trace * z
		self.w[x] -= self.alpha * d_active
		self.w = np.clip(self.w, -50, 50)


def CartPole_trueOnlineSarsaLambda():
	mdp = CartPole()
	run_count = 30
	episode_count = 200
	alphas = np.arange(0.2, 2, 0.2) / 8
	lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
	epsilon = 0
	ags = {'lambda=0.1': TrueOnlineSarsaLambda(mdp=mdp, lam=lam1, epsilon=epsilon),
	       'lambda=0.2': TrueOnlineSarsaLambda(mdp=mdp, lam=lam2, epsilon=epsilon),
	       'lambda=0.3': TrueOnlineSarsaLambda(mdp=mdp, lam=lam3, epsilon=epsilon),
	       'lambda=0.4': TrueOnlineSarsaLambda(mdp=mdp, lam=lam4, epsilon=epsilon),
	       'lambda=0.5': TrueOnlineSarsaLambda(mdp=mdp, lam=lam5, epsilon=epsilon),
	       'lambda=0.6': TrueOnlineSarsaLambda(mdp=mdp, lam=lam6, epsilon=epsilon),
	       'lambda=0.7': TrueOnlineSarsaLambda(mdp=mdp, lam=lam7, epsilon=epsilon),
	       'lambda=0.8': TrueOnlineSarsaLambda(mdp=mdp, lam=lam8, epsilon=epsilon),
	       'lambda=0.9': TrueOnlineSarsaLambda(mdp=mdp, lam=lam9, epsilon=epsilon),
	       'lambda=1.0': TrueOnlineSarsaLambda(mdp=mdp, lam=lam10, epsilon=epsilon)
	       }

	best_lambda = 0.4
	best_alpha = 0.225
	alphas = np.array([best_alpha])
	
	action_lambdas = []
	action_alphas = []
	episode_rewards = []

	for ag_name, ag in tqdm(ags.items(), desc='agents'):
		np.random.seed(687)

		with tqdm(total=len(alphas) * run_count * episode_count, leave=False) as pbar:

			avg_reward_per_episode = np.zeros(len(alphas))
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

						if ag_name == 'lambda=' + str(best_lambda):
							avg_action_count_per_episode_alpha[e] += ag.action_count
							reward_learning_curve[e] += ag.episode_reward
							std[e][run] += ag.episode_reward
					avg_reward_per_episode[i] += ag.total_rewards / episode_count
				avg_reward_per_episode[i] /= run_count
				avg_action_count_per_episode /= run_count
				avg_action_count_per_episode_alpha /= run_count
				reward_learning_curve /= run_count
				mean = np.mean(std, axis=1)
				std = np.std(std, axis=1)

				if ag_name == 'lambda=' + str(best_lambda):
					action_alphas.append(avg_action_count_per_episode_alpha)
					episode_rewards.append(mean)
			action_lambdas.append(avg_action_count_per_episode)

		plt.plot(alphas, avg_reward_per_episode, label=r'' + ag_name.replace('lambda', '$\lambda$'))
		plt.savefig('figures/cartPole-avgRewards.png')

	plt.xlabel(r'$\alpha$')
	plt.ylabel('Cart-Pole\n Average Rewards\n (average over first {} episodes and {} runs)'.format(episode_count, run_count))
	plt.legend()
	plt.title('True Online Sarsa Lambda (CartPole)')
	plt.savefig('figures/cartPole-avgRewards.png')
	plt.close()

	lams = [key for key in ags.keys()]
	for i in range(len(action_lambdas)):
		plt.plot(action_lambdas[i], np.arange(episode_count), label=r'' + lams[i].replace('lambda', '$\lambda$'))
	plt.xlabel('Average actions taken per episode')
	plt.ylabel('Episodes')
	plt.legend()
	plt.title('Action count for each lambda')
	plt.savefig('figures/cartPole-avgActionCount.png')
	plt.close()
	
	for i in range(len(action_alphas)):
		episode_rewards[i] = np.array(episode_rewards[i])
		x1 = mean - std
		x2 = mean + std
		plt.plot(np.arange(episode_count), episode_rewards[i], label=r'' + '$\\alpha$=' + str(alphas[i]))
		plt.fill_between(np.arange(episode_count), x1, x2, color='lightsteelblue')
	plt.xlabel('Episode')
	plt.ylabel('Average reward per episode')
	# plt.legend()
	plt.title('Avg reward per episode for alpha='+str(best_alpha) + '  for lambda='+str(best_lambda))
	plt.savefig('figures/cartPole-final-avgRewards-alpha.png')
	plt.close()


if __name__ == '__main__':
	np.random.seed(687)
	CartPole_trueOnlineSarsaLambda()
