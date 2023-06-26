import numpy as np
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm
from tiles3 import IHT, tiles
import math
import numpy as np
from collections import defaultdict


class Agent:
	def __init__(self, mdp, fn, gamma=1, epsilon=0.1, alpha=0.1):
		self.mdp, self.run, self.gamma, self.epsilon, self.alpha = mdp, lambda: fn(self), gamma, epsilon, alpha

	def select_action(self, state):
		prob = np.random.rand()
		actions = self.mdp.get_actions(state)
		if prob < self.epsilon:
			return actions[np.random.choice(len(actions))]
		else:
			possible_actions = self.mdp.get_actions(state)
			if possible_actions[0] is None:
				return None
			q_values = [self.get_q_value(state, a) for a in possible_actions]
			best_actions = [a for i, a in enumerate(possible_actions) if
			                np.round(q_values[i], 8) == np.round(np.max(q_values), 8)]
			return best_actions[np.random.choice(len(best_actions))]

	def get_q_value(self, state, action):
		return self.q_values[(state, action)]

	def reset(self):
		self.q_values = defaultdict(float)
		self.num_updates = 0


class CartPole:
	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.max_x1, self.max_x2, self.max_x3, self.max_x4 = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
		self.min_x1, self.min_x2, self.min_x3, self.min_x4 = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

	def initial_state(self):
		self.state = self.env.reset()[0]
		return self.state

	def get_actions(self, state):
		return [0, 1]

	def next_state_reward(self, action):
		next_state, reward, done, truncated, _ = self.env.step(action)
		return next_state, reward, done or truncated

	def run_nstep_sarsa_episode(self, ag):
		T = float('inf')
		t = 0
		n = ag.n
		gamma = ag.gamma

		state = self.initial_state()
		s_hist = np.zeros((n + 1, 4))
		s_hist[0] = state
		r_hist = np.zeros(n + 1)

		action = ag.select_action(state)
		ag.action_count += 1
		a_hist = np.zeros(n + 1)
		a_hist[0] = action

		ag.episode_reward = 0
		while True:
			if t < T:
				state, reward, done = self.next_state_reward(action)
				s_hist[(t + 1) % (n + 1)] = state
				r_hist[(t + 1) % (n + 1)] = reward

				if done:
					T = t + 1
				else:
					action = ag.select_action(state)
					ag.action_count += 1
					a_hist[(t + 1) % (n + 1)] = action

			tau = t - n + 1
			if tau >= 0:
				G = 0
				gamma_coeff = gamma ** np.arange(min(n, T - tau))
				for i in range(tau, min(tau + n, T)):
					G += gamma_coeff[i - tau] * r_hist[(i + 1) % (n + 1)]

				if tau + n < T:
					t_state = s_hist[(tau + n) % (n + 1)]
					t_action = a_hist[(tau + n) % (n + 1)]
					G += gamma ** n * ag.get_q_value(t_state, t_action)

				tau_state = s_hist[tau % (n + 1)]
				tau_action = a_hist[tau % (n + 1)]

				features = ag.features(tau_state, tau_action)
				ag.w[features] += ag.alpha * (G - np.sum(ag.w[features]))

			t += 1
			ag.episode_reward += 1
			if tau == T - 1:
				break

		ag.episode_reward = t
		ag.total_rewards += t

		return t


class NStepSarsaAgent(Agent):
	def __init__(self, mdp, n, num_tilings=8, num_tiles=8, max_size=8192, **kwargs):
		self.n = n
		self.max_size = max_size
		self.num_tilings = num_tilings
		self.num_tiles = num_tiles
		self.fn = mdp.run_nstep_sarsa_episode

		super().__init__(mdp=mdp, fn=self.fn, **kwargs)

	def reset(self):
		self.iht = IHT(self.max_size)
		self.state_scale_factor = [self.num_tiles / abs(self.mdp.max_x1 - self.mdp.min_x1),
		                           self.num_tiles / abs(self.mdp.max_x2 - self.mdp.min_x2),
		                           self.num_tiles / abs(self.mdp.max_x3 - self.mdp.min_x3),
		                           self.num_tiles / abs(self.mdp.max_x4 - self.mdp.min_x4)]

		self.features = lambda state, action: tiles(self.iht,
		                                         self.num_tilings,
		                                         [state[0] * self.state_scale_factor[0],
		                                          state[1] * self.state_scale_factor[1],
		                                          state[2] * self.state_scale_factor[2],
		                                          state[3] * self.state_scale_factor[3]],
		                                         [action])

		self.w = np.zeros(self.max_size)
		self.total_rewards = 0
		self.episode_reward = 0
		self.action_count = 0

	def get_q_value(self, state, action):
		active_tiles = self.features(state, action)
		return np.sum(self.w[active_tiles])


def CartPole_EpisodicSemiGradientNStepSarsa():
	mdp = CartPole()
	n = 4
	ns = [4]
	agent = NStepSarsaAgent(mdp, n=n, epsilon=0.1)
	run_count = 20
	episode_count = 1000
	# alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
	# alphas /= 8
	alphas = np.array([0.05])

	alpha_episode_rewards = np.zeros((len(alphas), episode_count))
	alpha_total_rewards = np.zeros(len(alphas))
	action_alphas = np.zeros((len(alphas), episode_count))
	std = np.zeros((len(alphas), episode_count, run_count))

	for i, alpha in enumerate(alphas):
		agent.alpha = alpha
		agent.reset()
		for j in range(run_count):
			agent.reset()

			for e in tqdm(range(episode_count)):
				alpha_episode_rewards[i, e] += agent.run()
				std[i,e,j] = agent.episode_reward
				action_alphas[i, e] += agent.action_count
			alpha_total_rewards[i] += agent.total_rewards

	alpha_total_rewards /= (episode_count * run_count)
	alpha_episode_rewards /= run_count
	action_alphas /= run_count
	std = np.std(std,axis=2)

	for a in range(len(alphas)):
		x1 = alpha_episode_rewards[a] - std[a]
		x2 = alpha_episode_rewards[a] + std[a]
		plt.plot(np.arange(episode_count), alpha_episode_rewards[a], label=r'' + '$\\alpha=$' + str(alphas[a]))
		plt.fill_between(np.arange(episode_count), x1, x2, color='lightsteelblue')
	plt.xlabel('Episode')
	plt.ylabel('Average reward per episode')
	plt.legend()
	plt.title('Episodic Semi-Gradient N-step SARSA (cartPole) for ' + '$\\alpha$=' + str(alphas[a]) + ', n=' + str(n))
	plt.savefig('figures/nstep-cartPole-opt-avg-rewards.png')
	plt.close()

	for a in range(len(alphas)):
		plt.plot(action_alphas[a], np.arange(episode_count), label=r'' + '$\\alpha=$' + str(alphas[a]))
	plt.ylabel('Episode')
	plt.xlabel('Average action count')
	plt.legend()
	# plt.title('Average action count for each alpha')
	plt.title('Average action count for ' + '$\\alpha$=' + str(alphas[a]) + ', n=' + str(n))
	plt.savefig('figures/nstep-cartPole-opt-avg-action-count.png')
	plt.close()


if __name__ == '__main__':
	np.random.seed(687)
	CartPole_EpisodicSemiGradientNStepSarsa()
