import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from tiles3 import IHT, tiles
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


class MountainCar:
	def __init__(self):
		self.env = gym.make('MountainCar-v0')
		self.x_min, self.x_max = (-1.2, 0.5)
		self.v_min, self.v_max = (-0.07, 0.07)

	def initial_state(self):
		self.state = self.env.reset()[0]
		return self.state

	def get_actions(self, state):
		return [0, 1, 2]

	def next_state_reward(self, action):
		next_state, reward, done, _, _ = self.env.step(action)
		return next_state, reward, done

	def run_nstep_sarsa_episode(self,agent):
		T = float('inf')
		t = 0
		n = agent.n
		gamma = agent.gamma

		state = self.initial_state()
		s_hist = np.zeros((n + 1,2))
		s_hist[0] = state
		r_hist = np.zeros(n + 1)

		action = agent.select_action(state)
		agent.action_count += 1
		a_hist = np.zeros(n + 1)
		a_hist[0] = action

		agent.episode_reward = 0
		while True:
			if t < T:
				state, reward, done = self.next_state_reward(action)
				s_hist[(t + 1) % (n + 1)] = state
				r_hist[(t + 1) % (n + 1)] = reward

				if done or t >= 1000:
					T = t + 1
				else:
					action = agent.select_action(state)
					agent.action_count += 1
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
					G += gamma ** n * agent.get_q_value(t_state, t_action)

				tau_state = s_hist[tau % (n + 1)]
				tau_action = a_hist[tau % (n + 1)]

				features = agent.tiles(tau_state, tau_action)
				agent.w[features] += agent.alpha * (G - np.sum(agent.w[features]))

			t += 1
			agent.episode_reward += 1
			if tau == T - 1:
				break

		agent.episode_reward = t
		agent.total_rewards += t

		return t


class NStepSarsaAgent(Agent):
	def __init__(self, mdp, n, num_tilings=8, num_tiles=8, max_size=4096,**kwargs):
		self.n = n
		self.fn = mdp.run_nstep_sarsa_episode
		self.max_size = max_size
		self.num_tilings = num_tilings
		self.num_tiles = num_tiles
		super().__init__(mdp=mdp, fn=self.fn, **kwargs)

	def reset(self):
		self.iht = IHT(self.max_size)
		self.state_scale_factor = [self.num_tiles / abs(self.mdp.x_max - self.mdp.x_min),
		                           self.num_tiles / abs(self.mdp.v_max - self.mdp.v_min)]

		self.tiles = lambda state, action: tiles(self.iht,
		                                         self.num_tilings,
		                                         [state[0] * self.state_scale_factor[0],
		                                          state[1] * self.state_scale_factor[1]],
		                                         [action])
		self.w = np.zeros(self.max_size)
		self.total_rewards = 0
		self.episode_reward = 0
		self.action_count = 0

	def get_q_value(self, state, action):
		active_tiles = self.tiles(state, action)
		return np.sum(self.w[active_tiles])


def MountainCar_EpisodicSemiGradientNStepSarsa():
	mdp = MountainCar()
	run_count = 20
	episode_count = 100
	alphas = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
	alphas /= 8
	alphas = np.array([0.025])
	n = 8
	agent = NStepSarsaAgent(mdp, n=n, epsilon=0.1)

	alpha_episode_rewards = np.zeros((len(alphas),episode_count))
	alpha_total_rewards = np.zeros(len(alphas))
	action_alphas = np.zeros((len(alphas),episode_count))
	std = np.zeros((len(alphas),episode_count,run_count))

	alphas = [0.025]
	ns = [1, 2, 4, 8]

	alpha_episode_rewards = np.zeros((len(alphas), len(ns), episode_count))
	alpha_total_rewards = np.zeros((len(alphas),len(ns)))
	action_alphas = np.zeros((len(alphas), len(ns), episode_count))
	for i, alpha in enumerate(alphas):
		for j, n in enumerate(ns):
			mdp = MountainCar()
			agent = NStepSarsaAgent(mdp, n=n, epsilon=0.1)
			agent.alpha = alpha
			agent.reset()
			for run in range(run_count):
				agent.reset()

				for e in tqdm(range(episode_count)):
					alpha_episode_rewards[i, j, e] += mdp.run_nstep_sarsa_episode(agent)
					action_alphas[i, j, e] += agent.action_count
				alpha_total_rewards[i, j] += agent.total_rewards

	alpha_total_rewards /= run_count
	alpha_episode_rewards /= run_count
	action_alphas = np.mean(action_alphas, axis=2)

	for n in range(len(ns)):
		plt.plot(np.arange(episode_count), alpha_episode_rewards[0][n], label= r'' + 'n=' + str(ns[n]))
	plt.ylabel('Average reward per episode')
	plt.xlabel('n')
	plt.legend()
	plt.title('Average rewards for each n for alpha = 0.025')
	plt.savefig('figures/nstep-mountainCar-avg-total_rewards-n.png')
	plt.close()


if __name__ == '__main__':
	np.random.seed(687)
	MountainCar_EpisodicSemiGradientNStepSarsa()