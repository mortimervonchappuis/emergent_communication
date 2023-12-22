import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
from stackedautoencoder import AutoEncoder
from communication import Language, ActorCom, CriticCom
from control import ActorCtrl, CriticCtrl

from gridworld import GridWorld
from random import randint



class BaseAgent:
	def __init__(self, actor, critic, gamma=0.99, lambd=0.95):
		self.gamma  = gamma
		self.lambd  = lambd
		self.actor  = actor()
		self.critic = critic()
		self.actor_data  = []
		self.critic_data = []


	def parse_trajectory(self, trajectory, reward_key, com=False):
		embeddings = tf.concat(trajectory['embeddings'], axis=0)
		rewards    = tf.stack(trajectory[reward_key])
		actions    = tf.stack(trajectory['actions'])
		if com:
			message = tf.stack(tf.concat(trajectory['message'], axis=0))
			values  = self.critic(embeddings, message)
		else:
			values  = self.critic(embeddings)
		returns    = self.RET(rewards)
		advantages = self.GAE(values, rewards)
		self.actor_data.extend(list(zip(embeddings, advantages, actions, message)))
		self.critic_data.extend(list(zip(embeddings, returns, message)))


	def reset(self):
		self.actor_data  = []
		self.critic_data = []


	def RET(self, rewards):
		rewards = tf.concat([rewards, [0.0]], axis=0)
		def scan(returns, rewards):
			return rewards + self.gamma * returns
		return tf.scan(scan, rewards, reverse=True)


	def GAE(self, values, rewards):
		shifted   = tf.concat([values[1:], [values[-1]]], axis=0)
		temp_diff = rewards + self.gamma * shifted - values
		def scan(advantage, temp_diff):
			return temp_diff + self.gamma * self.lambd * advantage
		return tf.scan(scan, temp_diff, reverse=True)


	def train(self):
		#shuffle(self.critic_data)
		actor_data  = list(map(tf.stack, zip(*self.actor_data)))
		critic_data = list(map(tf.stack, zip(*self.critic_data)))
		actor_target = -self.actor.train(actor_data)
		critic_loss  = self.critic.train(critic_data)
		self.reset()
		return actor_target, critic_loss



class Agent:
	def __init__(self, babble_number=0):
		self.babble_number = babble_number
		self.autoencoder = AutoEncoder()
		#self.language    = Language()
		self.ctrl = BaseAgent(actor=ActorCtrl, 
				      critic=CriticCtrl)
		self.com  = BaseAgent(actor=ActorCom, 
				      critic=CriticCom)


	def observe(self, observation, training=False):
		return self.autoencoder.encoder(observation, training=training)


	def speak(self, embedding, training=False):
		return self.com.actor(embedding, return_logits=True, training=training)


	def babble(self, embedding, training=False):
		return [self.com.actor(embedding, training=training)[0] for _ in range(self.babble_number)]


	def act(self, embedding, message, training=False):
		logits = self.ctrl.actor(embedding, message, training=training)
		cat    = tfp.distributions.Categorical(logits=logits)
		action = cat.sample()
		return action


	def information(self, message, log_posterior, batch_size=64, training=False):
		batch     = self.gather(batch_size)
		embedding = self.observe(batch)
		logit     = self.log_posterior(message, embedding, training=training)
		log_prior = tf.reduce_mean(logit)
		return log_posterior - log_prior


	def train(self, trajectories):
		for trajectory in trajectories:
			self.ctrl.parse_trajectory(trajectory, reward_key='rewards')
			self.com.parse_trajectory(trajectory,  reward_key='information')
		ctrl_results = self.ctrl.train()
		com_results  = self.com.train()
		#lang_results = self.language.train(corpus)
		return ctrl_results, com_results#, lang_results


	def gather(self, n):
		data = []
		for _ in range(n):
			x, y = 3, 5
			grid = GridWorld(x, y)
			i = randint(2, x * 3 + 1)
			j = randint(2, y * 3 + 1)
			grid.state['alpha'] = (i, j)
			#print(len(grid.grid.squares))
			#print(len(grid.grid.squares[0]))
			#exit()
			observations = []
			for _ in range(4):
				actions = randint(0, 4), randint(0, 4)
				obs, rewards, dones = grid.step(actions)
				observations.append(obs[0])
			data.append(observations)
		batch = tf.constant(data)
		return batch



class Environment:
	def __init__(self, agent, env_class):
		self.agent = agent()
		self.env_class = env_class
		self.env   = self.env_class()
		self.obs   = self.env.reset()
		self.reset()


	def reset(self):
		self.trajectory = {'alpha': defaultdict(list), 
				   'beta':  defaultdict(list)}
		#self.corpus = []
		self.env = self.env_class()
		self.obs = self.env.reset()
		return self.obs


	def step(self, training=False):
		obs_a, obs_b = self.obs
		# EMBEDDINGS
		emb_a  = self.agent.observe(obs_a[None,...])#, training=training)
		emb_b  = self.agent.observe(obs_b[None,...])#, training=training)
		# MESSAGES
		mes_a, logit_a = self.agent.speak(emb_a,training=training)
		mes_b, logit_b = self.agent.speak(emb_b,training=training)
		# BABBLING (additional messages for language model)
		#bab_a  = self.agent.babble(emb_a, training=training)
		#bab_b  = self.agent.babble(emb_b, training=training)
		# ACTING
		act_a  = self.agent.act(emb_a, mes_a)
		act_b  = self.agent.act(emb_b, mes_b)
		# MUTUAL INFORMATION
		info_a = self.agent.information(mes_a, logit_a)
		info_b = self.agent.information(mes_b, logit_b)
		# ENV STEP
		self.obs, rewards, dones = self.env.step((act_a, act_b))
		(rew_a, rew_b), (done_a, done_b) = rewards, dones
		# SAVE
		alpha_data = (obs_a, emb_a, mes_a, act_a, rew_a, info_a, done_a)
		beta_data  = (obs_b, emb_b, mes_b, act_b, rew_b, info_b, done_b)
		self.save('alpha', *alpha_data)
		self.save('beta',  *beta_data)
		return self.obs, rewards, dones


	def save(self, agent, obs, emb, mes, act, rew, info, done):
		trajectory = self.trajectory[agent]
		if trajectory['dones'] and trajectory['dones'][-1]:
			return
		trajectory['observations'].append(obs)
		trajectory['embeddings'].append(emb)
		trajectory['messages'].append(mes)
		#self.corpus.extend(bab)
		trajectory['actions'].append(act)
		trajectory['rewards'].append(rew)
		trajectory['information'].append(info)
		trajectory['dones'].append(done)


	def train_ctrl(self):
		self.agent.parse_trajectory(self.trajectory['alpha'])
		self.agent.parse_trajectory(self.trajectory['beta'])
		return self.agent.train()


	def train(self, episodes):
		with tqdm(total=episodes) as bar:
			done = False
			obs = self.reset()
			while not done:
				obs, rewards, dones = self.step()
				done = np.all(dones)
			trajectories = [self.trajectory['alpha'], self.trajectory['beta']]
			ctrl_results, com_results = self.agent.train(trajectories)
			bar.update(1)
		
		




if __name__ == '__main__':
	from gridworld import StackedSingleGridWorld
	#env = Environment(Agent, GridWorld)
	enc = AutoEncoder()
	enc.load_weights('autoencoder')
	#env.step()
	mes    = tf.constant([[4, 5]])
	agent  = BaseAgent(actor=ActorCtrl, critic=CriticCtrl)
	epochs = 100
	Ts = []
	for epoch in range(epochs):
		trajectory = {'embeddings': [], 'rewards': [], 'message': [], 'actions': []}
		env  = StackedSingleGridWorld(T=300, X=2, Y=5)
		obs  = env.reset()[None,...]
		done = False
		while not done:
			emb    = enc.encoder(obs)
			# TRAINIG == TRUE IS DANGER ZONE
			logits = agent.actor(emb, mes)#, training=True)
			cat    = tfp.distributions.Categorical(logits=logits)
			act    = cat.sample()
			#print(logits)
			obs, rew, done = env.step(act)
			obs = obs[None,...]
			trajectory['embeddings'].append(emb)
			trajectory['rewards'].append(rew)
			trajectory['message'].append(mes)
			trajectory['actions'].append(act)
		agent.parse_trajectory(trajectory, reward_key='rewards', com=True)
		actor_target, critic_loss = agent.train()
		print(env)
		print('EPOCH', epoch)
		print('RETURN', sum(trajectory['rewards']))
		print('ACTOR', actor_target, 'CRITIC', critic_loss)
		print('LEN', env.t)
		Ts.append(env.t)
	from matplotlib import pyplot as plt
	plt.scatter(range(epochs), Ts)
	plt.show()
		
	
