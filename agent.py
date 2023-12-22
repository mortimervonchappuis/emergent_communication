import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
from stackedautoencoder import AutoEncoder
from communication import Language, ActorCom, CriticCom
from control import ActorCtrl, CriticCtrl
from gridworld import GridWorld, StackedMultiGridWorld, VectorStackedMultiEnv
from random import randint
from transformer import SetTransformer



class Actor(tf.keras.Model):
	def __init__(self, 
		     vocab_size, 
		     max_length, 
		     embed_size,
		     learning_rate=1e-3,  
		     **kwargs):
		super().__init__(**kwargs)
		self.ctrl = ActorCtrl(vocab_size=vocab_size, 
				      embed_size=embed_size)
		self.com  = ActorCom(vocab_size=vocab_size, 
				     max_length=max_length, 
				     embed_size=embed_size)
		self.com_embed  = SetTransformer(num_heads=8, 
						 key_dim=512, 
						 emb_dim=1)
		self.ctrl_embed = SetTransformer(num_heads=8, 
					         key_dim=512, 
					         emb_dim=4)
		#self.ctrl_linear = tf.keras.layers.Dense(64, activation='silu')
		self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def call(self, embeddings, actions, messages_in, messages_out, training=False):
		embedding_com  = self.com_embed(embeddings,  training=training)[:,0,:]
		com_logits     = self.com.log_posterior(messages_out, embedding_com, training=training)
		embedding_ctrl = self.ctrl_embed(embeddings,  training=training)
		ctrl_logits    = self.ctrl.log_posterior(embedding_ctrl, messages_in, actions, training=training)
		return ctrl_logits + com_logits



class Agent:
	def __init__(self, 
		     vocab_size, 
		     max_length, 
		     embed_size, 
		     gamma=0.99, 
		     lambd=0.95, 
		     alpha=1.0):
		self.gamma  = gamma
		self.lambd  = lambd
		self.alpha  = alpha
		self.auto   = AutoEncoder()
		self.auto.load_weights('auto_base')
		self.auto.trainable = False
		self.actor  = Actor(vocab_size=vocab_size, 
				    max_length=max_length, 
				    embed_size=embed_size)
		self.critic = CriticCtrl(vocab_size=vocab_size, 
					 embed_size=embed_size)


	def parse_trajectory(self, trajectory):
		embeddings   = tf.stack(trajectory['observations' ], axis=0)
		rewards      = tf.stack(trajectory['rewards'    ], axis=0)
		information  = tf.stack(trajectory['information'], axis=0)
		#
		rewards      = tf.cast(rewards, dtype=tf.float32) + self.alpha * tf.cast(information, dtype=tf.float32)
		#
		actions      = tf.stack(trajectory['actions'])
		messages_in  = tf.stack(list(map(tf.ragged.stack, trajectory['messages_in' ])))
		messages_out = tf.stack(list(map(tf.ragged.stack, trajectory['messages_out'])))
		advantages, returns = [], []
		for i in range(embeddings.shape[0]):
			#values = self.critic(embeddings[i,1:,...], messages_in[i,...])
			#returns.append(self.L_RET(values, rewards[i,...]))
			returns.append(self.RET(rewards[i,...]))
			#advantages.append(self.GAE(values, rewards[i,...]))
		embeddings   = tf.concat(list(embeddings[:,:-1,...]), axis=0)
		#advantages   = tf.concat(advantages,                  axis=0)
		returns      = tf.concat(returns,                     axis=0)
		actions      = tf.concat(list(actions),               axis=0)
		messages_in  = tf.concat(list(messages_in),           axis=0)
		messages_out = tf.concat(list(messages_out),          axis=0)
		#return embeddings[:-1,...], advantages, returns, actions, messages_in, messages_out
		return embeddings[:-1,...], returns, actions, messages_in, messages_out


	def RET(self, rewards):
		rewards = tf.concat([rewards, [0.0]], axis=0)
		def scan(returns, rewards):
			return rewards + self.gamma * returns
		return tf.scan(scan, rewards, reverse=True)


	def L_RET(self, values, rewards):
		def scan(l_ret, elems):
			value, reward = elems
			return reward + self.gamma * ((1 - self.lambd) * value + self.lambd * l_ret)
		return tf.scan(scan, (values, rewards), 0.0, reverse=True)


	def GAE(self, values, rewards):
		shifted   = tf.concat([values[1:], [values[-1]]], axis=0)
		temp_diff = rewards + self.gamma * shifted - values
		def scan(advantage, temp_diff):
			return temp_diff + self.gamma * self.lambd * advantage
		return tf.scan(scan, temp_diff, reverse=True)


	def speak(self, observations, training=False):
		embedding = self.actor.com_embed(observations,  training=training)[:,0,:]
		return self.actor.com(embedding, return_logits=True, training=training)[:2]


	def act(self, observations, message, training=False):
		embedding = self.actor.ctrl_embed(observations,  training=training)
		#embedding = self.actor.ctrl_linear(embedding, training=training)
		probs  = self.actor.ctrl(embedding, message, training=training)
		dist   = tfp.distributions.Bernoulli(probs=probs)
		action = dist.sample()
		return action


	def information(self, observations, message, log_posterior, training=False):
		embedding  = self.actor.com_embed(observations,  training=training)[:,0,:]
		batch_size = message.shape[0]
		log_prior  = []
		for i in range(batch_size):
			messages  = tf.stack([message[i,...]] * (batch_size - 1), axis=0)
			part_emb  = tf.concat([embedding[:i,...], embedding[i+1:,...]], axis=0)
			logit     = self.actor.com.log_posterior(messages, part_emb, training=training)
			log_prior.append(tf.reduce_mean(logit))
		log_prior = tf.stack(log_prior)
		return log_posterior - log_prior


	def train(self, trajectories):
		EMB, ADV, RET, ACT, MSI, MSO = [], [], [], [], [], []
		for trajectory in trajectories:
			emb, ret, act, m_i, m_o = self.parse_trajectory(trajectory)
			#emb, adv, ret, act, m_i, m_o = self.parse_trajectory(trajectory)
			EMB.append(emb)
			#ADV.append(adv)
			RET.append(ret)
			ACT.append(act)
			MSI.append(m_i)
			MSO.append(m_o)
		embeddings   = tf.concat(EMB, axis=0)
		#advantages   = tf.concat(ADV, axis=0)
		returns      = tf.concat(RET, axis=0)
		actions      = tf.concat(ACT, axis=0)
		messages_in  = tf.concat(MSI, axis=0)
		messages_out = tf.concat(MSO, axis=0)
		#data = list(zip(embeddings, advantages, returns, actions, messages_in, messages_out))
		data = list(zip(embeddings, returns, actions, messages_in, messages_out))
		shuffle(data)
		#embeddings, advantages, returns, actions, messages_in, messages_out = map(list, zip(*data))
		embeddings, returns, actions, messages_in, messages_out = map(list, zip(*data))
		embeddings   = tf.stack(embeddings, axis=0)
		#advantages   = tf.stack(advantages, axis=0)
		returns      = tf.stack(returns,    axis=0)
		actions      = tf.stack(actions,    axis=0)
		messages_in  = tf.ragged.stack(messages_in,  axis=0).with_row_splits_dtype(tf.int64)
		messages_out = tf.ragged.stack(messages_out, axis=0).with_row_splits_dtype(tf.int64)
		# ACTOR
		with tf.GradientTape() as tape:
			log_probs = self.actor(embeddings, 
					       actions, 
					       messages_in, 
					       messages_out, 
					       training=True)
			actor_loss = tf.reduce_mean(- log_probs * returns)
			#actor_loss = tf.reduce_mean(- log_probs * avantages)
		gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor.adam.apply_gradients(zip(gradients, self.actor.trainable_weights))
		return actor_loss
		critic_epochs = 4
		critic_batch  = 32
		batch_size    = embeddings.shape[0]
		for i in range(critic_epochs):
			for j in range(int(np.ceil(batch_size/critic_batch))):
				emb = embeddings[ j * critic_batch: (j + 1) * critic_batch,...]
				mes = messages_in[j * critic_batch: (j + 1) * critic_batch,...]
				ret = returns[    j * critic_batch: (j + 1) * critic_batch,...]
				with tf.GradientTape() as tape:
					val = self.critic(emb, 
							  mes, 
							  training=True)
					critic_loss = tf.reduce_mean((val - ret)**2)
				gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
				self.critic.adam.apply_gradients(zip(gradients, self.critic.trainable_weights))
		return actor_loss, critic_loss



class Environment:
	def __init__(self, 
		     env_class, 
		     vocab_size, 
		     max_length, 
		     embed_size, 
		     number_env, 
		     horizon=7):
		self.horizon = horizon
		self.agent   = Agent(vocab_size=vocab_size, 
				     max_length=max_length, 
				     embed_size=embed_size)
		self.env = env_class(horizon=horizon, 
				     batch_size=number_env,
				     n_images=4)
		self.number_env = number_env
		self.reset()


	def reset(self):
		self.trajectory = {'alpha': defaultdict(lambda: [[] for _ in range(self.number_env)]), 
				   'beta':  defaultdict(lambda: [[] for _ in range(self.number_env)])}
		self.obs = self.env.reset()
		return self.obs


	def step(self, training=False):
		#obs_a, obs_b = self.obs
		obs_a, obs_b = self.obs
		# EMBEDDINGS
		#emb_a = self.embedding(obs_a)
		#emb_b = self.embedding(obs_b)
		#print(emb_a.shape, emb_b.shape)
		#exit()
		# MESSAGES
		mes_a, logit_a = self.agent.speak(obs_a, training=training)
		mes_b, logit_b = self.agent.speak(obs_b, training=training)
		# BABBLING (additional messages for language model)
		#bab_a  = self.agent.babble(emb_a, training=training)
		#bab_b  = self.agent.babble(emb_b, training=training)
		# ACTING
		act_a  = self.agent.act(obs_a, mes_a)
		act_b  = self.agent.act(obs_b, mes_b)
		#print(act_a, act_b)
		# MUTUAL INFORMATION
		info_a = self.agent.information(obs_a, mes_a, logit_a)
		info_b = self.agent.information(obs_b, mes_b, logit_b)
		# ENV STEP
		self.obs, rewards, dones = self.env.step((act_a, act_b))
		(rew_a, rew_b), (done_a, done_b) = rewards, dones
		# SAVE
		alpha_data = (obs_a, mes_a, mes_b, act_a, rew_a, info_a, done_a)
		beta_data  = (obs_b, mes_b, mes_a, act_b, rew_b, info_b, done_b)
		self.save('alpha', *alpha_data)
		self.save('beta',  *beta_data)
		return self.obs, rewards, dones


	def save(self, agent, obs, mes_out, mes_in, act, rew, info, done):
		trajectory = self.trajectory[agent]
		for i, d in enumerate(done):
			if len(trajectory['dones'][i]) > 0 and trajectory['dones'][i][-1]:
				continue
			trajectory['observations'][i].append(obs[i,...])
			trajectory['messages_in' ][i].append(mes_in[i,...])
			trajectory['messages_out'][i].append(mes_out[i,...])
			#self.corpus.extend(bab)
			trajectory['actions'     ][i].append(act[i,...])
			trajectory['rewards'     ][i].append(rew[i,...])
			trajectory['information' ][i].append(info[i,...])
			trajectory['dones'       ][i].append(d)


	def train(self, episodes):
		with tqdm(total=episodes) as bar:
			for episode in range(episodes):
				done = False
				obs  = self.reset()
				while not done:
					obs, rewards, dones = self.step()
					done = np.all(dones)
				# LAST EMBEDDING
				obs_a, obs_b = self.obs
				#emb_a  = self.agent.observe(obs_a)
				#emb_b  = self.agent.observe(obs_b)
				for i in range(self.number_env):
					self.trajectory['alpha']['observations'][i].append(obs_a[i,...])
					self.trajectory['beta' ]['observations'][i].append(obs_b[i,...])
				trajectories  = [self.trajectory['alpha'], self.trajectory['beta']]
				actor_loss = self.agent.train(trajectories)
				returns_alpha = [sum(rewards) for rewards in self.trajectory['alpha']['rewards']]
				returns_beta  = [sum(rewards) for rewards in self.trajectory['beta' ]['rewards']]
				returns       = returns_alpha + returns_beta
				avr_return    = sum(returns)/len(returns)
				info_alpha    = [sum(info)/len(info) for info in self.trajectory['alpha']['information']]
				info_beta     = [sum(info)/len(info) for info in self.trajectory['beta' ]['information']]
				avr_info      = 0.5 * (sum(info_alpha)/len(info_alpha) + sum(info_beta)/len(info_beta))
				print(*self.trajectory['alpha']['messages_out'][i][:10], sep='\n')
				bar.update(1)
				bar.set_postfix({'actor':  actor_loss.numpy(), 
						 #'critic': critic_loss.numpy(), 
						 'return': avr_return, 
						 'info':   avr_info.numpy()})



if __name__ == '__main__':
	vocab_size = 8
	max_length = 4
	embed_size = 3
	number_env = 5
	from signalinggame import SignalingGame
	env = Environment(SignalingGame, 
			  vocab_size=vocab_size, 
			  max_length=max_length, 
			  embed_size=embed_size, 
			  number_env=number_env)
	env.train(100)
	"""
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
		
	
"""