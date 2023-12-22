import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
from stackedautoencoder import AutoEncoder
from communication import Language, ActorCom, CriticCom
from control import ActorCtrl, CriticCtrl

from gridworld import GridWorld, StackedMultiGridWorld, VectorStackedMultiEnv, VectorPartialStackedMultiEnv
from random import randint
from pickle import dump, load



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
		self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def ctrl_logits(self, embeddings, actions, messages, training=False):
		return self.ctrl.log_posterior(embeddings, messages, actions, training=training)


	def com_logits(self, embeddings, messages, training=False):
		return self.com.log_posterior(messages, embeddings, training=training)


	def call(self, embeddings, actions, messages, training=False):
		ctrl_logits = self.ctrl.log_posterior(embeddings, messages, actions, training=training)
		com_logits  = self.com.log_posterior(messages, embeddings, training=training)
		return ctrl_logits + com_logits



class Critic(tf.keras.Model):
	def __init__(self, 
		     vocab_size, 
		     embed_size,
		     learning_rate=1e-3,  
		     **kwargs):
		super().__init__(**kwargs)
		self.ctrl = CriticCtrl(vocab_size=vocab_size, 
				       embed_size=embed_size)
		self.com  = CriticCom(vocab_size=vocab_size, 
				      embed_size=embed_size)
		self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def com_value(self, emb, training=False):
		return self.com(emb, training=training)


	def ctrl_value(self, emb, mes, training=False):
		return self.ctrl(emb, mes, training=training)



class Agent:
	def __init__(self, 
		     vocab_size, 
		     max_length, 
		     embed_size, 
		     gamma=0.99, 
		     lambd=0.95, 
		     alpha=1.0):
		self.gamma = gamma
		self.lambd = lambd
		self.alpha = alpha
		self.auto  = AutoEncoder()
		self.auto.load_weights('auto_base')
		self.auto.trainable = False
		self.actor  = Actor(vocab_size=vocab_size, 
				    max_length=max_length, 
				    embed_size=embed_size)
		self.critic = Critic(vocab_size=vocab_size, 
				     embed_size=embed_size)


	def parse_trajectory(self, trajectory):
		embeddings_a = tf.stack(trajectory['emb_alpha'  ], axis=0)
		embeddings_b = tf.stack(trajectory['emb_beta'   ], axis=0)
		rewards      = tf.stack(trajectory['rewards'    ], axis=0)
		information  = tf.stack(trajectory['information'], axis=0)
		rewards      = tf.cast(rewards, dtype=tf.float32) + self.alpha * tf.cast(information, dtype=tf.float32)
		actions      = tf.stack(trajectory['actions'])
		messages     = tf.stack(list(map(tf.ragged.stack, trajectory['messages'])))
		advantages_a, advantages_b = [], []
		returns_a,    returns_b    = [], []
		# CTRL
		for i in range(embeddings_a.shape[0]):
			values = self.critic.ctrl_value(embeddings_a[i,1:,...], messages[i,...])
			returns_a.append(self.L_RET(values,  rewards[i,...]))
			advantages_a.append(self.GAE(values, rewards[i,...]))
		# COM
		for i in range(embeddings_b.shape[0]):
			values = self.critic.com_value(embeddings_b[i,1:,...])
			returns_b.append(self.L_RET(values,  rewards[i,...]))
			advantages_b.append(self.GAE(values, rewards[i,...]))
		embeddings_a = tf.concat(list(embeddings_a[:,:-1,...]), axis=0)
		embeddings_b = tf.concat(list(embeddings_b[:,:-1,...]), axis=0)
		advantages_a = tf.concat(advantages_a,                  axis=0)
		advantages_b = tf.concat(advantages_b,                  axis=0)
		returns_a    = tf.concat(returns_a,                     axis=0)
		returns_b    = tf.concat(returns_b,                     axis=0)
		actions      = tf.concat(list(actions),                 axis=0)
		messages     = tf.concat(list(messages),                axis=0)
		return embeddings_a[:-1,...], embeddings_b[:-1,...], advantages_a, advantages_b, returns_a, returns_b, actions, messages


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


	def observe(self, observation, training=False):
		return self.auto.encoder(observation, training=training)


	def speak(self, embedding, training=False):
		return self.actor.com(embedding, return_logits=True, training=training)[:2]


	def babble(self, embedding, training=False):
		return [self.actor.com(embedding, training=training)[0] for _ in range(self.babble_number)]


	def act(self, embedding, message, training=False):
		logits = self.actor.ctrl(embedding, message, training=training)
		cat    = tfp.distributions.Categorical(logits=logits)
		action = cat.sample()
		return action


	def information(self, message, log_posterior, batch_size=16, training=False):
		batch     = self.gather(batch_size)
		embedding = self.observe(batch)
		log_prior = []
		for i in range(message.shape[0]):
			messages  = tf.stack([message[i,...]] * batch_size, axis=0)
			logit     = self.actor.com.log_posterior(messages, embedding, training=training)
			log_prior.append(tf.reduce_mean(logit))
		log_prior = tf.stack(log_prior)
		return log_posterior - log_prior


	def train(self, trajectory, critic_epochs=4, critic_batch=32):
		#EMB_A, EMB_B, ADV, RET, ACT, MSG = [], [], [], [], [], []
		#for trajectory in trajectories:
		#	emb_a, amb_b, adv, ret, act, msg = self.parse_trajectory(trajectory)
		#	EMB_A.append(emb_a)
		#	EMB_B.append(emb_b)
		#	ADV.append(adv)
		#	RET.append(ret)
		#	ACT.append(act)
		#	MSG.append(msg)
		EMB_A, EMB_B, ADV_A, ADV_B, RET_A, RET_B, ACT, MSG = self.parse_trajectory(trajectory)
		embeddings_a = tf.concat(EMB_A, axis=0)
		embeddings_b = tf.concat(EMB_B, axis=0)
		advantages_a = tf.concat(ADV_A, axis=0)
		advantages_b = tf.concat(ADV_B, axis=0)
		returns_a    = tf.concat(RET_A, axis=0)
		returns_b    = tf.concat(RET_B, axis=0)
		actions      = tf.concat(ACT,   axis=0)
		messages     = tf.concat(MSG,   axis=0)
		data = list(zip(embeddings_a, embeddings_b, advantages_a, advantages_b, returns_a, returns_b, actions, messages))
		shuffle(data)
		embeddings_a, embeddings_b, advantages_a, advantages_b, returns_a, returns_b, actions, messages = map(list, zip(*data))
		embeddings_a = tf.stack(embeddings_a,    axis=0)
		embeddings_b = tf.stack(embeddings_b,    axis=0)
		advantages_a = tf.stack(advantages_a,    axis=0)
		advantages_b = tf.stack(advantages_b,    axis=0)
		returns_a    = tf.stack(returns_a,       axis=0)
		returns_b    = tf.stack(returns_b,       axis=0)
		actions      = tf.stack(actions,         axis=0)
		messages     = tf.ragged.stack(messages, axis=0).with_row_splits_dtype(tf.int64)
		# ACTOR CTRL
		with tf.GradientTape() as tape:
			log_probs = self.actor.ctrl_logits(embeddings_a, 
							   actions, 
							   messages, 
							   training=True)
			actor_ctrl_loss = - tf.reduce_mean(log_probs * advantages_a)
		gradients = tape.gradient(actor_ctrl_loss, self.actor.ctrl.trainable_variables)
		self.actor.ctrl.adam.apply_gradients(zip(gradients, self.actor.ctrl.trainable_weights))
		# ACTOR COM
		with tf.GradientTape() as tape:
			log_probs = self.actor.com_logits(embeddings_b, 
							  messages, 
							  training=True)
			actor_com_loss = - tf.reduce_mean(log_probs * advantages_b)
		gradients = tape.gradient(actor_com_loss, self.actor.com.trainable_variables)
		self.actor.com.adam.apply_gradients(zip(gradients, self.actor.com.trainable_weights))
		# CRITIC CTRL
		batch_size = embeddings_a.shape[0]
		for i in range(critic_epochs):
			for j in range(int(np.ceil(batch_size/critic_batch))):
				emb = embeddings_a[ j * critic_batch: (j + 1) * critic_batch,...]
				mes = messages[     j * critic_batch: (j + 1) * critic_batch,...]
				ret = returns_a[    j * critic_batch: (j + 1) * critic_batch,...]
				with tf.GradientTape() as tape:
					val = self.critic.ctrl_value(emb, mes, training=True)
					critic_ctrl_loss = tf.reduce_mean((val - ret)**2)
				gradients = tape.gradient(critic_ctrl_loss, self.critic.ctrl.trainable_variables)
				self.critic.ctrl.adam.apply_gradients(zip(gradients, self.critic.ctrl.trainable_weights))
		# CRITIC COM
		for i in range(critic_epochs):
			for j in range(int(np.ceil(batch_size/critic_batch))):
				emb = embeddings_b[ j * critic_batch: (j + 1) * critic_batch,...]
				ret = returns_b[    j * critic_batch: (j + 1) * critic_batch,...]
				with tf.GradientTape() as tape:
					val = self.critic.com(emb, training=True)
					critic_com_loss = tf.reduce_mean((val - ret)**2)
				gradients = tape.gradient(critic_com_loss, self.critic.com.trainable_variables)
				self.critic.com.adam.apply_gradients(zip(gradients, self.critic.com.trainable_weights))
		return actor_ctrl_loss, actor_com_loss, critic_ctrl_loss, critic_com_loss


	def gather(self, n):
		data = []
		for _ in range(n):
			x, y = 3, 5
			grid = GridWorld(x, y)
			i = randint(2, x * 3 + 1)
			j = randint(2, y * 3 + 1)
			grid.state['alpha'] = (i, j)
			observations = []
			for _ in range(4):
				actions = randint(0, 4), randint(0, 4)
				obs, rewards, dones = grid.step(actions)
				observations.append(obs[0])
			data.append(observations)
		batch = tf.constant(data)
		return batch



class Environment:
	def __init__(self, 
		     env_class, 
		     info_scale, 
		     vocab_size, 
		     max_length, 
		     embed_size, 
		     number_env, 
		     horizon=30):
		self.horizon = horizon
		self.agent   = Agent(alpha=info_scale, 
				     vocab_size=vocab_size, 
				     max_length=max_length, 
				     embed_size=embed_size)
		self.env_class  = env_class
		self.number_env = number_env
		self.reset()


	def reset(self):
		self.trajectory = defaultdict(lambda: [[] for _ in range(self.number_env)])
		self.env = self.env_class(T=self.horizon, 
					  W=self.number_env)
		self.obs = self.env.reset()
		return self.obs


	def step(self, training=False):
		# ALPHA: act
		# BETA:  speak
		obs_a, obs_b = self.obs
		# EMBEDDINGS
		emb_a  = self.agent.observe(obs_a)
		emb_b  = self.agent.observe(obs_b)
		# MESSAGES
		mes, logit_mes = self.agent.speak(emb_b)
		# ACTING
		act    = self.agent.act(emb_a, mes)
		# MUTUAL INFORMATION
		info   = self.agent.information(mes, logit_mes)
		# ENV STEP
		self.obs, rew, done = self.env.step(act)
		# SAVE
		data = (obs_a, obs_b, emb_a, emb_b, mes, act, rew, info, done)
		self.save(*data)
		return self.obs, rew, done


	def save(self, obs_a, obs_b, emb_a, emb_b, mes, act, rew, info, done):
		for i, d in enumerate(done):
			if len(self.trajectory['dones'][i]) > 0 and self.trajectory['dones'][i][-1]:
				continue
			self.trajectory['obs_alpha'  ][i].append(obs_a[i,...])
			self.trajectory['obs_beta'   ][i].append(obs_b[i,...])
			self.trajectory['emb_alpha'  ][i].append(emb_a[i,...])
			self.trajectory['emb_beta'   ][i].append(emb_b[i,...])
			self.trajectory['messages'   ][i].append(mes  [i,...])
			self.trajectory['actions'    ][i].append(act  [i,...])
			self.trajectory['rewards'    ][i].append(rew  [i,...])
			self.trajectory['information'][i].append(info [i,...])
			self.trajectory['dones'      ][i].append(d)


	def train(self, episodes):
		with tqdm(total=episodes) as bar:
			stats = {'actor com':   [], 
				 'actor ctrl':  [], 
				 'critic com':  [], 
				 'critic ctrl': [], 
				 'returns':     [], 
				 'info':        [], 
				 'messages':    []}
			for episode in range(episodes):
				done = False
				obs  = self.reset()
				while not done:
					obs, rewards, dones = self.step()
					done = np.all(dones)
				# LAST EMBEDDING
				obs_a, obs_b = self.obs
				emb_a  = self.agent.observe(obs_a)
				emb_b  = self.agent.observe(obs_b)
				for i in range(self.number_env):
					self.trajectory['emb_alpha'][i].append(emb_a[i,...])
					self.trajectory['emb_beta' ][i].append(emb_b[i,...])
				actor_ctrl_loss, actor_com_loss, critic_ctrl_loss, critic_com_loss = self.agent.train(self.trajectory)
				returns     = [sum(rewards) for rewards in self.trajectory['rewards']]
				returns     =  sum(returns)/len(returns)
				information = [sum(info)/len(info) for info in self.trajectory['information']]
				information =  sum(information)/len(information)
				print(*map(lambda x: x.numpy()[1:-1], self.trajectory['messages'][0][:10]), sep='\n')
				bar.update(1)
				bar.set_postfix({'actor ctrl':  actor_ctrl_loss.numpy(), 
						 'actor com':   actor_com_loss.numpy(), 
						 'critic ctrl': critic_ctrl_loss.numpy(), 
						 'critic com':  critic_com_loss.numpy(), 
						 'return':      returns, 
						 'info':        information.numpy()})
				stats['actor ctrl' ].append(actor_ctrl_loss.numpy())
				stats['actor com'  ].append(actor_com_loss.numpy())
				stats['critic ctrl'].append(critic_ctrl_loss.numpy())
				stats['critic com' ].append(critic_com_loss.numpy())
				stats['returns'    ].append(returns)
				stats['info'       ].append(information.numpy())
				stats['messages'   ].append(tf.ragged.stack(self.trajectory['messages'][0])[:,1:-1])
		return stats



if __name__ == '__main__':
	info_scale = 1
	vocab_size = 8
	max_length = 4
	embed_size = 3
	number_env = 5
	episodes   = 6
	trials     = 7
	file_name  = 'results.data'
	stats      = []
	for trial in range(trials):
		env = Environment(VectorPartialStackedMultiEnv, 
				  info_scale=info_scale, 
				  vocab_size=vocab_size, 
				  max_length=max_length, 
				  embed_size=embed_size, 
				  number_env=number_env)
		stat = env.train(episodes)
		stats.append(stat)
	with open(file_name, 'wb') as file:
		dump(stats, file)
	#with open(file_name, 'rb') as file:
	#	stats = load(file)
	print(stats)