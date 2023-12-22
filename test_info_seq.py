import tensorflow as tf
import numpy as np
import random
from tensorflow_probability import distributions as tfd



n_samples  = 4
n_symbols  = 8
batch_size = 64
value_size = 16
max_length = 1
learning_rate = 1e-2

max_entropy = max_length * tf.math.log(float(n_symbols))



X = np.random.normal(size=(n_samples, value_size))
X = tf.cast(X, dtype=tf.float32)




def Batch(batch_size):
	obs_a, obs_b, target = [], [], []
	for i in range(batch_size):
		idx_a = random.choice(list(range(n_samples)))
		coin  = random.choice([True, False])
		coin = idx_a % 2
		if coin:
			idx_b = random.choice(list(range(idx_a)) + list(range(idx_a + 1, n_samples)))
		else:
			idx_b = idx_a
		obs_a.append(X[idx_a,...])
		obs_b.append(X[idx_b,...])
		target.append(tf.cast(coin, dtype=tf.float32))
	return tf.stack(obs_a, axis=0), tf.stack(obs_b, axis=0), tf.stack(target, axis=0)



class Speak(tf.keras.Model):
	def __init__(self, n_symbols, max_length, value_size, **kwargs):
		super().__init__(**kwargs)
		self.emb = tf.keras.layers.Dense(value_size, activation='silu')
		self.gru = tf.keras.layers.GRU(value_size, return_state=True, return_sequences=True)
		self.out = tf.keras.layers.Dense(n_symbols + 1, activation='softmax')
		self.n_symbols  = n_symbols
		self.max_length = max_length
		self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def get_begining_token(self, batch_size):
		return tf.repeat(tf.one_hot(self.n_symbols + 1, depth=self.n_symbols + 2, dtype=tf.float32)[None,None,:], batch_size, axis=0)

	
	def get_ending_token(self, batch_size):
		return tf.repeat(tf.one_hot(self.n_symbols, depth=self.n_symbols + 2, dtype=tf.float32)[None,None,:], batch_size, axis=0)
		

	def call(self, state):
		batch_size = state.shape[0]
		eos    = self.get_ending_token(batch_size)
		token  = self.get_begining_token(batch_size)	
		tokens = []
		hidden = self.emb(state)
		probs  = []
		for i in range(max_length):
			# PREDICTION
			pred, hidden = self.gru(token, initial_state=hidden)
			prob   = self.out(tf.squeeze(pred))
			# SAMPLING
			symbol = tfd.Categorical(probs=prob).sample()
			symbol = tf.argmax(prob, axis=1)
			token  = tf.one_hot(symbol, depth=self.n_symbols + 2, dtype=tf.float32)[:,None,:]
			tokens.append(token)
			probs.append(prob)
		tokens   = tf.concat(tokens, axis=1)
		probs    = tf.stack(probs, axis=1)
		#print(probs)
		term     = tf.transpose(tf.reduce_all(tokens == eos, axis=2), (1, 0))
		mask     = tf.scan(tf.math.logical_or, term, tf.fill(batch_size, False))
		mask     = tf.transpose(mask, (1, 0))
		mask_eos = tf.concat([tf.fill(batch_size, False)[:,None], mask[:,:-1]], axis=1)
		symbols  = tf.argmax(tokens, axis=2)
		tokens   = tf.ragged.boolean_mask(tokens,  tf.logical_not(mask))
		probs    = tf.ragged.boolean_mask(probs,   tf.logical_not(mask_eos))
		symbols  = tf.ragged.boolean_mask(symbols, tf.logical_not(mask_eos))
		probs    = tf.gather(probs, symbols, batch_dims=2, axis=2)
		return symbols, tokens, probs


	def logits(self, state, symbols):
		batch_size = state.shape[0]
		tokens  = tf.one_hot(symbols, depth=self.n_symbols + 2, dtype=tf.float32)
		inputs  = tf.concat([self.get_begining_token(batch_size), tokens], axis=1)
		outputs = tf.concat([tokens, self.get_ending_token(batch_size)], axis=1)
		hidden  = self.emb(state)
		pred, state = self.gru(inputs, initial_state=hidden)
		probs   = self.out(pred)
		probs = tf.gather(probs[:,:-1,:], symbols[:,:,None], batch_dims=2, axis=2)
		logits  = tf.reduce_sum(tf.math.log(1e-8 + probs), axis=(1, 2))
		return logits



class Listen(tf.keras.Model):
	def __init__(self, value_size, **kwargs):
		super().__init__(**kwargs)
		self.emb = tf.keras.layers.Dense(value_size, activation='silu')
		self.gru = tf.keras.layers.GRU(value_size, return_sequences=False)
		self.out = tf.keras.layers.Dense(1, activation='sigmoid')
		self.n_symbols  = n_symbols
		self.adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def get_begining_token(self, batch_size):
		return tf.repeat(tf.one_hot(self.n_symbols + 1, depth=self.n_symbols + 2, dtype=tf.float32)[None,None,:], batch_size, axis=0)

	
	def get_ending_token(self, batch_size):
		return tf.repeat(tf.one_hot(self.n_symbols, depth=self.n_symbols + 2, dtype=tf.float32)[None,None,:], batch_size, axis=0)
		
		
	def call(self, state, token):
		batch_size = state.shape[0]
		hidden  = self.emb(state)
		#inputs = tf.concat([token, hidden[:,None,:]], axis=2)
		#pred    = self.gru(inputs)
		pred    = self.gru(token, initial_state=hidden)
		probs   = self.out(tf.squeeze(pred))
		actions = tf.cast(probs > tf.random.uniform((batch_size, 1)), dtype=tf.float32)
		actions = tf.squeeze(actions)
		return actions, probs


	def logits(self, state, token, action):
		pred = self(state, token)
		prob = action * pred + (1 - action) * (1 - pred)
		return tf.math.log(1e-8 + prob)








speak  = Speak(n_symbols=n_symbols, 
	       max_length=max_length, 
	       value_size=value_size)
listen = Listen(value_size=value_size)






epochs = 1000
Rs = []
for epoch in range(epochs):
	obs_a, obs_b, target = Batch(batch_size)
	with tf.GradientTape(persistent=True) as tape:
		symbols, tokens, probs_speak = speak(obs_a)
		logits_speak = tf.reduce_sum(tf.math.log(probs_speak), axis=1)
		actions, probs_listen = listen(obs_a, tokens)
		logits_listen = tf.math.log(1e-6 + probs_listen)
		rewards = tf.cast(actions == target, dtype=tf.float32) - 0.5
		loss_speak  = - tf.squeeze(logits_speak)  * rewards
		loss_listen = - tf.squeeze(logits_listen) * rewards
		#loss_listen = tf.reduce_mean((probs_listen - 1 + target)**2)
		#print(loss_speak)
		#print(loss_listen)
		#exit()
	#gradients = tape.gradient(loss_speak,      speak.trainable_weights)
	#speak.adam.apply_gradients(zip(gradients,  speak.trainable_weights))
	gradients = tape.gradient(loss_listen,     listen.trainable_weights)
	listen.adam.apply_gradients(zip(gradients, listen.trainable_weights))
	#print(symbols)
	print(tf.reduce_mean(rewards).numpy())
	Rs.append(tf.reduce_mean(rewards).numpy())
	print(tf.reduce_mean(loss_speak).numpy(), tf.reduce_mean(loss_listen).numpy())
	
from matplotlib import pyplot as plt


plt.plot(range(len(Rs)), Rs)
plt.show()