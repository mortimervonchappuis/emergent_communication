import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from symfunc import symlog
from transformer import Transformer



class ActorCtrl(tf.keras.Model):
	def __init__(self, action_dims=5, vocab_size=8, embed_size=2, **kwargs):
		super().__init__(**kwargs)
		self.action_dims = action_dims
		self.vocab_size  = vocab_size
		self.embedding   = tf.keras.layers.Embedding(vocab_size + 3, embed_size)
		self.gru         = tf.keras.layers.GRU(64, 
						       return_sequences=False, 
						       return_state=False)
		self.transformer = Transformer(num_heads=8, key_dim=64, emb_dim=4)
		self.dense_in  = tf.keras.layers.Dense(64, activation='silu')
		self.dense_out = tf.keras.layers.Dense(1, activation='sigmoid')
		#self.norm_0  = tf.keras.layers.BatchNormalization()
		#self.dense_1 = tf.keras.layers.Dense(64, activation='elu')
		#self.norm_1  = tf.keras.layers.BatchNormalization()
		#self.dense_2 = tf.keras.layers.Dense(action_dims)
		self.adam    = tf.keras.optimizers.Adam(learning_rate=1e-3)


	def listen(self, message, training=False):
		embedding = self.embedding(message, training=training)
		return self.gru(embedding, training=training)


	def call(self, obs_emb, message, training=False):
		mes_emb = self.listen(message, training=training)
		# obs_emb (B, S, key), mes_emb = (B, emb)
		#print(obs_emb.shape, mes_emb.shape)
		#print(obs_emb)
		obs_emb = self.dense_in(obs_emb, training=training)
		#print(obs_emb.shape, mes_emb.shape)
		#exit()
		emb   = self.transformer(obs_emb, mes_emb, training=training)
		probs = self.dense_out(emb)
		return probs

		x = tf.concat([obs_emb, mes_emb], axis=1)
		x = self.dense_0(x, training=training)
		x = self.norm_0( x, training=training)
		x = self.dense_1(x, training=training)
		x = self.norm_1( x, training=training)
		x = self.dense_2(x, training=training)
		return x


	def posterior(self, obs_emb, message, training=False):
		return self(obs_emb, message, training=training)


	def log_posterior(self, obs_emb, message, actions=None, training=False):
		x      = self(obs_emb, message, training=training)
		logits = tf.reduce_sum(tf.math.log(1e-8 + x), axis=1)[...,None]
		return logits


	def train(self, data):
		embeddings,  advantages, actions, message = data
		with tf.GradientTape() as tape:
			logits = self.log_posterior(embeddings, message)
			logits = tf.gather(logits, actions, batch_dims=1)[...,0]
			loss = - tf.reduce_mean(logits * advantages)
		gradients = tape.gradient(loss, self.trainable_weights)
		self.adam.apply_gradients(zip(gradients, self.trainable_weights))
		return loss.numpy()



class CriticCtrl(tf.keras.Model):
	def __init__(self, action_dims=5, vocab_size=8, embed_size=2, **kwargs):
		super().__init__(**kwargs)
		self.action_dims = action_dims
		self.vocab_size  = vocab_size
		self.embedding   = tf.keras.layers.Embedding(vocab_size + 3, embed_size)
		self.gru         = tf.keras.layers.GRU(64, 
						       return_sequences=False, 
						       return_state=False)
		self.dense_0 = tf.keras.layers.Dense(64, activation='elu')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.dense_1 = tf.keras.layers.Dense(64, activation='elu')
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.dense_2 = tf.keras.layers.Dense(1)
		self.adam    = tf.keras.optimizers.Adam(learning_rate=1e-3)


	def listen(self, message, training=False):
		embedding = self.embedding(message, training=training)
		return self.gru(embedding, training=training)


	def call(self, obs_emb, message, training=False):
		mes_emb = self.listen(message, training=training)
		obs_emb = tf.concat(obs_emb, axis=0)
		x = tf.concat([obs_emb, mes_emb], axis=1)
		x = self.dense_0(x, training=training)
		x = self.norm_0( x, training=training)
		x = self.dense_1(x, training=training)
		x = self.norm_1( x, training=training)
		x = self.dense_2(x, training=training)
		return x[...,0]


	def train(self, data):
		embeddings, returns, message = data
		with tf.GradientTape() as tape:
			values = self(embeddings, message)
			loss   = tf.reduce_mean(symlog(returns - values)**2)
		gradients = tape.gradient(loss, self.trainable_weights)
		self.adam.apply_gradients(zip(gradients, self.trainable_weights))
		return loss.numpy()



if __name__ == '__main__':
	act = ActorCtrl()
	crt = CriticCtrl()
	emb = np.ones((2, 64))
	mes = tf.ragged.constant([[5, 3, 0, 0, 1, 4], [5, 0, 3, 4]])
	print(act.posterior(emb, mes))
	print(crt(emb, mes))

