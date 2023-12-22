import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np



#encoder
#communicator
#language
#actor
#critic



class Encoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.conv_0  = tf.keras.layers.Conv2D(kernel_size=3, 
						      filters=64, 
						      padding='same')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.conv_1  = tf.keras.layers.Conv2D(kernel_size=2, 
						      filters=64, 
						      padding='valid')
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.conv_2  = tf.keras.layers.Conv2D(kernel_size=2, 
						      filters=64, 
						      padding='same')
		self.norm_2  = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()
		self.dense   = tf.keras.layers.Dense(64)


	def call(self, observation, training=False):
		# BLOCK 0
		x = self.conv_0(observation, training=training)
		x = self.norm_0(x, training=training)
		x = tf.nn.elu(x)
		# BLOCK 1
		x = self.conv_1(x, training=training)
		x = self.norm_1(x, training=training)
		x = tf.nn.elu(x)
		# BLOCK 2
		x = self.conv_2(x, training=training)
		x = self.norm_2(x, training=training)
		x = tf.nn.elu(x)
		# OUTPUT
		x = self.flatten(x)
		x = self.dense(x)
		return tf.nn.elu(x)



class Decoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dense   = tf.keras.layers.Dense(4 * 64)
		self.reshape = tf.keras.layers.Reshape((2, 2, 64))
		self.conv_0  = tf.keras.layers.Conv2DTranspose(kernel_size=2, 
							       filters=64)
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.conv_1  = tf.keras.layers.Conv2DTranspose(kernel_size=3, 
							       filters=64)
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.conv_2  = tf.keras.layers.Conv2D(kernel_size=2, 
						      filters=6, 
						      padding='same')
		self.norm_2  = tf.keras.layers.BatchNormalization()


	def call(self, embedding, training=False):
		# UPSAMPLING
		x = self.dense(embedding, training=training)
		x = self.reshape(x)
		# BLOCK 0
		x = self.conv_0(x, training=training)
		x = self.norm_0(x, training=training)
		x = tf.nn.elu(x)
		# BLOCK 1
		x = self.conv_1(x, training=training)
		x = self.norm_1(x, training=training)
		x = tf.nn.elu(x)
		# BLOCK 2
		x = self.conv_2(x, training=training)
		x = self.norm_2(x, training=training)
		x = tf.nn.sigmoid(x)
		return x



class CtrlActor(tf.keras.Model):
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
		self.dense_2 = tf.keras.layers.Dense(action_dims)


	def listen(self, message, training=False):
		embedding = self.embedding(message, training=training)
		return self.gru(embedding, training=training)


	def call(self, obs_emb, message, training=False):
		mes_emb = self.listen(message, training=training)
		x = tf.concat([obs_emb, mes_emb], axis=1)
		x = self.dense_0(x, training=training)
		x = self.norm_0( x, training=training)
		x = self.dense_1(x, training=training)
		x = self.norm_1( x, training=training)
		x = self.dense_2(x, training=training)
		return x


	def posterior(self, obs_emb, message, training=False):
		x = self(obs_emb, message, training=training)
		return tf.nn.softmax(x, axis=1)


	def log_posterior(self, obs_emb, message, training=False):
		#     e^a  /    (e^a + e^b + e^c)
		# log(e^a  /    (e^a + e^b + e^c))
		# log(e^a) - log(e^a + e^b + e^c)
		# a        - log(e^a + e^b + e^c)
		x      = self(obs_emb, message, training=training)
		norms  = tf.reduce_sum(tf.math.exp(x))
		logits = x - tf.math.log(norms)
		return logits



class CtrlCritic(tf.keras.Model):
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
		self.dense_2 = tf.keras.layers.Dense(action_dims)


	def listen(self, message, training=False):
		embedding = self.embedding(message, training=training)
		return self.gru(embedding, training=training)


	def call(self, obs_emb, message, training=False):
		mes_emb = self.listen(message, training=training)
		x = tf.concat([obs_emb, mes_emb], axis=1)
		x = self.dense_0(x, training=training)
		x = self.norm_0( x, training=training)
		x = self.dense_1(x, training=training)
		x = self.norm_1( x, training=training)
		x = self.dense_2(x, training=training)
		return x


	def posterior(self, obs_emb, message, training=False):
		x = self(obs_emb, message, training=training)
		return tf.nn.softmax(x, axis=1)


	def log_posterior(self, obs_emb, message, training=False):
		#     e^a  /    (e^a + e^b + e^c)
		# log(e^a  /    (e^a + e^b + e^c))
		# log(e^a) - log(e^a + e^b + e^c)
		# a        - log(e^a + e^b + e^c)
		x      = self(obs_emb, message, training=training)
		norms  = tf.reduce_sum(tf.math.exp(x))
		logits = x - tf.math.log(norms)
		return logits



if __name__ == '__main__':
	enc = Encoder()
	dec = Decoder()
	com = ComActor()
	lan = Language()
	act = CtrlActor()
	batch_size = 2
	obs = tf.ones((batch_size, 5, 5, 6))
	emb = enc(obs)
	rec = dec(emb)
	#print(rec.shape)
	#print(rec)
	#exit()
	#print(emb.shape)
	#exit()
	message = com(emb)
	#print('MESSAGE')
	#print(message)
	#print(lan.log_prior(message))
	#print(lan.prior(message))
	#print(message.shape)
	#print(lan.loss(message))
	#print(message)
	print(tf.reduce_sum(act.posterior(emb, message), axis=1))
	#print(act.posterior(emb, message))
	#print(act.log_posterior(emb, message))
