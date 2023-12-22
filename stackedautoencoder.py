import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
from math import tau, e
from symfunc import symlog



class Encoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.conv_0  = tf.keras.layers.Conv3D(kernel_size=(2, 3, 3), 
						      filters=32, 
						      padding='same')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.conv_1  = tf.keras.layers.Conv3D(kernel_size=(2, 3, 3), 
						      filters=48, 
						      padding='valid')
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.conv_2  = tf.keras.layers.Conv3D(kernel_size=(3, 2, 2), 
						      filters=64, 
						      padding='valid')
		self.norm_2  = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()
		self.dense   = tf.keras.layers.Dense(128) # 256
		# 1024, 0.72
		#  512, 0.73
		#  256, 0.69
		#  128, 0.67

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
		x = self.dense(x, training=training)
		#x = tf.nn.elu(x)
		return x



class Decoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dense = tf.keras.layers.Dense(4 * 64)
		self.reshape = tf.keras.layers.Reshape((1, 2, 2, 64))
		self.conv_0  = tf.keras.layers.Conv3DTranspose(kernel_size=(3, 2, 2), 
							       filters=48, 
							       padding='valid')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.conv_1  = tf.keras.layers.Conv3DTranspose(kernel_size=(2, 3, 3), 
							       filters=32, 
							       padding='valid')
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.conv_2  = tf.keras.layers.Conv3D(kernel_size=(2, 3, 3), 
						      filters=9, 
						      padding='same')
		self.norm_2  = tf.keras.layers.BatchNormalization()


	def logits(self, embedding, training=False):
		# UPSAMPLING
		x = self.dense(embedding, training=training)
		x = tf.nn.elu(x)
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
		return x


	def call(self, embeddings, training=False):
		x = self.logits(embeddings, training=training)
		x = tf.nn.softmax(x, axis=-1)
		return x



class AutoEncoder(tf.keras.Model):
	def __init__(self, learning_rate=1e-5, **kwargs): # 3e-3
		# 1e-2, 7.47
		super().__init__(**kwargs)
		heavyside      = lambda x: tf.cast(x > 0., dtype=x.dtype)
		self.heavyside = heavyside
		self.diff_step = lambda x: heavyside(x) + tf.nn.sigmoid(x) - tf.stop_gradient(tf.nn.sigmoid(x))
		self.encoder      = Encoder()
		self.decoder      = Decoder()
		self.dropout      = tf.keras.layers.Dropout(0.00)
		self.crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
		self.optimizer    = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def call(self, observations, training=False):
		embedding = self.encoder(observations, training=training)
		embedding = self.activation(embedding)
		reconst   = self.decoder(embedding,    training=training)
		return reconst


	def logits(self, observations, training=False):
		embedding = self.encoder(observations, training=training)
		embedding = self.activation(embedding)
		reconst   = self.decoder.logits(embedding, training=training)
		return reconst


	def activation(self, x, training=False):
		# GUMBLE SOFT MAX
		#sig_x = tf.nn.sigmoid(x)
		#uniform = tf.random.uniform(shape=x.shape)
		#gumble  = -tf.math.log(-tf.math.log(uniform))
		#embedding = tf.nn.sigmoid((gumble + tf.math.log(sig_x))/t)
		# STRAIGHT THROUGH
		#cont = tf.nn.sigmoid(x) # 0.65
		#cont = symlog(x) # 0.63
		#embedding = self.heavyside(x) + cont - tf.stop_gradient(cont)
		#embedding = self.dropout(embedding, training=training)
		embedding = tf.nn.elu(x)
		return embedding



	def loss(self, observations, training=False):
		# ENCODING
		embedding    = self.encoder(observations, training=training)
		# MIXING
		#mu_log  = 0.0 # 0.0
		#mu_sig  = 0.9 # 1.0
		#eta     = 0.4 # 0.0
		#mu_lin  = 0.2 # 0.0
		#sig_emb = tf.nn.sigmoid(embedding)
		#smooth = tf.nn.elu(embedding)
		#log_emb = symlog(embedding)
		#step   = self.heavyside(embedding)
		#linear = embedding * mu_lin
		#embedding = mu * smooth + (1 - mu) * (step + smooth - tf.stop_gradient(smooth))
		#embedding = mu_log * log_emb + (1 - mu_log) * (sig_emb   + eta * (log_emb - tf.stop_gradient(log_emb)))
		#embedding = mu_sig * step    + (1 - mu_sig) * (embedding)#+ eta * (log_emb - tf.stop_gradient(log_emb)))
		#embedding = embedding + linear - tf.stop_gradient(linear) * 0.2
		#embedding = step + embedding + sig_emb + log_emb - tf.stop_gradient(embedding + sig_emb + log_emb)
		
		#embedding = step + embedding * 0.2 + log_emb - tf.stop_gradient(log_emb)
		#embedding = step + log_emb - tf.stop_gradient(log_emb)
		#embedding = step + sig_emb - tf.stop_gradient(sig_emb)
		#embedding = sig_emb
		#embedding = log_emb

		#sig_log = (1 + log_emb) * 0.5

		#uniform = tf.random.uniform(shape=embedding.shape)
		#gumble  = -tf.math.log(-tf.math.log(uniform))
		#embedding = tf.nn.sigmoid((gumble + tf.math.log(sig_emb))/temp)
		embedding = self.activation(embedding, training=training)


		#embedding = mixing * step + (1 - mixing) * sig_emb + sig_log - tf.stop_gradient(sig_log)
		#embedding = mixing * step + (1 - mixing) * sig_emb + embedding * 0.1 - tf.stop_gradient(embedding * 0.1)
		#embedding = tf.nn.elu(embedding)
		#embedding = step + log_emb

		# DECODING
		reconst      = self.decoder(embedding, training=training)

		weight = np.array([0.19348371, 
				   0.20477606, 
				   0.1813844, 
				   0.18138356, 
				   0.01477567, 
				   0.06327286, 
				   0.08619132, 
				   0.03554482, 
				   0.0391876])
		stats  = np.array([0.36941406, 
				   0.15759766, 
				   0.32519531, 
				   0.07865234, 
				   0.00861328, 
				   0.01337891, 
				   0.01697266, 
				   0.0140625, 
				   0.01611328]) # Probability for each square type

		# LOSS
		#crossentropy = self.crossentropy(observations, reconst)
		cr_right = - tf.reduce_mean(tf.reduce_sum(0.005/stats[None,None,None,None,...] * reconst * tf.math.log(tf.cast(observations, dtype=tf.float32) + 1e-6), axis=-1))
		#cr_left  = - tf.reduce_mean(tf.reduce_sum(tf.cast(observations, dtype=tf.float32) * tf.math.log(reconst + 1e-6), axis=-1))
		cr_left  = - tf.reduce_mean(tf.reduce_sum(weight[None,None,None,None,...]**0.3 * 0.1/stats[None,None,None,None,...] * tf.cast(observations, dtype=tf.float32) * tf.math.log(reconst + 1e-6), axis=-1))
		crossentropy = cr_left #+ cr_right
		# QUANTITIES
		prob = tf.math.reduce_mean(embedding, axis=0)
		diff = prob - embedding
		diff = tf.math.sign(diff) * diff
		covariance = tf.math.reduce_mean(tf.math.sqrt((diff[:,None,:] * diff[:,:None])**2))
		entropy = - tf.reduce_mean(prob  * tf.math.log(1e-6 +     prob) \
				    + (1 - prob) * tf.math.log(1e-6 + 1 - prob)) / tf.math.log(2.0)
		print(crossentropy.numpy(), entropy.numpy(), covariance.numpy())
		return crossentropy #+ covariance# - entropy # - 0.1 * covariance


	def gather(self, n, m):
		from gridworld import GridWorld
		from random import randint
		data = []
		for _ in range(n):
			x, y = 2, 5
			grid = GridWorld(X=x, Y=y)
			i = randint(2, x * 3 + 1)
			j = randint(2, y * 3 + 1)
			grid.state['alpha'] = (i, j)
			observations = []
			for _ in range(m):
				actions = randint(0, 4), randint(0, 4)
				obs, rewards, dones = grid.step(actions)
				observations.append(obs[0])
			data.append(observations)
		batch = tf.constant(data)
		#print(grid)
		return batch


	def train(self, epochs, steps, plot=False):
		losses     = []
		accuracies = []
		with tqdm(total=epochs) as bar:
			for epoch in range(epochs):
				for step in range(steps):
					batch = self.gather(64, 4) # 256
					with tf.GradientTape() as tape:
						loss = self.loss(batch, training=True)
					gradients = tape.gradient(loss, self.trainable_weights)
					self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
					losses.append(loss.numpy())
				bar.update(1)
				accuracy = self.accuracy(self.gather(64, 4))
				accuracies.append(accuracy)
				bar.set_postfix({'acc': accuracy, 'loss': loss.numpy()})
		if plot:
			from matplotlib import pyplot as plt
			#plt.plot(np.linspace(0, steps * epochs, steps * epochs), losses)
			plt.plot(np.linspace(0, steps * epochs, epochs),         accuracies)
			plt.xlabel('steps')
			plt.ylabel('loss/accuracy')
			plt.show()
		return losses, accuracies


	def accuracy(self, observations):
		stats = np.array([0.36941406, 
				  0.15759766, 
				  0.32519531, 
				  0.07865234, 
				  0.00861328, 
				  0.01337891, 
				  0.01697266, 
				  0.0140625, 
				  0.01611328])
		reconst    = self(observations)
		max_val    = tf.math.reduce_max(reconst, axis=-1)[:,:,:,None]
		image      = tf.cast(max_val == reconst, dtype=tf.float64)
		comparison = image == observations
		correct    = tf.math.argmax(reconst, axis=-1) == tf.math.argmax(observations, axis=-1)
		accuracy   = tf.math.reduce_mean(tf.cast(correct, dtype=tf.float64))
		failed     = list(tf.math.argmax(observations, axis=-1)[~correct])
		cat_acc    = {i: failed.count(i)/(reconst.shape[0] * stats[i]) for i in range(9)}
		print(cat_acc)
		#return comparison
		return accuracy.numpy()



class MiniCoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dense_0 = tf.keras.layers.Dense(1024)
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.dense_1 = tf.keras.layers.Dense(1024)
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.dense_2 = tf.keras.layers.Dense(1024)
		self.norm_2  = tf.keras.layers.BatchNormalization()
		self.dense_3 = tf.keras.layers.Dense(1024)
		self.norm_3  = tf.keras.layers.BatchNormalization()
		self.auto    = AutoEncoder()
		self.auto.load_weights('autoencoder_1024_stg_symlog_weigthed')
		self.auto.trainable = False
		self.heavyside = lambda x: tf.cast(x > 0., dtype=x.dtype)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


	def gather(self, *args, **kwargs):
		return self.auto.gather(*args, **kwargs)


	def train(self, epochs, steps, plot=False):
		losses     = []
		accuracies = []
		with tqdm(total=epochs) as bar:
			for epoch in range(epochs):
				for step in range(steps):
					batch = self.gather(64, 4) # 256
					with tf.GradientTape() as tape:
						loss = self.loss(batch, training=True)
					gradients = tape.gradient(loss, self.trainable_weights)
					self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
					losses.append(loss.numpy())
				bar.update(1)
				accuracy = self.consistency(self.gather(256, 4))
				accuracies.append(accuracy)
				bar.set_postfix({'acc': accuracy, 'loss': loss.numpy()})
		if plot:
			from matplotlib import pyplot as plt
			plt.plot(np.linspace(0, steps * epochs, steps * epochs), losses)
			plt.plot(np.linspace(0, steps * epochs, epochs),         accuracies)
			plt.xlabel('steps')
			plt.ylabel('loss/accuracy')
			plt.show()
		return losses, accuracies
		


	def activation(self, x):
		cont = symlog(x)
		embedding = self.heavyside(x) + cont - tf.stop_gradient(cont)
		return embedding



	def loss(self, observations, training=False):
		x = self.auto.encoder(observations, training=training)
		embedding = self.activation(x)
		x = self.encode(embedding, training=training)
		reconst = self.decode(x,  training=training)
		reconst = tf.nn.sigmoid(reconst)
		crossentropy = -tf.math.reduce_mean(embedding * tf.math.log(1e-6 + reconst) + (1 - embedding) * tf.math.log(1e-6 + 1 - reconst))

		prob = tf.math.reduce_mean(x, axis=0)
		#print(prob)
		diff = prob - x
		diff = tf.math.sign(diff) * diff
		covariance = tf.math.reduce_mean((diff[:,None,:] * diff[:,:,None])**2)
		entropy = - tf.reduce_mean(prob  * tf.math.log(1e-6 +     prob) \
				    + (1 - prob) * tf.math.log(1e-6 + 1 - prob)) / tf.math.log(2.0)
		print(crossentropy.numpy(), entropy.numpy(), covariance.numpy())
		return crossentropy + covariance * 10 - entropy


	def call(self, observations, training=False):
		x = self.auto.encoder(observations, training=training)
		x = self.activation(x)
		x = self.encode(x, training=training)
		x = self.decode(x, training=training)
		x = self.activation(x)
		return self.auto.decoder(x, training=training)


	def encode(self, embedding, training=False):
		x = self.dense_0(embedding)
		x = self.norm_0(x, training=training)
		x = tf.nn.elu(x)
		x = self.dense_1(x)
		x = self.norm_1(x, training=training)
		return self.activation(x)


	def decode(self, encoding, training=False):
		x = self.dense_2(encoding)
		x = self.norm_2(x, training=training)
		x = tf.nn.elu(x)
		x = self.dense_3(x)
		x = self.norm_3(x, training=training)
		return x


	def consistency(self, observations):
		embedding  = self.auto.encoder(observations)
		embedding  = self.activation(embedding)
		encoding   = self.encode(embedding)
		decoding   = self.decode(encoding)
		decoding   = self.activation(decoding)
		return tf.math.reduce_mean(embedding * decoding + (1 - embedding) * (1 - decoding)).numpy()


	def accuracy(self, observations):
		stats = np.array([0.36111328, 
				  0.15224609, 
				  0.34851562, 
				  0.07541016, 
				  0.00833984,
				  0.01363281, 
				  0.01429688, 
				  0.01386719, 
				  0.01257813])
		reconst    = self(observations)
		max_val    = tf.math.reduce_max(reconst, axis=-1)[:,:,:,None]
		image      = tf.cast(max_val == reconst, dtype=tf.float64)
		comparison = image == observations
		correct    = tf.math.argmax(reconst, axis=-1) == tf.math.argmax(observations, axis=-1)
		accuracy   = tf.math.reduce_mean(tf.cast(correct, dtype=tf.float64))
		failed     = list(tf.math.argmax(observations, axis=-1)[~correct])
		cat_acc    = {i: failed.count(i)/(reconst.shape[0] * stats[i]) for i in range(9)}
		print(cat_acc)
		#return comparison
		return accuracy.numpy()



if __name__ == '__main__':
	#batch = gather(8, 4)
	#print(batch)
	#exit()
	obs = np.ones((2, 4, 5, 5, 9))
	auto = AutoEncoder()
	#auto = MiniCoder()
	#print(auto(obs).shape)
	#exit()
	#conv = tf.keras.layers.Conv3D(9, (2, 3, 3))
	#print(conv(obs).shape)
	# 0.74
	
	auto.load_weights('auto_base')
	print(tf.math.reduce_sum(auto.gather(512, 4), axis=(0, 1, 2, 3)))
	print(auto.accuracy(auto.gather(32, 4)))
	auto.train(100, 20, plot=True)
	auto.save_weights('auto_base')
	print(auto.accuracy(auto.gather(128, 4)))
