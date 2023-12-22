import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm



class Encoder(tf.keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.conv_0  = tf.keras.layers.Conv2D(kernel_size=3, 
						      filters=64, 
						      padding='same')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.conv_1  = tf.keras.layers.Conv2D(kernel_size=3, 
						      filters=64, 
						      padding='valid')
		self.norm_1  = tf.keras.layers.BatchNormalization()
		self.conv_2  = tf.keras.layers.Conv2D(kernel_size=2, 
						      filters=64, 
						      padding='valid')
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
						      filters=9, 
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
		x = tf.nn.softmax(x, axis=3)
		return x



class Autoencoder(tf.keras.Model):
	def __init__(self, learning_rate=1e-4, **kwargs):
		super().__init__(**kwargs)
		self.encoder      = Encoder()
		self.decoder      = Decoder()
		self.crossentropy = tf.keras.losses.CategoricalCrossentropy()
		self.optimizer    = tf.keras.optimizers.Adam(learning_rate=learning_rate)


	def call(self, observations, training=False):
		embedding = self.encoder(observations, training=training)
		reconst   = self.decoder(embedding,    training=training)
		return reconst


	def loss(self, observations, training=False):
		reconst = self(observations, training=training)
		return self.crossentropy(observations, reconst)


	def gather(self, n, m):
		from gridworld import GridWorld
		from random import randint
		data = []
		for _ in range(n):
			x, y = 3, 5
			grid = GridWorld(x, y)
			#print(len(grid.grid.squares))
			#print(len(grid.grid.squares[0]))
			#exit()
			for _ in range(m):
				i, j = randint(0, (x - 1) * 3 + 2), randint(0, (y - 1) * 3 + 2)
				patch = grid[i:5+i, j:5+j]
				obs   = grid.encode(patch)
				data.append(obs)
				#print(i, j)
				#print(patch)
		batch = tf.constant(data)
		return batch


	def train(self, steps, plot=False):
		losses = []
		with tqdm(total=steps) as bar:
			for step in range(steps):
				batch = self.gather(8, 4)
				with tf.GradientTape() as tape:
					loss = self.loss(batch)
				gradients = tape.gradient(loss, self.trainable_weights)
				self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
				losses.append(loss.numpy())
				bar.update(1)
		if plot:
			from matplotlib import pyplot as plt
			plt.plot(range(steps), losses)
			plt.xlabel('steps')
			plt.ylabel('loss')
			plt.show()
		return losses


	def reconstruct(self, observations):
		reconst    = self(observations)
		max_val    = tf.math.reduce_max(reconst, axis=-1)[:,:,:,None]
		image      = tf.cast(max_val == reconst, dtype=tf.float64)
		comparison = image == observations
		correct    = tf.math.argmax(reconst, axis=-1) == tf.math.argmax(observations, axis=-1)
		accuracy   = tf.math.reduce_mean(tf.cast(correct, dtype=tf.float64))
		#return comparison
		return accuracy.numpy()



if __name__ == '__main__':
	#batch = gather(8, 4)
	#print(batch)
	#exit()
	obs = np.ones((2, 5, 5, 9))
	auto = Autoencoder()
	print(auto.reconstruct(auto.gather(32, 4)))
	exit()
	auto.train(10, plot=True)
	auto.save_weights('autoencoder')
	print(auto.reconstruct(auto.gather(8, 4)))
	print(auto.loss(obs))
	print(tf.reduce_sum(auto(obs), axis=3))
