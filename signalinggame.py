import tensorflow as tf
import numpy as np
from dataset import generate



class SignalingGame:
	RATIO = 0.5
	def __init__(self, batch_size, n_images, horizon=8):
		self.batch_size = batch_size
		self.n_images   = n_images
		self.dataset    = (obj for obj in generate(n_images, batch_size))
		self.horizon    = horizon
		self.embedding  = VGG = tf.keras.applications.vgg16.VGG16(include_top=False)
		self.pooling    = tf.keras.layers.GlobalAveragePooling2D()


	def reset(self):
		self.time = 0
		self.img_A, self.img_B, self.idx = next(self.dataset)
		img_A = tf.concat(tf.unstack(self.img_A, axis=1), axis=0)
		img_B = tf.concat(tf.unstack(self.img_B, axis=1), axis=0)
		emb_A = self.pooling(self.embedding(img_A))
		emb_B = self.pooling(self.embedding(img_B))
		self.emb_A = tf.stack(tf.split(emb_A, self.n_images, axis=0), axis=1)
		self.emb_B = tf.stack(tf.split(emb_B, self.n_images, axis=0), axis=1)
		return self.emb_A, self.emb_B


	def step(self, actions):
		# TIME STEP
		self.time += 1
		done = self.time == self.horizon
		# CONTROL ACTIONS
		actions_A, actions_B = actions
		actions_A = tf.cast(actions_A[:,:,0], dtype=tf.int64)
		actions_B = tf.cast(actions_B[:,:,0], dtype=tf.int64)
		correct_A = tf.cast(tf.reduce_all(self.idx == actions_A, axis=1), dtype=tf.float32)
		correct_B = tf.cast(tf.reduce_all(self.idx == actions_B, axis=1), dtype=tf.float32)
		reward_A  = self.RATIO * correct_A + (1 - self.RATIO) * correct_B
		reward_B  = self.RATIO * correct_B + (1 - self.RATIO) * correct_A
		# TERMINATION
		emb_A, emb_B = self.emb_A, self.emb_B
		dones = tf.stack([done] * self.batch_size, axis=0)
		if done:
			self.reset()
		return (emb_A, emb_B), (reward_A, reward_B), (dones, dones)



if __name__ == '__main__':
	batch_size = 2
	n_images   = 4
	x = tf.range(2 * 3 * 4)
	x = tf.reshape(x, (2, 3 * 4))
	print(x)
	y = tf.reshape(x, (2, 4, 3))
	print(y)
	z = tf.reshape(y, (2, 3 * 4))
	print(z)
	#exit()
	env = SignalingGame(batch_size=batch_size, n_images=n_images)
	obs_A, obs_B = env.reset()
	print(env.idx[0,...])
	#img_A = tf.concat(tf.unstack(obs_A[0,...], axis=1), axis=0)
	#img_B = tf.concat(tf.unstack(obs_B[0,...], axis=1), axis=0)
	#img   = tf.concat([img_A, img_B], axis=1)
	#img   = img - np.min(img)
	#img   = img / np.max(img)
	#from matplotlib import pyplot as plt
	#plt.imshow(img)
	#plt.show()

	done = False
	while not done:
		actions  = np.random.randint(2, size=(batch_size, n_images))
		actions  = tf.stack([actions, actions], axis=1)
		messages = actions
		values_A, values_B = env.step(actions, messages)
		enb_A, rew_A, done_A = values_A
		enb_B, rew_B, done_B = values_B
		print(emb_A.shape, emb_B.shape)
		print(rew_A[0,...].numpy(), actions[0, 0,...].numpy(), env.idx[0,...].numpy())
		done = done_A or done_B
