import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np



def generate(n_img, batch_size=16):
	dataset = tfds.load('imagenet2012_subset', split='validation', download=True)
	print(dataset.cardinality())
	dataset = dataset.map(lambda x: tf.image.resize(x['image'], (224, 224)))
	dataset = dataset.map(tf.keras.applications.vgg16.preprocess_input)
	while True:
		data = dataset.shuffle(1024)
		data = data.batch(2)
		data = data.batch(n_img)
		data = data.map(lambda x: (x, np.random.randint(2, size=n_img)))
		data = data.map(lambda x, i: (x[:,1,...], tf.gather(x, i, batch_dims=1, axis=1), i))
		data = data.batch(batch_size)
		for obs_a, obs_b, idx in data:
			yield obs_a, obs_b, idx
	#return data



if __name__ == '__main__':
	gen = generate(4)
	
	img_a, img_b, idx = next(iter(gen))
	img_a, img_b, idx = img_a[0,...], img_b[0,...], idx[0,...]
	print(img_a.shape, img_b.shape)
	#print(idx)
	#exit()
	img_a = tf.concat(tf.unstack(img_a, axis=0), axis=1)
	img_b = tf.concat(tf.unstack(img_b, axis=0), axis=1)
	img = tf.concat([img_a, img_b], axis=0)
	img = img - np.min(img)
	print(np.max(img))
	img = img / np.max(img)
	#print(img)
	print(idx)
	print(img.shape)
	
	from matplotlib import pyplot as plt
	plt.imshow(img.numpy())
	plt.show()
	

