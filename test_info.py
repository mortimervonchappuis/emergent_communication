import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np



batch_size = 16
n_symbols  = 8
value_size = 32
X = np.random.normal(size=(batch_size, value_size))

model = tf.keras.layers.Dense(n_symbols, activation='softmax')
model = tf.keras.layers.GRU(n_symbols)
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)


epochs = 10000
for epoch in range(epochs):
	with tf.GradientTape() as tape:
		probs = model(X)
		logits = tf.math.log(probs)
		information = - probs * logits
		#print(tf.reduce_sum(probs, axis=0))
		
		dist = tfd.Categorical(logits=logits)
		x    = dist.sample()
		#posterior = tf.gather(logits, x, batch_dims=1)
		#prior     = tf.gather(tf.reduce_mean(logits, axis=0), x)

		#prior = tf.reduce_sum(probs * logits, axis=0)[None,...]
		#P(M) = sum P(M, O)
		#H(M) = - sum P(m) log P(m)
		prior = tf.reduce_mean(probs, axis=0) # sum / batch_size == mean
		log_prior = tf.math.log(prior)
		prior_entropy = - tf.reduce_sum(prior * log_prior)
		#P(M|O) = f
		#H(M|O) = - sum P(M, O) log P(M|O)
		posterior = probs
		log_posterior = tf.math.log(posterior)
		posterior_entropy = - tf.reduce_sum(posterior/batch_size * log_posterior)
		#posterior = tf.reduce_sum(logits, axis=1)
		#print(posterior.shape, prior.shape)
		mutual_information = prior_entropy - posterior_entropy
		#mutual_information = tf.reduce_mean(posterior - prior)
		#print((posterior - prior).shape)
		#print(mutual_information.shape)
		#exit()
		correlation = tf.linalg.matmul(probs, probs, transpose_b=True)
		#print(correlation)
		correlation = 10 * tf.math.log(1e-8 + tf.reduce_mean(correlation))
		#exit()
		loss = correlation * 0 - mutual_information
	gradients = tape.gradient(loss, model.trainable_weights)
	adam.apply_gradients(zip(gradients, model.trainable_weights))
	print(mutual_information.numpy())
	print(correlation.numpy())
	#print(probs[0,...].numpy())
	print(x.numpy())
	print(prior_entropy.numpy(), posterior_entropy.numpy())
	print(tf.math.argmax(probs, axis=1))
	argmax = list(tf.math.argmax(probs, axis=1))
	frequencies = dict((x, argmax.count(x)) for x in range(n_symbols))
	print(np.std(list(frequencies.values())))
	print(dict((x, argmax.count(x)) for x in range(n_symbols)))
	print(np.mean(np.std(probs, axis=1))) 
	#print(gradients)

