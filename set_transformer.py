import tensorflow as tf



class MAB(tf.keras.Model):
	def __init__(self, num_heads, key_dim, activation='silu', **kwargs):
		super().__init__(**kwargs)
		self.MHA = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
		self.LN1 = tf.keras.layers.LayerNormalization()
		self.LN2 = tf.keras.layers.LayerNormalization()
		self.RFF = tf.keras.layers.Dense(key_dim, activation=activation)


	def call(self, X, Y, training=False):
		H = self.LN1(X + self.MHA(X, Y, Y, training=training), training=training)
		Z = self.LN2(H + self.RFF(H,       training=training), training=training)
		return Z



class SAB(MAB):
	def call(self, X, training=False):
		return super().call(X, X, training=training)



class PMA(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, activation='silu', **kwargs):
		super().__init__(**kwargs)
		self.S   = tf.Variable(tf.random.normal(shape=(1, emb_dim, key_dim)))
		self.RFF = tf.keras.layers.Dense(key_dim, activation=activation)
		self.MAB = MAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, training=False):
		batch_size = X.shape[0]
		S = tf.repeat(self.S, [batch_size], axis=0)
		Z = self.RFF(X, training=training)
		Z = self.MAB(S, Z, training=training)
		return Z



class Decoder(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, **kwargs):
		super().__init__(**kwargs)
		self.RFF = tf.keras.layers.Dense(key_dim)
		self.PMA = PMA(num_heads=num_heads, key_dim=key_dim, emb_dim=emb_dim)
		self.SAB = SAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, training=False):
		Z = self.PMA(X, training=training)
		Z = self.SAB(Z, training=training)
		Y = self.RFF(Z, training=training)
		return Y



if __name__ == '__main__':
	batch_size = 16
	set_size   = 4
	key_dim    = 8
	num_heads  = 12
	emb_dim    = 1

	from random import shuffle
	X1 = tf.random.normal(shape=(batch_size, set_size, key_dim))
	Xs = tf.unstack(X1, axis=1)
	shuffle(Xs)
	X2 = tf.stack(Xs, axis=1)
	
	dec = Decoder(num_heads=num_heads, key_dim=key_dim, emb_dim=emb_dim)
	
	Y1 = dec(X1)
	Y2 = dec(X2)
	print(tf.math.round(Y1 - Y2, 6))

	epochs = 1000
	adam = tf.keras.optimizers.Adam(learning_rate=3e-5)
	for e in range(epochs):
		X = tf.random.normal(shape=(batch_size, set_size, key_dim))
		X_mean   = tf.math.reduce_mean(X, axis=2)
		X_argmax = tf.math.argmax(X_mean, axis=1)
		T = tf.gather(X, X_argmax, axis=1, batch_dims=1)
		with tf.GradientTape() as tape:
			Y = dec(X, training=True)
			L = tf.reduce_mean((T - Y)**2)
		grads = tape.gradient(L, dec.trainable_weights)
		adam.apply_gradients(zip(grads, dec.trainable_weights))
		print(L.numpy())
	print(Y)
	print(T)
	print(Y - T)
	dec.summary()
	dec.PMA.summary()
	dec.SAB.summary()

