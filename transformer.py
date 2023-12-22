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
		#self.RFF = tf.keras.layers.Dense(key_dim, activation=activation)
		self.MAB = MAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, training=False):
		batch_size = X.shape[0]
		S = tf.repeat(self.S, [batch_size], axis=0)
		#Z = self.RFF(X, training=training)
		Z = self.MAB(S, X, training=training)
		return Z



class PMAS(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, activation='silu', **kwargs):
		super().__init__(**kwargs)
		self.emb_dim = emb_dim
		#self.RFF = tf.keras.layers.Dense(key_dim, activation=activation)
		self.MAB = MAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, S, training=False):
		batch_size = X.shape[0]
		#S = tf.repeat(S, [self.emb_dim], axis=1)
		S = tf.stack([S] * self.emb_dim, axis=1)
		#Z = self.RFF(X, training=training)
		Z = self.MAB(S, X, training=training)
		return Z



class Encoder(tf.keras.Model):
	def __init__(self, num_heads, key_dim, **kwargs):
		super().__init__(**kwargs)
		self.SAB1 = SAB(num_heads=num_heads, key_dim=key_dim)
		self.SAB2 = SAB(num_heads=num_heads, key_dim=key_dim)
		self.SAB3 = SAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, training=False):
		Z = self.SAB1(X, training=training)
		Z = self.SAB2(Z, training=training)
		Z = self.SAB3(Z, training=training)
		return Z



class SetDecoder(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, **kwargs):
		super().__init__(**kwargs)
		self.RFF = tf.keras.layers.Dense(key_dim, activation='silu')
		self.PMA = PMA(num_heads=num_heads, key_dim=key_dim, emb_dim=emb_dim)
		self.SAB = SAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, training=False):
		Z = self.PMA(X, training=training)
		Z = self.SAB(Z, training=training)
		Y = self.RFF(Z, training=training)
		return Y



class Decoder(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, **kwargs):
		super().__init__(**kwargs)
		self.RFF  = tf.keras.layers.Dense(key_dim, activation='silu')
		self.PMAS = PMAS(num_heads=num_heads, key_dim=key_dim, emb_dim=emb_dim)
		self.SAB  = SAB(num_heads=num_heads, key_dim=key_dim)


	def call(self, X, S, training=False):
		Z = self.PMAS(X, S, training=training)
		Z = self.SAB(Z,     training=training)
		Y = self.RFF(Z,     training=training)
		return Y



class SetTransformer(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=1, **kwargs):
		super().__init__(**kwargs)
		self.encoder = Encoder(num_heads, key_dim)
		self.decoder = SetDecoder(num_heads, key_dim, emb_dim)


	def call(self, X, training=False):
		Z = self.encoder(X, training=training)
		Y = self.decoder(Z, training=training)
		return Y



class Transformer(tf.keras.Model):
	def __init__(self, num_heads, key_dim, emb_dim=4, **kwargs):
		super().__init__(**kwargs)
		self.encoder = Encoder(num_heads, key_dim)
		self.decoder = Decoder(num_heads, key_dim, emb_dim)


	def call(self, X, S, training=False):
		Z = self.encoder(X, training=training)
		Y = self.decoder(Z, S, training=training)
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
	
	f = SetTransformer(num_heads=num_heads, key_dim=key_dim, emb_dim=emb_dim)
	
	Y1 = f(X1)
	Y2 = f(X2)
	print(tf.math.round(Y1 - Y2, 6))

	epochs = 1000
	adam = tf.keras.optimizers.Adam(learning_rate=3e-5)
	for e in range(epochs):
		X = tf.random.normal(shape=(batch_size, set_size, key_dim))
		X_mean   = tf.math.reduce_mean(X, axis=2)
		X_argmax = tf.math.argmax(X_mean, axis=1)
		X_argmin = tf.math.argmin(X_mean, axis=1)
		X_max = tf.gather(X, X_argmax, axis=1, batch_dims=1)
		X_min = tf.gather(X, X_argmin, axis=1, batch_dims=1)
		T = X_max * X_min
		print(T.shape)
		T = T * tf.concat([tf.ones((batch_size, key_dim//2)), tf.zeros((batch_size, key_dim-key_dim//2))], axis=1)
		with tf.GradientTape() as tape:
			Y = f(X, training=True)
			L = tf.reduce_mean((T - Y)**2)
		grads = tape.gradient(L, f.trainable_weights)
		adam.apply_gradients(zip(grads, f.trainable_weights))
		print(L.numpy())
	print(Y)
	print(T)
	print(Y - T)
	f.summary()
	f.encoder.summary()
	f.encoder.SAB1.summary()
	f.encoder.SAB2.summary()
	f.encoder.SAB3.summary()
	f.decoder.summary()
	f.decoder.PMA.summary()
	f.decoder.SAB.summary()

