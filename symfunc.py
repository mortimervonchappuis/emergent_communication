import tensorflow as tf



symlog    = lambda x: tf.math.sign(x) * tf.math.log(tf.math.abs(x) + 1)
heavyside = lambda x: tf.cast(x > 0., dtype=x.dtype)
diff_step = lambda x: heavyside(x) + tf.nn.sigmoid(x) - tf.stop_gradient(tf.nn.sigmoid(x))


if __name__ == '__main__':
	x = tf.Variable(-10.)
	
	with tf.GradientTape() as tape:
		l = diff_step(x)
	print(tape.gradient(l, x))
	print(l.numpy())