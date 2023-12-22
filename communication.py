import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np



class Communication(tf.keras.Model):
	def __init__(self, vocab_size=25, embed_size=3, **kwargs):
		super().__init__(**kwargs)
		self.end_idx    = vocab_size
		self.start_idx  = vocab_size + 1
		self.pad_idx    = vocab_size + 2
		self.vocab_size = vocab_size + 2
		self.embedding  = tf.keras.layers.Embedding(vocab_size + 3, embed_size)
		self.dense_in   = tf.keras.layers.Dense(64, activation='silu')
		self.gru        = tf.keras.layers.GRU(64,
						      return_sequences=True,
						      return_state=True)
		self.dense_out  = tf.keras.layers.Dense(vocab_size + 1)


	def start_token(self, batch_size):
		return tf.constant([self.start_idx] * batch_size)[:,None,...]


	def end_token(self, batch_size):
		return tf.constant([self.end_idx] * batch_size)[:,None,...]


	def pad_token(self, batch_size):
		return tf.constant([self.pad_idx] * batch_size)[:,None,...]


	def call(self, tokens, states=None, return_state=False, initial=False, training=False):
		x         = self.embedding(tokens, training=training)
		if states is None:
			hidden = self.gru.get_initial_state(x)
		elif initial:
			hidden = self.dense_in(states)
		else:
			hidden = states
		print(hidden.shape, x.shape)
		x, states = self.gru(x, initial_state=hidden, training=training)
		x         = self.dense_out(x, training=training)
		x         = tf.nn.softmax(x, axis=-1)
		if return_state:
			return x, states
		else:
			return x


	def logits(self, tokens, states=None, initial=False, training=False):
		# log(exp(P_a)   /   (exp(P_a) + exp(P_b) + exp(P_c))
		# log(exp(P_a)) - log(exp(P_a) + exp(P_b) + exp(P_c))
		# P_a           - log(exp(P_a) + exp(P_b) + exp(P_c))
		inputs  = tokens[:,:-1]
		outputs = tokens[:,+1:]
		x       = self(inputs, states, initial=initial, training=training)
		logits  = tf.math.log(x)
		#print('TOKENS', tokens[0,:])
		#print('LOGITS', logits[0,:,:])
		#exit()
		#print()
		#print(logits.shape)
		#print(outputs.shape, inputs.shape)
		#print(logits)
		#print(outputs)

		logit   = tf.gather(logits, outputs, batch_dims=2)
		return tf.reduce_sum(logit, axis=1)



class ActorCom(tf.keras.Model):
	def __init__(self, vocab_size=25, embed_size=3, max_length=25, **kwargs):
		super().__init__(**kwargs)
		self.max_length = max_length
		self.com = Communication(vocab_size, embed_size)
		self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4)


	def call(self, embedding, return_logits=False, training=False):
		state  = embedding
		#state  = self.com.dense_in(embedding, training=training)
		batch  = embedding.shape[0]
		token  = self.com.start_token(batch)
		eos    = self.com.end_token(batch)
		pad    = self.com.pad_token(batch)
		tokens = [token]
		done   = tf.zeros((batch, 1), dtype=tf.int32)
		probs  = []
		post   = []
		masks  = []
		for i in range(self.max_length):
			prob, state = self.com(tokens=token, 
						 states=state, 
						 return_state=True, 
						 initial=i == 0, 
						 training=training)
			cat   = tfp.distributions.Categorical(probs=prob)
			token = cat.sample()
			post.append(prob)
			prob  = tf.gather(prob[:,0,:], token, batch_dims=1)
			# END OF SENTENCE
			term  = tf.cast(token == eos, dtype=tf.int32)
			# PARTIAL MASKS
			pad_mask   =      done
			token_mask = (1 - done) * (1 - term)
			eos_mask   = (1 - done) *      term
			# MASKING
			token = pad * pad_mask + token * token_mask + eos * eos_mask
			tokens.append(token)
			# MASKING PROBS
			prob_mask = tf.cast(done, dtype=tf.float32)
			prob = prob_mask + prob * (1 - prob_mask)
			# UPDATE MASK
			done  = done + (1 - done) * term
			#print(prob_mask[0,...])
			probs.append(prob)
			masks.append(done)
		final = done * pad + (1 - done) * eos
		tokens.append(final)
		masks = tf.stack(masks, axis=1)
		probs = tf.stack(probs, axis=1)
		posterior = tf.stack(post, axis=1)
		message = tf.concat(tokens, axis=1)
		message = tf.RaggedTensor.from_tensor(message, padding=self.com.pad_idx)
		if return_logits:
			logits = tf.math.log(probs)[:,:,0]
			#print('CALL', logits[0,...])
			#print(masks[0,...])
			#print(message[0,...])
			logits = tf.reduce_sum(logits, axis=1)
			#raise ValueError
			return message, logits, posterior, masks
		else:
			return message


	def posterior(self, tokens, embeddings, training=False):
		embeddings = self.com.dense_in(embeddings)
		log_posterior = self.com.logits(tokens[:,:1+self.max_length], embeddings, training=training)
		return tf.math.exp(log_posterior)


	def log_posterior(self, tokens, embeddings, training=False):
		embeddings = self.com.dense_in(embeddings)
		return self.com.logits(tokens[:,:1+self.max_length], embeddings, training=training)


	def batch_log_prior_approx(self, tokens, embeddings, training=False):
		logits = []
		batch_size = tokens.shape[0]
		for i in range(batch_size):
			token = tf.stack([tokens[i,...]] * (batch_size - 1), axis=0)
			embedding = tf.concat([embeddings[:i,...], embeddings[i+1:,...]], axis=0)
			logit = self.log_posterior(token, embedding, training=training)
			logits.append(tf.reduce_mean(logit))
			# P(M) = sum_o P(M, O) = sum_o P(M|O) P(O) = E_o[P(M|O)]
		#print()
		#print(self.log_posterior(tokens[i,...][None,...], embeddings[i-1,...][None,...]))
		#print(self.log_posterior(tokens[i,...][None,...], embeddings[i,...][None,...]))
		return tf.stack(logits, axis=0)



class CriticCom(tf.keras.Model):
	def __init__(self, vocab_size=25, embed_size=3, **kwargs):
		super().__init__(**kwargs)
		self.dense_0 = tf.keras.layers.Dense(64, activation='elu')
		self.norm_0  = tf.keras.layers.BatchNormalization()
		self.dense_1 = tf.keras.layers.Dense(1)
		self.mse     = tf.keras.losses.MeanSquaredError()
		self.adam    = tf.keras.optimizers.Adam(learning_rate=1e-3)



	def call(self, embedding, training=False):
		x = self.dense_0(embedding, training=training)
		x = self.norm_0(x,  training=training)
		x = self.dense_1(x, training=training)
		return x[...,0]



class Language(tf.keras.Model):
	def __init__(self, vocab_size=25, embed_size=3, **kwargs):
		super().__init__(**kwargs)
		self.com = Communication(vocab_size, embed_size)
		self.cross_entropy  = tf.keras.losses.SparseCategoricalCrossentropy()
		self.adam = tf.keras.optimizers.Adam(learning_rate=1e-3)



	def call(self, message, training=False):
		logits = self.com(message, training=training)
		return tf.nn.softmax(logits, axis=2)


	def prior(self, message, training=False):
		log_prior = self.com.logits(message, training=training)
		return tf.math.exp(log_prior)


	def log_prior(self, message, training=False):
		return self.com.logits(message, training=training)


	def loss(self, message):
		inputs  = message[:,:-1]
		outputs = message[:,1:]
		priors  = self(inputs)
		#print(priors[0,:])
		#print(outputs[0,:])
		#exit()
		return self.cross_entropy(outputs, priors)



if __name__ == '__main__':
	def gather(n, m):
		from gridworld import GridWorld
		from random import randint
		data = []
		for _ in range(n):
			x, y = 3, 5
			grid = GridWorld(x, y)
			i = randint(2, x * 3 + 1)
			j = randint(2, y * 3 + 1)
			grid.state['alpha'] = (i, j)
			#print(len(grid.grid.squares))
			#print(len(grid.grid.squares[0]))
			#exit()
			observations = []
			for _ in range(m):
				actions = randint(0, 4), randint(0, 4)
				obs, rewards, dones = grid.step(actions)
				observations.append(obs[0])
			data.append(observations)
		batch = tf.constant(data)
		return batch

	from stackedautoencoder import AutoEncoder

	embed_size = 4
	vocab_size = 8
	max_length = 4
	actor  = ActorCom(embed_size=embed_size, 
			  vocab_size=vocab_size, 
			  max_length=max_length)
	actor_target = ActorCom(embed_size=embed_size, 
			  vocab_size=vocab_size, 
			  max_length=max_length)
	critic = CriticCom(embed_size=embed_size, 
			   vocab_size=vocab_size)
	#lang   = Language(embed_size=embed_size, 
	#		  vocab_size=vocab_size)

	auto   = AutoEncoder()
	auto.load_weights('auto_base')
	#actor.load_weights('com_batch_approx')

	token_info = tf.math.log(float(vocab_size + 1))
	token_cost = .5 * token_info
	polyak     = 1e-1
	print(token_info)

	epochs = 300
	lang_losses = []
	actr_losses = []
	crit_losses = []
	mes_lengths = []
	mes_len_std = []
	targets     = []
	token_entropies = []
	

	for epoch in range(epochs):
		batch = gather(64, 4)
		#batch = tf.concat([batch] * 32, axis=0)
		#combatch = tf.concat([batch, ur_batch], axis=0)
		#emb   = auto.encoder(combatch)
		emb = auto.encoder(batch)
		print(f'### EPOCH {epoch} ###')
		with tf.GradientTape(persistent=True) as tape:
			mes, log_post, post, mask = actor(emb, return_logits=True, training=True)
			log_prior = actor.batch_log_prior_approx(mes, emb, training=True)
			
			mes_one = tf.one_hot(mes, vocab_size)[:,1:-1,:]
			token_count = tf.math.reduce_sum(mes_one, axis=(0, 1))
			token_probs = token_count / tf.math.reduce_sum(token_count)
			token_count_batch = tf.math.reduce_sum(mes_one, axis=1)
			token_probs_batch = token_count_batch / (1e-6 + tf.math.reduce_sum(token_count_batch, axis=-1)[:,None])
			token_entropies_batch = - tf.reduce_sum(token_probs_batch * tf.math.log(1e-6 + token_probs_batch), axis=-1)
			#ratio = tf.math.exp(log_post_target - log_post)
			mask = tf.cast(1 - mask[:,:,:,None], dtype=tf.float32)
			
			# TOKEN ENTROPY VARIATIONS
			token_entropy = - tf.reduce_sum(token_probs * tf.math.log(1e-6 + token_probs))
			#token_entropy = - tf.reduce_mean(tf.reduce_sum(mask * post * tf.math.log(post), axis=-1), axis=(1, 2))
			print('M          ', mes[0,1:-1].numpy())
			print('log(P(m|o))', log_post[0,...].numpy())
			#log_prior   = lang.log_prior(mes)
			print('log(P(m))  ', log_prior[0,...].numpy())
			# E_m,o[I(M, O)] = p(m, o) * [-log(P(m)) + log(P(m|o))]
			information = log_post - log_prior ###
			# FUNZT OHNE * token_cost
			cost        = (np.array(list(map(len, mes))) - 2) * token_cost
			# MES LEN STD SCALING
			mes_len = np.array(list(map(len, mes)), dtype=np.float32)
			#std = (1 + tf.math.reduce_std(mes_len)) / (tf.math.reduce_mean(mes_len) + 1)
			#std = tf.math.reduce_std(mes_len) / (tf.math.reduce_mean(mes_len) + 1)
			ml = tf.math.reduce_mean(mes_len)
			std = (1e-3 + tf.math.abs(ml - mes_len)) / (1e-3 + mes_len)
			#std = (1e-3 + tf.math.reduce_std(mes_len)) / (1e-3 + mes_len)
			
			#target = information - cost
			target = information 
			target = target * 1e0
			# YES
			#target      = (information - cost + 1) * std + token_entropies_batch
			# YES
			#target      = (information - cost + 1) *  token_entropy      * std
			# YES
			#target      = (information - cost + 1) * (token_entropy + 1) * std
			# YES
			#target      = (information - cost + 1) *  token_entropy      * (1 + std)
			#target      = (information - cost + 1) * (token_entropy + 1)
			#target      = (information - cost + 1) * (token_entropy + 1) * (1 + std)
			# YES
			#target      = (information - cost + 1) * (1 + std)
			# YES
			#target      = (information - cost + 1) * std
			
			#target      = token_entropy
			
			#lang_loss   = lang.loss(mes)
			
			#actr_loss   = -tf.reduce_mean(log_post * value)
			#actr_loss   = tf.reduce_mean(information)
			#print('YES', log_post)
			#actr_loss   = -tf.reduce_mean(log_post * tf.stop_gradient(target))
			#actr_loss   = -tf.reduce_mean(log_post_target * tf.stop_gradient(target * ratio))
			#actr_loss   = -tf.reduce_mean(log_post)
			#actr_loss   = -tf.reduce_mean(log_post * tf.stop_gradient(value))
			#actr_loss   = -tf.reduce_mean(log_post * tf.stop_gradient(target))
			actr_loss   = -tf.reduce_mean(target)
			
			#crit_loss   = critic.mse(target, value)
			#print(value)
			#print(information)
			#print('entropy', auto.entropy(emb).numpy())
			std = tf.reduce_mean(std)
			print('M_len       ',  sum(map(len, mes))/mes.shape[0])
			print('M_std       ',  std.numpy())
			#print(log_prior)
			print('H_t         ',  tf.reduce_mean(token_entropy).numpy())
			print('H_t(m)      ',  tf.reduce_mean(token_entropies_batch).numpy())
			print('H(m|o)      ', -tf.reduce_mean(log_post).numpy())
			print('H(m)        ', -tf.reduce_mean(log_prior).numpy())
			print('I(m, o)     ',  tf.reduce_mean(information).numpy())
			#print('V ~ I       ',  tf.reduce_mean(value).numpy())
			print('C           ',  tf.reduce_mean(cost).numpy())
			print('J           ',  tf.reduce_mean(target).numpy())
			print(mes[:,1:-1].numpy())
		gradients = tape.gradient(actr_loss, actor.trainable_weights)
		actor.adam.apply_gradients(zip(gradients, actor.trainable_weights))
		print('#######')
		print(actor.log_posterior(mes[0,...][None,...], emb[0,...][None,...]))
		print(actor.log_posterior(mes[0,...][None,...], emb[0+1,...][None,...]))
		print('#######')
		#print(epoch, lang_loss.numpy(), actr_loss.numpy(), crit_loss.numpy())
		mes_len = np.array(list(map(len, mes)), dtype=np.float32)
		#lang_losses.append(lang_loss.numpy())
		actr_losses.append(actr_loss.numpy())
		#crit_losses.append(crit_loss.numpy())
		targets.append(tf.reduce_mean(target).numpy())
		mes_lengths.append(tf.math.reduce_mean(mes_len))
		mes_len_std.append(tf.math.reduce_std(mes_len))
		token_entropies.append(tf.reduce_mean(token_entropy).numpy())
	from matplotlib import pyplot as plt
	#actor.save_weights('com_batch_approx')
	plt.ylabel('actor/critic loss')
	#plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), actr_losses)
	#plt.ylabel('actor loss')
	#plt.xlabel('steps')
	#plt.show()
	#plt.plot(range(epochs), crit_losses)
	#plt.ylabel('critic loss')
	plt.xlabel('steps')
	plt.show()
	
	plt.plot(range(epochs), targets)
	plt.ylabel('return')
	plt.xlabel('steps')
	plt.show()
	plt.plot(range(epochs), mes_lengths)
	plt.ylabel('message len/std')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), mes_len_std)
	plt.plot(range(epochs), token_entropies)
	plt.ylabel('token entropy')
	plt.ylabel('mes len/std token entropy')
	#plt.xlabel('steps')
	plt.show()

	"""
	from matplotlib import pyplot as plt
	plt.plot(range(epochs), lang_losses)
	plt.ylabel('language/actor/critic loss')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), actr_losses)
	#plt.ylabel('actor loss')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), crit_losses)
	#plt.ylabel('critic loss')
	plt.xlabel('steps')
	plt.show()
	plt.plot(range(epochs), targets)
	plt.ylabel('return')
	plt.xlabel('steps')
	plt.show()
	plt.plot(range(epochs), mes_lengths)
	plt.ylabel('message len/std')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), mes_len_std)
	plt.plot(range(epochs), token_entropies)
	plt.ylabel('token entropy')
	plt.ylabel('mes len/std token entropy')
	#plt.xlabel('steps')
	plt.show()

	exit()
	emb = np.zeros((2, 64))
	act = ActorCom()
	crt = CriticCom()
	lan = Language()
	mes = act(emb)
	val = crt(emb)
	pri = lan.prior(mes)
	#print(val)
	print(mes)
	print(pri)



	for epoch in range(epochs):
		batch = gather(64, 4)
		#batch = tf.concat([batch] * 32, axis=0)
		#combatch = tf.concat([batch, ur_batch], axis=0)
		#emb   = auto.encoder(combatch)
		emb = auto.encoder(batch)

		with tf.GradientTape(persistent=True) as tape:
			value = critic(emb)
			mes, log_post, post, mask = actor(emb, return_logits=True)

			log_prior = actor.batch_log_prior_approx(mes, emb, training=True)
			
			mes_one = tf.one_hot(mes, vocab_size)[:,1:-1,:]
			token_count = tf.math.reduce_sum(mes_one, axis=(0, 1))
			token_probs = token_count / tf.math.reduce_sum(token_count)
			token_count_batch = tf.math.reduce_sum(mes_one, axis=1)
			token_probs_batch = token_count_batch / (1e-6 + tf.math.reduce_sum(token_count_batch, axis=-1)[:,None])
			token_entropies_batch = - tf.reduce_sum(token_probs_batch * tf.math.log(1e-6 + token_probs_batch), axis=-1)


			log_post_target = actor_target.log_posterior(mes, emb)
			ratio = tf.math.exp(log_post_target - log_post)
			mask = tf.cast(1 - mask[:,:,:,None], dtype=tf.float32)
			
			# TOKEN ENTROPY VARIATIONS
			token_entropy = - tf.reduce_sum(token_probs * tf.math.log(1e-6 + token_probs))
			#token_entropy = - tf.reduce_mean(tf.reduce_sum(mask * post * tf.math.log(post), axis=-1), axis=(1, 2))
			print('M           ', mes[0,1:-1].numpy())
			print(mask.shape, post.shape)
			print('log(P(m|o))', log_post[0,...].numpy())
			log_prior   = lang.log_prior(mes)
			print('log(P(m))  ', log_prior[0,...].numpy())
			# E_m,o[I(M, O)] = p(m, o) * [-log(P(m)) + log(P(m|o))]
			information = log_post_target - log_prior
			information = log_post - log_prior ###
			# FUNZT OHNE * token_cost
			cost        = (np.array(list(map(len, mes))) - 2) * token_cost

			# MES LEN STD SCALING
			mes_len = np.array(list(map(len, mes)), dtype=np.float32)
			#std = (1 + tf.math.reduce_std(mes_len)) / (tf.math.reduce_mean(mes_len) + 1)
			#std = tf.math.reduce_std(mes_len) / (tf.math.reduce_mean(mes_len) + 1)
			ml = tf.math.reduce_mean(mes_len)
			std = (1e-3 + tf.math.abs(ml - mes_len)) / (1e-3 + mes_len)
			#std = (1e-3 + tf.math.reduce_std(mes_len)) / (1e-3 + mes_len)
			
			target      = (information - cost + 1) * std + token_entropies_batch
			# YES
			#target      = (information - cost + 1) *  token_entropy      * std
			# YES
			#target      = (information - cost + 1) * (token_entropy + 1) * std
			# YES
			#target      = (information - cost + 1) *  token_entropy      * (1 + std)
			#target      = (information - cost + 1) * (token_entropy + 1)
			#target      = (information - cost + 1) * (token_entropy + 1) * (1 + std)
			# YES
			#target      = (information - cost + 1) * (1 + std)
			# YES
			#target      = (information - cost + 1) * std
			
			#target      = token_entropy
			lang_loss   = lang.loss(mes)
			#actr_loss   = -tf.reduce_mean(log_post * value)
			#actr_loss   = tf.reduce_mean(information)
			#print('YES', log_post)
			#actr_loss   = -tf.reduce_mean(log_post * tf.stop_gradient(target))
			actr_loss   = -tf.reduce_mean(log_post_target * tf.stop_gradient(target * ratio))
			#actr_loss   = -tf.reduce_mean(log_post)
			#actr_loss   = -tf.reduce_mean(log_post * tf.stop_gradient(value))
			
			crit_loss   = critic.mse(target, value)
			#print(value)
			#print(information)
			#print('entropy', auto.entropy(emb).numpy())
			std = tf.reduce_mean(std)
			print('M_len       ',  sum(map(len, mes))/mes.shape[0])
			print('M_std       ',  std.numpy())
			#print(log_prior)
			print('H_t         ',  tf.reduce_mean(token_entropy).numpy())
			print('H_t(m)      ',  tf.reduce_mean(token_entropies_batch).numpy())
			print('H(m|o)      ', -tf.reduce_mean(log_post).numpy())
			print('H(m)        ', -tf.reduce_mean(log_prior).numpy())
			print('I(m, o)     ',  tf.reduce_mean(information).numpy())
			print('V ~ I       ',  tf.reduce_mean(value).numpy())
			print('C           ',  tf.reduce_mean(cost).numpy())
			print('J           ',  tf.reduce_mean(target).numpy())
		gradients = tape.gradient(lang_loss, lang.trainable_weights)
		lang.adam.apply_gradients(zip(gradients, lang.trainable_weights))
		gradients = tape.gradient(actr_loss, actor_target.trainable_weights)
		actor_target.adam.apply_gradients(zip(gradients, actor_target.trainable_weights))
		gradients = tape.gradient(crit_loss, critic.trainable_weights)
		critic.adam.apply_gradients(zip(gradients, critic.trainable_weights))

		#print(epoch, lang_loss.numpy(), actr_loss.numpy(), crit_loss.numpy())
		mes_len = np.array(list(map(len, mes)), dtype=np.float32)
		lang_losses.append(lang_loss.numpy())
		actr_losses.append(actr_loss.numpy())
		crit_losses.append(crit_loss.numpy())
		targets.append(tf.reduce_mean(target).numpy())
		mes_lengths.append(tf.math.reduce_mean(mes_len))
		mes_len_std.append(tf.math.reduce_std(mes_len))
		token_entropies.append(tf.reduce_mean(token_entropy).numpy())

		lang_epochs = 10
		for i in range(lang_epochs):
			batch = gather(64, 4)
			emb   = auto.encoder(batch)
			with tf.GradientTape() as tape:
				mes = actor(emb)
				lang_loss = lang.loss(mes)
			gradients = tape.gradient(lang_loss, lang.trainable_weights)
			lang.adam.apply_gradients(zip(gradients, lang.trainable_weights))
		
		weights = [actor.get_weights(), actor_target.get_weights()]
		actor.set_weights([(1 - polyak) * old + polyak * new for old, new in zip(*weights)])
			


	from matplotlib import pyplot as plt
	plt.plot(range(epochs), lang_losses)
	plt.ylabel('language/actor/critic loss')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), actr_losses)
	#plt.ylabel('actor loss')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), crit_losses)
	#plt.ylabel('critic loss')
	plt.xlabel('steps')
	plt.show()
	plt.plot(range(epochs), targets)
	plt.ylabel('return')
	plt.xlabel('steps')
	plt.show()
	plt.plot(range(epochs), mes_lengths)
	plt.ylabel('message len/std')
	plt.xlabel('steps')
	#plt.show()
	plt.plot(range(epochs), mes_len_std)
	plt.plot(range(epochs), token_entropies)
	plt.ylabel('token entropy')
	plt.ylabel('mes len/std token entropy')
	#plt.xlabel('steps')
	plt.show()

	exit()
	emb = np.zeros((2, 64))
	act = ActorCom()
	crt = CriticCom()
	lan = Language()
	mes = act(emb)
	val = crt(emb)
	pri = lan.prior(mes)
	#print(val)
	print(mes)
	print(pri)
	"""