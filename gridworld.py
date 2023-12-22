import numpy as np
from random import choice, choices, seed, random, randint
from itertools import product

"""
wall
landmark
empty
lava
wind
goal

Square routines
reward
positioning
"""

def left(position):
	x, y = position
	return (x - 1, y)


def right(position):
	x, y = position
	return (x + 1, y)


def up(position):
	x, y = position
	return (x, y + 1)


def down(position):
	x, y = position
	return (x, y - 1)



DIRECTIONS = {'left':  left, 
	      'up':    up, 
	      'right': right, 
	      'down':  down}



class Square:
	OPEN = True
	#def __init__(self, position, grid):
	#	self.position = position
	#	self.grid     = grid


	def __call__(self, reward=0.):
		"""
		This funciton is called whenever an agent enters this square.

		Arguments
		---------
		reward : (float)
			If the agent is moved through multiple squares in one step the reward is acumulated.

		Returns
		-------
		(float, (int, int), bool) : (reward, position, done)
			The returned values are the agents reward, its position and the episode termination flag.
		"""
		if not self.OPEN:
			raise Exception(f'{self.__class__.__name__} cannot be entered!')
		return self.routine(reward + self.REWARD)


	def copy(self):
		return self.__class__()


	def __eq__(self, other):
		return self.__class__ == other.__class__


	def routine(self, reward):
		"""
		Normal squares just modify the reward and do not move the agent.
		"""
		return reward, self.position, False


	def __bool__(self):
		"""
		Returns whether the square can be entered.
		"""
		return self.OPEN


	def __str__(self):
		return self.SYMBOL


	@property
	def T(self):
		return self.copy()


	@property
	def R(self):
		return self.copy()


	@property
	def H(self):
		return self.copy()


	@property
	def V(self):
		return self.copy()



class Landmark(Square):
	REWARD = 0
	SYMBOL = '🟨'



class Empty(Square):
	REWARD = 0
	SYMBOL = '⬜'



class Wall(Square):
	REWARD = 0
	OPEN   = False
	SYMBOL = '⬛'



class Lava(Square):
	REWARD = -1
	SYMBOL = '🟥'



class Goal(Square):
	REWARD = 1000
	SYMBOL = '❇️'
	def routine(self, reward):
		"""
		Normal squares just modify the reward and do not move the agent.
		"""
		return reward, self.position, True



class Wind(Square):
	REWARD = 0
	PROB = 0.7
	def direction(self, pos):
		return DIRECTIONS[self.DIRECTION](pos)


	def routine(self, reward):
		neighbor = self.grid[self.direction(self.position)]
		if neighbor and random() < self.PROB:
			return neighbor(reward)
		else:
			return reward, self.position, False


	@property
	def T(self):
		index = self.ORDER.index(self.__class__)
		# [1, 0, 3, 2]
		return  self.ORDER[(1 - index) % 4]()


	@property
	def R(self):
		index = self.ORDER.index(self.__class__)
		# [1, 2, 3, 0]
		return  self.ORDER[(index + 1) % 4]()


	@property
	def H(self):
		index = self.ORDER.index(self.__class__)
		# [2, 1, 0, 3]
		return  self.ORDER[(2 - index) % 4]()


	@property
	def V(self):
		index = self.ORDER.index(self.__class__)
		# [0, 3, 2, 1]
		return  self.ORDER[(0 - index) % 4]()



class NorthWind(Wind):
	DIRECTION = 'down'
	SYMBOL = '⬇️'



class EastWind(Wind):
	DIRECTION = 'left'
	SYMBOL = '⬅️'



class SouthWind(Wind):
	DIRECTION = 'up'
	SYMBOL = '⬆️'



class WestWind(Wind):
	DIRECTION = 'right'
	SYMBOL = '➡️'



Wind.ORDER = [WestWind, NorthWind, EastWind, SouthWind]



SQUARES = [Empty, Landmark, Wall, Lava, Goal, NorthWind, EastWind, SouthWind, WestWind]



class Grid:
	@classmethod
	def from_patches(cls, patches):
		x = patches[0][0].x_len
		y = patches[0][0].y_len
		i = len(patches[0])
		j = len(patches)
		grid = cls([[Empty() for _ in range((x * i))] for _ in range(y * j)])
		for j, line in enumerate(patches):
			for i, patch in enumerate(line):
				grid[i * x, j * y] = patch
		return grid


	def __init__(self, squares):
		self.squares = [[square.copy() for square in line] for line in squares]
		#print('SQUARES', squares)
		#print(self.squares)
		self.x_len = len(self.squares[0])
		self.y_len = len(self.squares)
		self.register()


	def copy(self):
		return Grid(squares=[[square.copy() for square in squares] for squares in self.squares])


	def register(self):
		for y, line in enumerate(self.squares):
			for x, square in enumerate(line):
				square.grid = self
				square.position = (x, y)


	def __getitem__(self, position):
		x, y = position
		if isinstance(x, int) and isinstance(y, int):
			# OUT OF RANGE CHECK
			if x >= self.x_len and y >= self.y_len:
				return None
			else:
				return self.squares[y][x]
		else:
			if not isinstance(x, slice):
				x = slice(x, x + 1)
			if not isinstance(y, slice):
				y = slice(y, y + 1)
			return Grid([line[x] for line in self.squares[y]])


	def roll_rows(self, n):
		return Grid([[self[(x + n) % self.x_len, y] for x in range(self.x_len)] for y in range(self.y_len)])


	def roll_cols(self, n):
		return Grid([[self[x, (y + n) % self.y_len] for x in range(self.x_len)] for y in range(self.y_len)])


	@property
	def T(self):
		return Grid([[self[x, y].T for y in range(self.y_len)] for x in range(self.x_len)]).copy()


	@property
	def R(self):
		return Grid([[self[x, y].R for y in range(self.y_len)] for x in range(self.x_len)][::-1]).copy()


	@property
	def H(self):
		return Grid([[self[x, y].H for x in range(self.x_len)][::-1] for y in range(self.y_len)]).copy()


	@property
	def V(self):
		return Grid([[self[x, y].V for x in range(self.x_len)] for y in range(self.y_len)][::-1]).copy()


	def __setitem__(self, key, value):
		"""
		[x, y] = Square
		[x, y] = Grid
		[xa:xb, ya:yb] = Square
		[xa:xb, ya:yb] = Grid
		"""
		x, y = key
		is_square = isinstance(value, Square)
		is_slice  = isinstance(x, slice) or isinstance(y, slice)
		if is_square:
			if is_slice:
				x_iter = range(x.start or 0, x.stop or self.x_len, x.step or 1) if isinstance(x, slice) else range(x, x + 1)
				y_iter = range(y.start or 0, y.stop or self.y_len, y.step or 1) if isinstance(y, slice) else range(y, y + 1)
				for x, y in product(x_iter, y_iter):
					self.squares[y][x] = value.copy()
			else:
				self.squares[y][x] = value.copy()
		else:
			x_other = range(value.x_len)
			y_other = range(value.y_len)
			if is_slice:
				x_self  = range(x.start or 0, x.stop or self.x_len, x.step or 1) if isinstance(x, slice) else range(x, x + 1)
				y_self  = range(y.start or 0, y.stop or self.y_len, y.step or 1) if isinstance(y, slice) else range(y, y + 1)
				for (x, i), (y, j) in product(zip(x_self, x_other), zip(y_self, y_other)):
					self.squares[y][x] = value[i, j].copy()
			else:
				for i, j in product(x_other, y_other):
					self.squares[y + j][x + i] = value[i, j].copy()
		self.register()


	def __eq__(self, other):
		if self.x_len != other.x_len or self.y_len != other.y_len:
			return False
		return all(self[x, y] == other[x, y] for x, y in product(range(self.x_len), range(self.y_len)))


	def __repr__(self):
		return '\n'.join(''.join(str(item) for item in line) for line in self.squares[::-1])



CHARCODE = {'L': Lava, 
	    'B': Wall, 
	    'M': Landmark, 
	    'V': Empty, 
	    'G': Goal, 
	    'E': EastWind, 
	    'N': NorthWind, 
	    'W': WestWind, 
	    'S': SouthWind}
patch = lambda p: Grid([[CHARCODE[char]() for char in line] for line in p.split('\n')])

with open('patches.txt', 'r') as file:
	text = file.read()

patches = [patch(p) for p in text.split('\n\n')]
rotations = [lambda p: p, 
	     lambda p: p.R, 
	     lambda p: p.R.R, 
	     lambda p: p.R.R.R]
symetries = [*rotations, *(lambda p, rot=rot: rot(p).H for rot in rotations)]
#rollrows  = list(lambda p, sym=sym, n=n: sym(p).roll_rows(n) for n in range(3) for sym in symetries)
transform = list(lambda p, row=row, n=n: row(p).H for n in range(3) for row in symetries)
PATCHES = list()
for p in patches:
	for f in transform:
		fp = f(p)
		if fp not in PATCHES:
			PATCHES.append(fp)
START = patch("""MVM
VVV
MVM""")
GOAL = patch("""MVM
VGV
MVM""")

WEIGHTS = [32,  2,  2,  2,  2,  8,  
	    8,  1,  1,  1,  1,  1,  
	    1,  1,  1,  1,  1,  1,  
	    1,  1,  1,  1,  1,  1,  
	    1,  8,  2,  2,  1,  1,  
	    1,  1, 32, 32,  1,  1, 
	   64,  8,  1,  1, 16, 32]

#for i, (p, w) in enumerate(zip(PATCHES, WEIGHTS)):
#	print(w)
#	print(p)
#print(*PATCHES, sep='\n\n')
#exit()


# 🔺🔻



class GridWorld:
	SHIFT_PROB = 0.1
	MASK_PROB  = 0.0
	NORTH_REWARD = 0.005
	DIST_REWARD = 1.0
	EXISTENCE_PENALTY = -0.01
	DIRECTIONS = [left, up, right, down, lambda x: x]
	def __init__(self, T=1000, X=3, Y=5):
		self.T    = T
		self.t    = 0
		self.grid = Grid.from_patches([choices(PATCHES, WEIGHTS, k=X) for y in range(Y)])
		self.grid = Grid([[Wall() for x in range(X * 3 + 4)] for y in range(Y * 3 + 4)])
		grid = Grid.from_patches([[choice(PATCHES) for x in range(X)] for y in range(Y)])
		self.start_patch = (int(X * 1.5 + 1), 2)
		self.goal_patch  = (int(X * 1.5 + 1), int((Y - 1) * 3 + 2))
		self.start_pos   = (int(X * 1.5 + 2), 3)
		self.goal_pos    = (int(X * 1.5 + 2), int((Y - 1) * 3 + 3))
		for x in range(X * 3):
			if random() < self.SHIFT_PROB:
				grid[x, :] = grid[x, :].roll_cols(randint(-2, 2))
		for y in range(Y * 3):
			if random() < self.SHIFT_PROB:
				grid[:, y] = grid[:, y].roll_rows(randint(-2, 2))
		self.grid[2, 2] = grid
		self.grid[self.start_patch] = START
		self.grid[self.goal_patch]  = GOAL
		self.grids = {'alpha': self.grid.copy(), 'beta': self.grid.copy()}
		for x, y in product(range(X * 3), range(Y * 3)):
			if random() < self.MASK_PROB:
				if not any(isinstance(self.grid[2 + x, 2 + y], s) for s in [Landmark, Goal]):
					grid = choice([self.grids['alpha'], self.grids['beta']])
					grid[2 + x, 2 + y] = Empty()
		self.state = {'alpha': self.start_pos, 'beta': self.start_pos}


	def encode(self, view):
		return np.array([[[1.0 if isinstance(view[i, j], s) else 0.0 \
				   for s in SQUARES]\
				   for j in range(5)]\
				   for i in range(5)])


	def _observation(self, agent):
		x, y = self.state[agent]
		grid = self.grids[agent]
		#print(x, y)
		#print(len(grid.squares), len(grid.squares[0]))
		view = grid[x-2:x+3,y-2:y+3]
		imag = self.encode(view)
		return imag


	@property
	def observations(self):
		alpha = self._observation('alpha')
		beta  = self._observation('beta')
		return alpha, beta


	def reset(self):
		self.state = {'alpha': self.start_pos, 'beta': self.start_pos}
		self.t = 0
		return self.observations


	def step(self, actions):
		a_alpha, a_beta = actions
		a_alpha, a_beta = int(a_alpha), int(a_beta)
		#print(a_alpha, a_beta)
		#exit()
		try:
			d_alpha, d_beta = self.DIRECTIONS[a_alpha], self.DIRECTIONS[a_beta]
		except:
			print(a_alpha, a_beta)
			raise NotImplemented
		p_alpha, p_beta = self.state['alpha'], self.state['beta']
		# SQUARES
		s_alpha = self.grid[d_alpha(p_alpha)]
		s_beta  = self.grid[d_beta(p_beta)]
		# DONES
		alpha_done = isinstance(self.grid[p_alpha], Goal)
		beta_done  = isinstance(self.grid[p_beta],  Goal)
		# REWARDS
		# existence
		ALPHA_REWARD = self.EXISTENCE_PENALTY
		BETA_REWARD  = self.EXISTENCE_PENALTY
		# north
		if a_alpha == 1:
			ALPHA_REWARD += self.NORTH_REWARD
		if a_beta == 1:
			BETA_REWARD  += self.NORTH_REWARD
		# STEP
		if s_alpha is not None and s_alpha.OPEN:
			r_alpha, p_alpha_next, d_alpha = s_alpha(ALPHA_REWARD)
		else:
			r_alpha, p_alpha_next, d_alpha = ALPHA_REWARD, p_alpha, False
		if s_beta is not None and s_beta.OPEN:
			r_beta, p_beta_next, d_beta = s_beta(BETA_REWARD)
		else:
			r_beta, p_beta_next, d_beta = BETA_REWARD, p_beta, False
		# distance
		alpha_dist = np.linalg.norm(np.array(p_alpha_next) - np.array(self.goal_pos))
		beta_dist  = np.linalg.norm(np.array(p_beta_next)  - np.array(self.goal_pos))
		r_alpha += min(0, 4 - alpha_dist) * self.DIST_REWARD
		r_beta  += min(0, 4 - beta_dist)  * self.DIST_REWARD
		#print(alpha_dist, beta_dist)
		#print(r_alpha, r_beta)
		#breakpoint()
		# FINALIZE
		self.state = {'alpha': p_alpha_next, 'beta': p_beta_next}
		if alpha_done:
			r_alpha = 0
			d_alpha = True
			self.state['alpha'] = p_alpha
		if beta_done:
			r_beta = 0
			d_beta = True
			self.state['beta'] = p_beta
		rewards = (float(r_alpha), float(r_beta))
		dones   = (d_alpha, d_beta)
		self.t += 1
		if self.t == self.T:
			dones = (True, True)
		return self.observations, rewards, dones


	def __getitem__(self, position):
		return self.grid[position]


	def __setitem__(self, position, square):
		self.grid[position] = square


	def __repr__(self):
		return str(self.grid)



class PartialGridWorld:
	SHIFT_PROB = 0.1
	MASK_PROB  = 0.0
	NORTH_REWARD = 0.005
	DIST_REWARD = 1.0
	EXISTENCE_PENALTY = -0.01
	DIRECTIONS = [left, up, right, down, lambda x: x]
	def __init__(self, T=1000, X=3, Y=5):
		self.T    = T
		self.t    = 0
		self.grid = Grid.from_patches([choices(PATCHES, WEIGHTS, k=X) for y in range(Y)])
		self.grid = Grid([[Wall() for x in range(X * 3 + 4)] for y in range(Y * 3 + 4)])
		grid = Grid.from_patches([[choice(PATCHES) for x in range(X)] for y in range(Y)])
		self.start_patch = (int(X * 1.5 + 1), 2)
		self.goal_patch  = (int(X * 1.5 + 1), int((Y - 1) * 3 + 2))
		self.start_pos   = (int(X * 1.5 + 2), 3)
		self.goal_pos    = (int(X * 1.5 + 2), int((Y - 1) * 3 + 3))
		for x in range(X * 3):
			if random() < self.SHIFT_PROB:
				grid[x, :] = grid[x, :].roll_cols(randint(-2, 2))
		for y in range(Y * 3):
			if random() < self.SHIFT_PROB:
				grid[:, y] = grid[:, y].roll_rows(randint(-2, 2))
		self.grid[2, 2] = grid
		self.grid[self.start_patch] = START
		self.grid[self.goal_patch]  = GOAL
		self.grids = {'alpha': self.grid.copy(), 'beta': self.grid.copy()}
		for x, y in product(range(X * 3), range(Y * 3)):
			if random() < self.MASK_PROB:
				if not any(isinstance(self.grid[2 + x, 2 + y], s) for s in [Landmark, Goal]):
					grid = choice([self.grids['alpha'], self.grids['beta']])
					grid[2 + x, 2 + y] = Empty()
		self.state = self.start_pos


	def encode(self, view):
		return np.array([[[1.0 if isinstance(view[i, j], s) else 0.0 \
				   for s in SQUARES]\
				   for j in range(5)]\
				   for i in range(5)])


	def _observation(self, agent):
		x, y = self.state
		grid = self.grids[agent]
		#print(x, y)
		#print(len(grid.squares), len(grid.squares[0]))
		view = grid[x-2:x+3,y-2:y+3]
		imag = self.encode(view)
		return imag


	@property
	def observations(self):
		alpha = self._observation('alpha')
		beta  = self._observation('beta')
		return alpha, beta


	def reset(self):
		self.state = self.start_pos
		self.t = 0
		return self.observations


	def step(self, action):
		action = int(action)
		#print(a_alpha, a_beta)
		#exit()
		try:
			d = self.DIRECTIONS[action]
		except:
			raise NotImplemented
		p = self.state
		# SQUARES
		s = self.grid[d(p)]
		# DONES
		done = isinstance(self.grid[p], Goal)
		# REWARDS
		REWARD = self.EXISTENCE_PENALTY
		# north
		if action == 1:
			REWARD += self.NORTH_REWARD
		# STEP
		if s is not None and s.OPEN:
			r, p_next, d = s(REWARD)
		else:
			r, p_next, d = REWARD, p, False
		# distance
		dist = np.linalg.norm(np.array(p_next) - np.array(self.goal_pos))
		r += min(0, 4 - dist) * self.DIST_REWARD
		#print(alpha_dist, beta_dist)
		#print(r_alpha, r_beta)
		#breakpoint()
		# FINALIZE
		self.state = p_next
		if done:
			r = 0
			d = True
			self.state = p_alpha
		rewards = float(r)
		dones   = d
		self.t += 1
		if self.t == self.T:
			dones = True
		return self.observations, rewards, dones


	def __getitem__(self, position):
		return self.grid[position]


	def __setitem__(self, position, square):
		self.grid[position] = square


	def __repr__(self):
		return str(self.grid)



class StackedSingleGridWorld(GridWorld):
	def __init__(self, N=4, **kwargs):
		self.N = N
		self.frame = []
		super().__init__(**kwargs)


	def stack(self, obs):
		N = min(len(self.frame), self.N)
		frame = self.frame[-N:] + [obs] * (self.N - N)
		return np.stack(frame)
		

	def reset(self):
		(obs, _) = super().reset()
		obs = self.stack(obs)
		self.frame = []
		return obs


	def step(self, actions):
		(obs, _), (rew, _), (dones, _) = super().step((actions, actions))
		obs = self.stack(obs)
		return obs, rew, dones



class StackedMultiGridWorld(GridWorld):
	def __init__(self, N=4, **kwargs):
		self.N = N
		self.frame_a = []
		self.frame_b = []
		self.dones   = False
		super().__init__(**kwargs)


	def stack(self, obs):
		obs_a, obs_b = obs
		N = min(len(self.frame_a), self.N)
		frame_a = self.frame_a[-N:] + [obs_a] * (self.N - N)
		frame_b = self.frame_b[-N:] + [obs_b] * (self.N - N)
		return np.stack(frame_a), np.stack(frame_b)
		

	def reset(self):
		(obs_a, obs_b) = super().reset()
		obs = self.stack((obs_a, obs_b))
		self.frame_a = []
		self.frame_b = []
		self.dones   = False
		return obs


	def step(self, actions):
		if self.dones:
			return self.obs, (0.0, 0.0), (True, True)
		(obs_a, obs_b), (rew_a, rew_b), (dones_a, dones_b) = super().step(actions)
		obs   = self.stack((obs_a, obs_b))
		rew   = (rew_a, rew_b)
		dones = (dones_a, dones_b)
		self.done = dones_a and dones_b
		self.obs  = obs
		return obs, rew, dones



class PartialStackedMultiGridWorld(PartialGridWorld):
	def __init__(self, N=4, **kwargs):
		self.N = N
		self.frame_a = []
		self.frame_b = []
		self.done    = False
		super().__init__(**kwargs)


	def stack(self, obs):
		obs_a, obs_b = obs
		N = min(len(self.frame_a), self.N)
		frame_a = self.frame_a[-N:] + [obs_a] * (self.N - N)
		frame_b = self.frame_b[-N:] + [obs_b] * (self.N - N)
		return np.stack(frame_a), np.stack(frame_b)
		

	def reset(self):
		(obs_a, obs_b) = super().reset()
		obs = self.stack((obs_a, obs_b))
		self.frame_a = []
		self.frame_b = []
		self.done    = False
		return obs


	def step(self, action):
		if self.done:
			return self.obs, (0.0, 0.0), (True, True)
		(obs_a, obs_b), rew, done = super().step(action)
		obs   = self.stack((obs_a, obs_b))
		self.done = done
		self.obs  = obs
		return obs, rew, done



class VectorStackedMultiEnv:
	def __init__(self, W=1, **kwargs):
		self.envs = [StackedMultiGridWorld(**kwargs) for w in range(W)]


	def step(self, actions):
		obs, rew, dones = zip(*(env.step(a) for env, a in zip(self.envs, zip(*actions))))
		obs   = np.array(list(zip(*obs)))
		rew   = np.array(list(zip(*rew)))
		dones = np.array(list(zip(*dones)))
		return obs, rew, dones


	def reset(self):
		obs_a, obs_b = zip(*(env.reset() for env in self.envs))
		obs_a, obs_b = np.array(obs_a), np.array(obs_b)
		return obs_a, obs_b



class VectorPartialStackedMultiEnv:
	def __init__(self, W=1, **kwargs):
		self.envs = [PartialStackedMultiGridWorld(**kwargs) for w in range(W)]


	def step(self, actions):
		obs, rew, dones = zip(*(env.step(a) for env, a in zip(self.envs, actions)))
		obs   = np.array(list(zip(*obs)))
		rew   = np.array(rew)
		dones = np.array(dones)
		return obs, rew, dones


	def reset(self):
		obs_a, obs_b = zip(*(env.reset() for env in self.envs))
		obs_a, obs_b = np.array(obs_a), np.array(obs_b)
		return obs_a, obs_b



if __name__ == '__main__':
	#seed(42)
	env = GridWorld(3, 5)
	#print(env.grids['beta'])
	print(env)
	#print(env[env.start_pos], env.start_pos)
	print(env[2:7, 2:7])
	exit()
	obs = env.reset()
	for i in range(16):
		print(obs[0])
		print()
		#print(obs[1])
		#alpha = int(input('alpha >>> '))
		#beta  = int(input('beta  >>> '))
		alpha, beta = 1, 1
		obs, reward, done = env.step((alpha, beta))
		print('reward:', reward)
		if any(done):
			print('DONE')
			print(obs[0])
			print(env[env.state['alpha']])
			#break

