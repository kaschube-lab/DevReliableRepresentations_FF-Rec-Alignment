import numpy as np


# Nonlinearity functions (Numpy implementation)
nl_linear = lambda x: x
nl_tanh = lambda x: np.tanh(x)
nl_rect = lambda x: np.clip(x, 0, np.inf)
nl_shallow_rect = lambda x: np.clip(0.1*x, 0, np.inf)
nl_clip = lambda x: np.clip(x, 0, 1)
nl_softplus = lambda x: np.log(1. + np.exp(x)) #

def setup_sigmoid(max_S, slope, offset):
	return lambda x: max_S/(1+np.exp(-4*slope*(x-offset)))
