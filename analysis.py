from pyvacy import analysis
import numpy as np
import matplotlib.pyplot as plt

X_len = 60000
params = {
    'delta': 1e-5,
    'device': 'cpu',
    'iterations': 1000,
    'l2_norm_clip': 4.,
    'l2_penalty': 0.001,
    'lr': 0.001,
    'microbatch_size': 1,
    'minibatch_size': 600,
    'noise_multiplier': 8
}

# for iteration in [1600, 12000, 80000]:

#     print(
#         analysis.epsilon(X_len,
#         params['minibatch_size'],
#         params['noise_multiplier'],
#         iteration,
#         params['delta']), f"iter{iteration}")

iterations = np.linspace(1, 10000+1, 100)
print(iterations.shape)
# delta_func = lambda it: analysis.delta(
#     X_len,
#     params['minibatch_size'],
#     params['noise_multiplier'],
#     it,
#     epsilon=0.5)
# eval_priv = np.array([delta_func(it) for it in iterations])

eps_func = lambda it: analysis.epsilon(
    X_len,
    params['minibatch_size'],
    params['noise_multiplier'],
    it,
    delta=1e-5)

# eval_priv = np.array([delta_func(it) for it in iterations])
eps = np.array([eps_func(it) for it in iterations])

# plt.title(f"eps={round(eps[0],3)}")
plt.plot(iterations/100, eps[:,0])
plt.hlines(0.5, 1, iterations[-1]/100)
plt.show()
# print(eval_priv)
