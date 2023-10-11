import numpy as np
import matplotlib.pyplot as plt
from time import time

from errorcausation import GaussianModel

model = GaussianModel(n_components=2, covariance_type="spherical")
model.startprob_ = np.array([0.5, 0.5])

model.transmat_ = np.array([[0.995, 0.005],
                            [0.01, 0.99]])

model.means_ = np.array([[0.0, 0.0], [1., 0.]])
model.covars_ = np.array([0.4, 0.4]) ** 2

t0 = time()
N = 80
n = 8

X, Z = model.simulate_data(N, repeats = 1000, plot = False)
t1 = time()
print(f"Time to simulate data: {t1 - t0:.3f}s")

def boxcar(x, n=8):
    shape = x.shape
    new_shape = (shape[0], shape[1] // n, n)
    return np.mean(x.reshape(new_shape), axis=2)


f_options = {
    'mean': lambda x: np.mean(x, axis=1),
    'flatten': lambda x: x.flatten(),
    'boxcar': lambda x: boxcar(x, n),
}
f = f_options['boxcar']

I = f(X[..., 0])
Q = f(X[..., 1])

stacked_data = np.stack([I, Q], axis=-1)

t0 = time()
model_to_fit = GaussianModel(n_components=2, covariance_type="spherical")
model_to_fit.fit(stacked_data)


plt.hist(I.flatten(), bins=100, alpha=0.5, label='I')
plt.show()

t1 = time()
print(f"Time to initialize and fit model: {t1 - t0:.3f}s")

print(f"startprob:  \n{model_to_fit.startprob_}")

p_00 = model.transmat_[0, 0] ** n
p_11 = model.transmat_[1, 1] ** n
p_01 = 1 - p_00
p_10 = 1 - p_11

transmat = np.array([[p_00, p_01],
                    [p_10, p_11]])


print(f"transmat: \n{model_to_fit.transmat_}\n--\n{transmat}")

print(f"means: \n{model_to_fit.means_}")
print(f"std: \n{model_to_fit.covar_to_std()} -- {np.diag(model.covar_to_std()) / np.sqrt(n)}")


