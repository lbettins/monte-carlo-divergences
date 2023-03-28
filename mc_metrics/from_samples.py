import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def kde(samples, optimize=False, kde_kwargs={'kernel' : 'gaussian', 'bandwidth' : 'scott'}):
    """ KDE from array of samples """
    if optimize:
        print("Finding optimal Gaussian KDE bandwidth for samples using BayesianSearchCV.")
        params = {'bandwidth' : Real(0.01, 10, prior='log-uniform')}
        grid = BayesSearchCV(KernelDensity(kernel='gaussian'), params)
        grid.fit(np.array(samples).reshape(-1, 1))
        f = grid.best_estimator_
    else:
        print("Generating distribution from samples using '{kernel}' KDE with \
              bandwidth '{bandwidth}'.".format(kernel=kde_kwargs['kernel'],
                                               bandwidth=kde_kwargs['bandwidth']))
        f = KernelDensity(**kde_kwargs).fit(samples.reshape(-1, 1))
    return f
