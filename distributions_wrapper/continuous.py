import numpy as np
import scipy.stats
import tensorflow_probability as tfp
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

class ContinuousDistribution(tfp.distributions.Distribution):
    def __init__(self, f=None, samples=None):
        if f is None and samples is None:
            raise ValueError('Supply a distribution or samples from a distribution.')
        if samples is not None:
            print("Finding optimal KDE bandwidth for samples using BayesianSearchCV.")
            params = {'bandwidth' : Real(0.01, 100, prior='log-uniform')}
            grid = BayesSearchCV(KernelDensity(kernel='gaussian'), params1)
            grid.fit(np.array(samples))
            f = grid.best_estimator_
        if isinstance(f, KernelDensity):
            self.type = type(f)
            self.f = lambda X: np.exp(f.score_samples(X))
            self.logf = f.score_samples
            self.sample = f.sample
        elif isinstance(f, tfp.distributions.Distribution):
            self.type = type(f)
            self.f = lambda X: f.prob(X).numpy().flatten()
            self.logf = lambda X: f.log_prob(X).numpy().flatten()
            self.sample = lambda n: f.sample(sample_shape=(n, 1))
        elif isinstance(f, scipy.stats._distn_infrastructure.rv_continuous_frozen):
            self.type = type(f)
            self.f = lambda X: f.pdf(X).flatten()
            self.logf = lambda X: f.logpdf(X).flatten()
            self.sample = lambda n: f.rvs((n, 1))
        else:
            raise TypeError('Distribution not recognized!')

    def f(self, X):
        """Probability Density Function"""
        return self.f(X)

    def logf(self, X):
        """Log-probability Density Function"""
        return self.logf(X)

    def sample(self, n):
        """Sampling function for n random samples taken from f(X)"""
        return self.sample(n)

    def get_type(self):
        return self.type

    def entropy(self, mc_samples=100_000):
        print("Calculating Entropy by Monte Carlo Integration")
        x = self.sample(mc_samples)
        return -self.logf(x).mean()

    def kl_div(self, other, from_samples=False, mc_samples=100_000):
        if from_samples:
            other = ContinuousDistribution(samples=other)
        else:
            other = ContinuousDistribution(f=other)
        print("Calculating KL Divergence by Monte Carlo Integration")
        x = self.sample(mc_samples)
        return (self.logf(x) - other.logf(x)).mean()

    def js_div(self, other, from_samples=False, mc_samples=100_000):
        if from_samples:
            other = ContinuousDistribution(samples=other)
        else:
            other = ContinuousDistribution(f=other)
        print("Calculating JS Divergence by Monte Carlo Integration")
        # We need to draw samples from self and other distributions independently
        # for Monte Carlo Integration to work.
        x_self = self.sample(mc_samples)
        x_other = other.sample(mc_samples)

        # Compute log-mixture for self samples
        m_self = 0.5*(self.f(x_self) + other.f(x_self))
        m_self_log = np.log(m_self)
        m_other = 0.5*(self.f(x_other) + other.f(x_other))
        m_other_log = np.log(m_other)
        return 0.5*(self.logf(x_self) - m_self_log).mean() +\
                0.5*(other.logf(x_other) - m_other_log).mean()
