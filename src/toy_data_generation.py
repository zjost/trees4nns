import numpy as np
from numpy.random import default_rng


class Distribution():
    supported = ['normal', 'gamma', 'lognormal']
    def __init__(self, family, parms):
        self.family = family # str
        self.name = self.family.lower()
        assert self.name in self.supported, f'{family} not supported'
        
        self.parms = parms # dict
        err_msg = 'Missing paramater {} for {} family!'
        if self.name=='normal':
            parm_names = ['loc', 'scale']
            for p in parm_names:
                assert p in self.parms, err_msg.format(p, self.family)
        elif self.name=='gamma':
            parm_names = ['shape', 'scale']
            for p in parm_names:
                assert p in self.parms, err_msg.format(p, self.family)
        elif self.name=='lognormal':
            parm_names = ['mean', 'sigma']
            for p in parm_names:
                assert p in self.parms, err_msg.format(p, self.family)
        
    def __str__(self):
        parms_str = ''
        for k, v in self.parms.items():
            parms_str += f'{k}={v:.3f}, '
        return f'{self.family}({parms_str[:-2]})'
    
    def __repr__(self):
        return self.__str__()
    
    def generate(self, rng, N):
        generator = getattr(rng, self.name)
        return generator(**self.parms, size=N)
        

class ToyDataGenerator():
    def __init__(self):
        self.rng = default_rng()
        self.distributions = list()
        
    def add_normal(self, loc, scale):
        """
        Commonly used for values coming from additive processes
        
        Args
        ------------
        loc: float
            Mean value
        scale: float
            Standard deviation
        """
        self.distributions.append(Distribution('Normal', dict(loc=loc, scale=scale)))
    
    def add_gamma(self, k, theta):
        """
        Commonly used for waiting times
        
        Args
        ------------
        k: float
            Shape parameter.  Skewness = 1/sqrt(k)
        theta: float
            Scale parameter.
        """
        self.distributions.append(Distribution('Gamma', dict(shape=k, scale=theta)))
        
    def add_lognormal(self, mu, sigma):
        """
        Commonly used for waiting times
        
        Args
        ------------
        mu: float
            Mean
        sigma: float
            Standard deviation
        """
        self.distributions.append(Distribution('LogNormal', dict(mean=mu, sigma=sigma)))
        
    def generate(self, N):
        vals = list()
        for dist in self.distributions:
            vals.append(dist.generate(self.rng, N).reshape(-1,1))
        return np.hstack(vals)
