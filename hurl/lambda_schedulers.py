import numpy as np

class LambdaScheduler:
    def __init__(self, init_lambd, n_epochs=50):
        self._n_epochs = n_epochs
        self._itr = 0
        self._init_lambd = init_lambd
        self._lambd = init_lambd

    def update(self):
        self._itr +=1

    def __call__(self):
        return min(1.0, max(0.0, self._lambd))

class ConstLS(LambdaScheduler):
    pass

class ExpAvgLS(LambdaScheduler):
    def update(self):
        super().update()
        self._lambd = (1-self._rate)*self._lambd + self._rate*1.0

    @property
    def _rate(self):
        return 1./self._n_epochs

class LinearLS(LambdaScheduler):
    def update(self):
        super().update()
        self._lambd = self._init_lambd + (1.0-self._init_lambd)*self._delta

    @property
    def _delta(self):  # a value in [0,1]
        return min(1.0,self._itr/self._n_epochs)

class TanhLS(LinearLS):
    @property
    def _delta(self):  # a value in [0,1]
        limit = 0.99
        delta = np.tanh(self._itr/max(1,self._n_epochs-1)*np.arctanh(limit)) / limit
        return min(1.0, delta)