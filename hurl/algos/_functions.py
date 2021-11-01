
import numpy as np
from dowel import tabular

from garage.np import discount_cumsum
from garage import StepType

class ExpAvg:
    def __init__(self, rate=1e-3, scale_target=1):
        self._x = 0
        self._x2 = 0
        self._rate = rate
        self._itr = 0
        self._scale_target = scale_target

    def update(self, vals):
        t = self._rate
        self._x = (1-t)*self._x + t*np.sum(vals)
        self._x2 = (1-t)*self._x2 + t*np.sum(vals**2)
        self._itr = (1-t)*self._itr + t*len(vals)

    @property
    def bias(self):
        return self._x/self._itr if self._itr>0 else self._x

    @property
    def scale(self):
        return np.sqrt(self._x2/self._itr - self.bias**2) if self._itr>0 else 1

    def normalize(self, vals):
        return (vals-self.bias)/max(1e-6, self.scale)*self._scale_target



class MaxAvg:
    """ An online normalizer based on'non-shrinking' upper and lower bounds.
        It uses a NormalizerStd to give an instantaneous estimate of upper and
        lower bounds. It then compares that to the current bounds, and accept
        the new bounds only if that expands the current ones. The shifting bias
        is centered at the centered with respect to the upper and lower bounds.
    """


    def __init__(self, rate=1e-3, scale_target=1):
        # new attributes
        self._avg = ExpAvg(rate=rate)
        self._upper_bound = None
        self._lower_bound = None
        self._scale_target = scale_target

        self.bias = 0
        self.scale = 1

    def update(self, x):
        self._avg.update(x)
        upper_bound_candidate = self._avg.bias + self._avg.scale
        lower_bound_candidate = self._avg.bias - self._avg.scale
        if self._upper_bound is None:
            self._upper_bound = upper_bound_candidate
            self._lower_bound = lower_bound_candidate
        else:
            self._upper_bound = np.maximum(self._upper_bound, upper_bound_candidate)
            self._lower_bound = np.minimum(self._lower_bound, lower_bound_candidate)

        self.bias = 0.5*self._upper_bound + 0.5*self._lower_bound
        self.scale = self._upper_bound-self.bias

    def normalize(self, vals):
        return (vals-self.bias)/max(1e-6, self.scale)*self._scale_target


def log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])


    # HACK Account for the modification of the ShortMDP wrapper
    mean_hs = []
    max_hs = []
    min_hs = []
    std_hs = []
    for eps in batch.split():
        hs = eps.env_infos['h']
        mean_hs.append(np.mean(hs))
        max_hs.append(np.max(hs))
        min_hs.append(np.min(hs))
        std_hs.append(np.std(hs))
    with tabular.prefix('ShortRL' + '/'):
        tabular.record('MeanHeuristic', np.mean(mean_hs))
        tabular.record('MaxHeuristic', np.mean(max_hs))
        tabular.record('MinHeuristic', np.mean(min_hs))
        tabular.record('StdHeuristic', np.mean(std_hs))
        # tabular.record('Lambda', np.mean(mean_lambds))


    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))

        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns



import inspect
from hurl.lambda_schedulers import LambdaScheduler
def algo_wrapper(algo_cls):
    """ Add helper functions for using heuristics.

        It assumes the base algo cls uses `_discount` as the discount factor in
        learning. The user needs to call `update_guidance_discount` within
        train.

    """

    spec = inspect.getfullargspec(algo_cls.__init__)
    assert 'heuristic' not in spec.kwonlyargs
    assert 'heuristic' not in spec.args
    assert 'lambd' not in spec.kwonlyargs
    assert 'lambd' not in spec.args

    class new_algo_cls(algo_cls):

        def __init__(self, *args, lambd=1.0, heuristic=None, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self, 'lambd')
            assert not hasattr(self, '_heuristic')
            assert not hasattr(self, '_discount0')
            assert hasattr(self, '_discount')
            self._lambd = lambd if isinstance(lambd, LambdaScheduler) else LambdaScheduler(lambd)
            self._heuristic = heuristic or (lambda x : np.zeros(len(x)))
            self._discount0 = self._discount  # save the original discount
            self._discount = self.guidance_discount # we will update self._discount

        @property
        def guidance_discount(self):
            # a smaller discount
            return self._lambd()*self._discount0

        def update_guidance_discount(self):
            # HACK Update lambd
            with tabular.prefix('ShortRL' + '/'):
                tabular.record('Lambda',self._lambd())
                tabular.record('GuidanceDiscount',self._discount)
                tabular.record('Discount0',self._discount0)
            self._lambd.update()
            self._discount = self.guidance_discount

        def heuristic(self, next_obs, terminals):
            assert len(terminals.shape)==1 or (len(terminals.shape)==2 and terminals.shape[1]==1)
            # terminals is a boolean nd.array which can be (N,) or (N,1)
            hs = self._heuristic(next_obs)
            if len(hs.shape)<len(terminals.shape):
                assert len(hs.shape)==1
                hs = hs[...,np.newaxis]
            elif len(hs.shape)>len(terminals.shape):
                assert len(hs.shape)==2 and hs.shape[1]==1
                hs = hs[:,0]
            assert hs.shape == terminals.shape
            return hs*(1-terminals)

        def reshape_rewards(self, rewards, next_obs, terminals):
            hs = self.heuristic(next_obs, terminals)
            assert rewards.shape == hs.shape == terminals.shape
            return rewards + (self._discount0-self._discount)*hs

    return new_algo_cls