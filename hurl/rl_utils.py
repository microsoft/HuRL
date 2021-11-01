# Some helper functions for using garage


import numpy as np
import torch

from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy, DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.sampler import FragmentWorker, LocalSampler, RaySampler
from garage.torch.optimizers import OptimizerWrapper


def get_mlp_policy(*,
                   env_spec,
                   stochastic=True,
                   clip_output=False,
                   hidden_sizes=(64, 64),
                   hidden_nonlinearity=torch.tanh,
                   min_std=np.exp(-20.),
                   max_std=np.exp(2.)):

    if stochastic and clip_output:
        return TanhGaussianMLPPolicy(
                    env_spec=env_spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=None,
                    min_std=min_std,
                    max_std=max_std)

    if stochastic and not clip_output:
        return GaussianMLPPolicy(env_spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=None)

    if not stochastic:
        return DeterministicMLPPolicy(
                    env_spec=env_spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=torch.tanh if use_tanh else None)



def get_mlp_value(form='Q',
                  *,
                  env_spec,
                  hidden_sizes=(256, 128),
                  hidden_nonlinearity=torch.tanh,
                  ensemble_size=1,
                  ensemble_mode='P'
                  ):

    if form=='Q':
        return ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None)
    if form=='V':
        return GaussianMLPValueFunction(
                env_spec=env_spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                learn_std=False)



def collect_episode_batch(policy, *,
                          env,
                          batch_size,
                          n_workers=4):
    """Obtain one batch of episodes."""
    sampler = get_sampler(policy, env=env, n_workers=n_workers)
    agent_update = policy.get_param_values()
    episodes = sampler.obtain_samples(0, batch_size, agent_update)
    return episodes

from garage.sampler import Sampler
import copy
from garage._dtypes import EpisodeBatch
class BatchSampler(Sampler):

    def __init__(self, episode_batch, randomize=True):
        self.episode_batch = episode_batch
        self.randomize = randomize
        self._counter = 0

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):

        ns = self.episode_batch.lengths
        if num_samples<np.sum(ns):
            if self.randomize:
                # Sample num_samples from episode_batch
                ns = self.episode_batch.lengths
                ind = np.random.permutation(len(ns))
                cumsum_permuted_ns = np.cumsum(ns[ind])
                itemindex = np.where(cumsum_permuted_ns>=num_samples)[0]
                if len(itemindex)>0:
                    ld = self.episode_batch.to_list()
                    j_max = min(len(ld), itemindex[0]+1)
                    ld = [ld[i] for i in ind[:j_max].tolist()]
                    sampled_eb = EpisodeBatch.from_list(self.episode_batch.env_spec,ld)
                else:
                    sampled_eb = None
            else:
                ns = self.episode_batch.lengths
                ind = np.arange(len(ns))
                cumsum_permuted_ns = np.cumsum(ns[ind])
                counter = int(self._counter)
                itemindex = np.where(cumsum_permuted_ns>=num_samples*(counter+1))[0]
                itemindex0 = np.where(cumsum_permuted_ns>num_samples*counter)[0]
                if len(itemindex)>0:
                    ld = self.episode_batch.to_list()
                    j_max = min(len(ld), itemindex[0]+1)
                    j_min = itemindex0[0]
                    ld = [ld[i] for i in ind[j_min:j_max].tolist()]
                    sampled_eb = EpisodeBatch.from_list(self.episode_batch.env_spec,ld)
                    self._counter+=1
                else:
                    sampled_eb = None
        else:
            sampled_eb = self.episode_batch

        return sampled_eb

    def shutdown_worker(self):
        pass


def get_sampler(policy,
                *,
                env,
                n_workers=4,
                **kwargs):  # other kwargs for the sampler

    if n_workers==1:
        return LocalSampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            worker_class=FragmentWorker,
                            **kwargs)
    else:
        return RaySampler(agents=policy,
                          envs=env,
                          max_episode_length=env.spec.max_episode_length,
                          n_workers=n_workers,
                          **kwargs)



from garage.replay_buffer import PathBuffer

def get_replay_buferr(capacity=int(1e6)):
    return PathBuffer(capacity_in_transitions=capacity)

def get_optimizer(obj, lr,
                  *,
                  max_optimization_epochs=1,
                  minibatch_size=128):

    return OptimizerWrapper((torch.optim.Adam, dict(lr=lr)),
                             obj,
                             max_optimization_epochs=max_optimization_epochs,
                             minibatch_size=minibatch_size)