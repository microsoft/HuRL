import copy
import numpy as np
import torch
from functools import partial
from garage.envs import GymEnv
from garage.torch import prefer_gpu
from garage.torch.optimizers import OptimizerWrapper

from hurl import rl_utils as ru
from hurl.algos.sac import SAC
from hurl.algos.bc import BC
from garage.torch.algos import VPG

__all__ = ['get_algo', 'log_performance', 'SAC', 'BC', 'VPG']

def get_algo(*,
             # required params
             algo_name,
             discount,
             env=None, # either env or episode_batch needs to be provided
             episode_batch=None,  # when provided, the algorithm is run in batch mode
             batch_size,  # batch size of the env sampler
             # heuristic guidance
             lambd=1.0,
             heuristic=None,
             reward_shaping_mode='hurl',
             # networks
             init_policy=None,  # learner policy
             policy_network_hidden_sizes=(256, 128),
             policy_network_hidden_nonlinearity=torch.tanh,
             value_natwork_hidden_sizes=(256, 128),
             value_network_hidden_nonlinearity=torch.tanh,
             # optimization
             policy_lr=1e-3,  # optimization stepsize for policy update
             value_lr=1e-3,  # optimization stepsize for value regression
             opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
             opt_n_grad_steps=1000,  # number of gradient updates per epoch
             num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
             steps_per_epoch=1,  # number of internal epochs steps per epoch
             n_epochs=None,  # number of training epochs
             randomize_episode_batch=True,
             reward_avg_rate=1e-3,
             reward_scale=1.0, # scaling factor on top the original reward
             # compute
             n_workers=4,  # number of workers for data collection
             use_gpu=False,  # try to use gpu, if implemented
             # algorithm specific hyperparmeters
             target_update_tau=5e-3, # for target network
             expert_policy=None,  # for BC
             gae_lambda=0.98,  # lambda of gae estimator
             **kwargs,
             ):
    # return alg for env with discount

    assert isinstance(env, GymEnv) or env is None
    assert not (env is None and episode_batch is None)
    assert batch_size is not None
    assert reward_shaping_mode in ['hurl', 'pbrs']

    # Parse algo_name
    value_ensemble_mode='P'
    value_ensemble_size=1
    if '_' in algo_name:
        algo_name, ensemble_mode = algo_name.split('_')
        value_ensemble_size= int(ensemble_mode[:-1]) # ensemble size of value network
        value_ensemble_mode= ensemble_mode[-1] # ensemble mode of value network, P or O

    # For normalized behaviors
    opt_n_grad_steps = int(opt_n_grad_steps/steps_per_epoch)
    n_epochs = n_epochs or np.inf
    num_timesteps = n_epochs * steps_per_epoch * batch_size


    # Define some helper functions used by most algorithms
    if episode_batch is None:
        get_sampler = partial(ru.get_sampler,
                              env=env,
                              n_workers=n_workers)
        env_spec = env.spec
    else:
        sampler = ru.BatchSampler(episode_batch=episode_batch, randomize=randomize_episode_batch)
        get_sampler = lambda p: sampler
        env_spec = episode_batch.env_spec
        num_evaluation_episodes=0

    if init_policy is None:
        get_mlp_policy = partial(ru.get_mlp_policy,
                                 env_spec=env_spec,
                                 hidden_sizes=policy_network_hidden_sizes,
                                 hidden_nonlinearity=policy_network_hidden_nonlinearity)
    else:
         get_mlp_policy = lambda *a, **kw : init_policy

    get_mlp_value = partial(ru.get_mlp_value,
                            env_spec=env_spec,
                            hidden_sizes=value_natwork_hidden_sizes,
                            hidden_nonlinearity=value_network_hidden_nonlinearity)

    get_replay_buferr = ru.get_replay_buferr
    max_optimization_epochs = max(1,int(opt_n_grad_steps*opt_minibatch_size/batch_size))
    get_wrapped_optimizer = partial(ru.get_optimizer,
                                    max_optimization_epochs=max_optimization_epochs,
                                    minibatch_size=opt_minibatch_size)

    # Create an algorithm instance
    if algo_name=='VPG':
        policy = get_mlp_policy(stochastic=True, clip_output=False)
        value_function = get_mlp_value('V',
                                       ensemble_mode=value_ensemble_mode,
                                       ensemble_size=value_ensemble_size)
        sampler = get_sampler(policy)
        algo = VPG(env_spec=env_spec,
                    policy=policy,
                    value_function=value_function,
                    sampler=sampler,
                    discount=discount,
                    center_adv=True,
                    positive_adv=False,
                    gae_lambda=gae_lambda,
                    policy_optimizer=OptimizerWrapper((torch.optim.Adam, dict(lr=policy_lr)),policy),
                    vf_optimizer=get_wrapped_optimizer(value_function, value_lr),
                    num_train_per_epoch=steps_per_epoch)


    elif algo_name=='SAC':
        # from garage.torch.algos import SAC
        policy = get_mlp_policy(stochastic=True, clip_output=True)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        replay_buffer = get_replay_buferr()
        sampler = get_sampler(policy)
        algo = SAC(env_spec=env_spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   sampler=sampler,
                   gradient_steps_per_itr=opt_n_grad_steps,
                   replay_buffer=replay_buffer,
                   min_buffer_size=int(0),
                   target_update_tau=target_update_tau,
                   discount=discount,
                   buffer_batch_size=opt_minibatch_size,
                   reward_scale=reward_scale,
                   steps_per_epoch=steps_per_epoch,
                   num_evaluation_episodes=num_evaluation_episodes,
                   policy_lr=policy_lr,
                   qf_lr=value_lr,
                   lambd=lambd,
                   heuristic=heuristic,
                   reward_avg_rate=reward_avg_rate,
                   reward_shaping_mode=reward_shaping_mode)

    elif algo_name=='BC':
        sampler=get_sampler(expert_policy)
        assert init_policy is not None
        if episode_batch is not None and expert_policy is None:
            expert_policy = copy.deepcopy(init_policy)  # this policy doesn't matter, since it runs in batch mode.
        assert expert_policy is not None
        algo = BC(env_spec,
                  init_policy,
                  source=expert_policy,
                  sampler=sampler,
                  batch_size=batch_size,
                  gradient_steps_per_itr=opt_n_grad_steps,
                  minibatch_size=opt_minibatch_size,
                  policy_lr=policy_lr,
                  loss='mse', #'log_prob' if isinstance(policy,StochasticPolicy) else 'mse'
                  )

    else:
        raise ValueError('Unknown algo_name')

    if use_gpu:
        prefer_gpu()
        if callable(getattr(algo, 'to', None)):
            algo.to()

    return algo
