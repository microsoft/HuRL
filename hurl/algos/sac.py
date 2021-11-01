
from dowel import tabular
import numpy as np

from garage import obtain_evaluation_episodes, StepType
from hurl.algos._functions import log_performance, ExpAvg, MaxAvg
from hurl.lambda_schedulers import LambdaScheduler
from garage.torch.algos import SAC as garageSAC
from garage.torch import as_torch_dict

import copy

class MvAvg:  # for logging
    def __init__(self):
        self._x = 0
        self._itr = 0

    def update(self, vals, weight):
        self._x += vals
        self._itr += weight

    @property
    def mean(self):
        return self._x /self._itr if self._itr>0 else 0.


class SAC(garageSAC):

    def __init__(self, *args,
                 lambd=1.0,
                 heuristic=None,
                 reward_avg_rate=1e-3,
                 reward_shaping_mode='hurl',  # 'hurl' or 'pbrs'
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert not hasattr(self, 'lambd')
        assert not hasattr(self, '_heuristic')
        assert not hasattr(self, '_reward_avg')
        assert not hasattr(self, '_reward_shaping_mode')
        assert hasattr(self, '_discount')

        self._lambd = lambd if isinstance(lambd, LambdaScheduler) else LambdaScheduler(lambd)
        self._heuristic = heuristic or (lambda x : np.zeros(len(x)))
        self._discount0 = self._discount  # save the original discount
        self._discount = self.guidance_discount # we will update self._discount
        self._reward_avg = MaxAvg(rate=reward_avg_rate, scale_target=1)
        assert reward_shaping_mode in ['hurl', 'pbrs']
        self._reward_shaping_mode = reward_shaping_mode

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
            tabular.record('RewardBias',self._reward_avg.bias)
            tabular.record('RewardScale',self._reward_avg.scale)
        self._lambd.update()
        self._discount = self.guidance_discount

    def heuristic(self, next_obs, terminals):
        assert len(terminals.shape)==1 or (len(terminals.shape)==2 and terminals.shape[1]==1)
        # terminals can be (N,) or (N,1)
        hs = self._heuristic(next_obs)
        if len(hs.shape)<len(terminals.shape):
            assert len(hs.shape)==1
            hs = hs[...,np.newaxis]
        elif len(hs.shape)>len(terminals.shape):
            assert len(hs.shape)==2 and hs.shape[1]==1
            hs = hs[:,0]
        assert hs.shape == terminals.shape
        return hs*(1-terminals)

    def reshape_rewards(self, rewards, next_obs, terminals, obs=None):
        if self._reward_shaping_mode=='hurl':
            hs = self.heuristic(next_obs, terminals)
            assert rewards.shape == hs.shape == terminals.shape
            return rewards + (self._discount0-self._discount)*hs
        elif self._reward_shaping_mode=='pbrs':
            # This is a pbrs version of hurl. Setting lambda=1 recovers the classic pbrs.
            hs_next = self.heuristic(next_obs, terminals)
            hs_now = self._heuristic(obs)
            hs_now = hs_now[...,np.newaxis]
            assert rewards.shape == hs_next.shape == terminals.shape == hs_now.shape
            return rewards + self._discount0 * hs_next - hs_now

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                    # update
                    self._reward_avg.update(path['rewards'])  # for logging

                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))
                self._reward_ratio = MvAvg()  #XXX for logging

                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()

            if self._num_evaluation_episodes>0:
                last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            # XXX Extra logging
            tabular.record('RewardRatio', self._reward_ratio.mean)

            #HACK Update lambda
            self.update_guidance_discount()
            trainer.step_itr += 1

        return np.mean(last_return) if last_return is not None else 0

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)

            # HACK Reshape and normalize the rewards
            samples = copy.deepcopy(samples)
            shaped_rewards = self.reshape_rewards(rewards=samples['reward'],
                                                  next_obs=samples['next_observation'],
                                                  terminals=samples['terminal'],
                                                  obs=samples['observation'])
            # for logging
            self._reward_ratio.update(vals=np.sum(np.abs(shaped_rewards)/(1e-7+np.abs(samples['reward']))),
                                      weight=len(shaped_rewards))
            samples['reward'] = shaped_rewards
            samples = as_torch_dict(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss


    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)

        # HACK: Log the heuristic values
        terminals = eval_episodes.step_types== StepType.TERMINAL
        eval_episodes.env_infos['h'] =self.heuristic(eval_episodes.next_observations, terminals)


        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)
        return last_return
