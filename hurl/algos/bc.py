from garage.torch.algos import BC as garageBC
from dowel import tabular
import numpy as np
from garage.torch import as_torch

class BC(garageBC):
    def __init__(self, *args,
                 gradient_steps_per_itr=1000,
                 minibatch_size=128,
                 **kwargs):
        self._gradient_steps_per_itr = gradient_steps_per_itr
        self._minibatch_size = minibatch_size
        kwargs['minibatches_per_epoch']=None
        super().__init__(*args, **kwargs)
        self.policy = self.learner  # Fix a bug in garage

    # NOTE Evaluate performance after training
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, for services such as
                snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for epoch in trainer.step_epochs():
            losses = self._train_once(trainer, epoch)
            if self._eval_env is not None:
                log_performance(epoch,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env,
                                    num_eps=10),
                                    discount=1.0)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))

    # NOTE Use a fixed number of updates and minibatch size instead.
    def _train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        batch = self._obtain_samples(trainer, epoch)
        losses = []
        for _ in range(self._gradient_steps_per_itr):
            minibatch = np.random.randint(len(batch.observations), size=self._minibatch_size)
            observations = as_torch(batch.observations[minibatch])
            actions = as_torch(batch.actions[minibatch])
            self._optimizer.zero_grad()
            loss = self._compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        return losses