"""utils
various utilities for classifier model training
"""
import random

import numpy as np
import torch


def set_seed(seed: int = 123):
    """set random seeds for libs

    :param seed: a random seed
    :type seed: int
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    pass


def seed_workers(worker_id):
    """set random seed for torch DataLoader workers"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    pass


class Log(object):
    """class that perform simple logging capabilities for ease of use and experiment
    reproduceability
    """

    def __init__(self, log_fpath: str = "./outputs/log.txt"):

        self.log_fpath = log_fpath

        pass

    def write(self, variable):

        with open(log_fpath) as f:
            f.write(q(variable) + "\n")

            f.close()

        pass


class EarlyStopping(object):
    """Custom callback used to monitor metrics for early stopping

    :param mode: Setting to determine if metric is better if low or high. Defaults
        to 'max'
    :type mode: str, optional
    :param min_delta: minimum change to permit
    :type min_delta: int, optional
    :param patience:number of epochs before stopping after stopping condition is
        met
    :type patience: int, optional
    """

    def __init__(self, mode="max", min_delta=0, patience=5):
        """Class constructor method"""

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.epoch = 0

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics: int, epoch: int) -> bool:
        """evaluate stopping condition
        :param metrics: metric to base conditon on
        :type metrics: int
        :param epoch: current epoch
        :type epoch: int
        :return: boolean indicator if stopping criterea are met or not
        :rytpe: bool
        """
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        """determine what makes a metric better"""
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta
        else:
            raise ValueError("mode " + mode + " is unknown!")


if __name__ == "__main__":

    pass
