from pathlib import Path
import sys

sys.path.append("..")

from src.maml.learner import Learner


def maml():

    learner = Learner(data_folder=Path("../data"), cache=False, k=10, task_weight=None)
    learner.train(
        num_iterations=2000, max_adaptation_steps=1, noise_factor=0, balancing_ratio=False
    )


if __name__ == "__main__":
    maml()
