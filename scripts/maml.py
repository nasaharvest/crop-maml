from pathlib import Path
import sys

sys.path.append("..")

from src.maml.learner import Learner


def maml():

    learner = Learner(data_folder=Path("../data"))
    learner.train()


if __name__ == "__main__":
    maml()
