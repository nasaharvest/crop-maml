from argparse import ArgumentParser

import sys

sys.path.append("..")

from src.maml.baselines import Pretrainer, train_model


def main():
    parser = ArgumentParser()

    args = Pretrainer.add_model_specific_args(parser).parse_args()
    model = Pretrainer(args)

    train_model(model, args)


if __name__ == "__main__":
    main()
