import dill
from pathlib import Path
from argparse import ArgumentParser

# this is still necessary so that the dilled(?) object
# can find src
import sys

sys.path.append("..")


def get_checkpoint(data_folder: Path, version: int, checkpoint: str) -> Path:

    log_folder = data_folder / "maml_models" / f"version_{version}"

    if checkpoint == "best_val":
        checkpoint = list(log_folder.glob("checkpoint_iteration*.pkl"))
        return checkpoint[0]
    else:
        return log_folder / "final_model.pkl"


def test_model():
    parser = ArgumentParser()

    # figure out which model to use
    parser.add_argument("--version", type=int, default=0)
    # one of best_val or final
    parser.add_argument("--checkpoint", type=str, default="best_val")

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--many_n", dest="many_n", action="store_true")
    parser.set_defaults(many_n=False)
    parser.add_argument("--dataset", type=str, default="common_beans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_grad_steps", type=int, default=2000)
    parser.add_argument("--test_mode", type=str, default="maml")
    parser.add_argument("--num_cv", type=int, default=1)
    parser.add_argument("--neg_samples", dest="neg_samples", action="store_true")
    parser.set_defaults(neg_samples=False)
    parser.add_argument("--sample_from", type=int, default=-1)

    args = parser.parse_args()

    model_path = get_checkpoint(Path("../data"), args.version, args.checkpoint)

    print(f"Using model {model_path}")

    with model_path.open("rb") as f:
        model = dill.load(f)

    if args.neg_samples:
        neg_samples = [x / 10 for x in range(1, 11)]
    else:
        neg_samples = [1]

    if args.many_n:
        for n in [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, -1]:

            for neg_sample in neg_samples:

                prefix = f"neg_samples_{neg_sample}_num_samples_{n}"
                if args.seed is not None:
                    prefix = f"{prefix}_seed_{args.seed}"

                if args.sample_from != -1:
                    prefix = f"{prefix}_sample_from_{args.sample_from}"

                model.test(
                    num_samples=n,
                    train_k=args.k,
                    num_grad_steps=args.num_grad_steps,
                    negative_sample_ratio=neg_sample,
                    prefix=prefix,
                    save_state_dict=False,
                    test_dataset_name=args.dataset,
                    seed=args.seed,
                    test_mode=args.test_mode,
                    num_cross_val=args.num_cv,
                    sample_from=args.sample_from,
                )
    else:
        for neg_sample in neg_samples:
            prefix = f"neg_samples_{neg_sample}"
            if args.num_samples is not None:
                prefix = f"{prefix}_num_samples_{args.num_samples}"
            if args.seed is not None:
                if prefix is None:
                    prefix = f"seed_{args.seed}"
                else:
                    prefix = f"{prefix}_seed_{args.seed}"
            if args.sample_from != -1:
                prefix = f"{prefix}_sample_from_{args.sample_from}"
            model.test(
                prefix=prefix,
                num_grad_steps=args.num_grad_steps,
                test_dataset_name=args.dataset,
                num_samples=args.num_samples,
                negative_sample_ratio=neg_sample,
                train_k=args.k,
                seed=args.seed,
                test_mode=args.test_mode,
                num_cross_val=args.num_cv,
                sample_from=args.sample_from,
            )


if __name__ == "__main__":
    test_model()
