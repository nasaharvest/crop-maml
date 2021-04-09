from argparse import ArgumentParser
from pathlib import Path

import sys

sys.path.append("..")

from src.maml.predict import predict


def prefix_from_name(model_name: str) -> str:
    return model_name[:-15]


def landcover_mapper():

    parser = ArgumentParser()

    # figure out which model to use
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--test_folder_name", type=str, default="earth_engine_region_busia")

    args = parser.parse_args()

    version_folder = Path(f"../data/maml_models/version_{args.version}")

    # hardcoded for now
    test_folder = Path(f"../data/raw/{args.test_folder_name}")
    test_files = test_folder.glob("*.tif")

    all_models = list(version_folder.glob(args.query))
    print(f"Using the following models: ")
    print(all_models)

    print(f"Using model {version_folder}")

    save_dirname = test_folder.name
    save_dir = version_folder / save_dirname
    save_dir.mkdir(exist_ok=True)

    for test_path in test_files:

        num_outfiles = 0

        output = save_dir / f"preds_{test_path.name}"
        if output.exists():
            print(f"{test_path.name} already run! skipping")
            continue

        print(f"Running for {test_path}")
        for model in all_models:
            if num_outfiles == 0:
                out = predict(version_folder, test_path, prefix=prefix_from_name(model.name))
            else:
                out["prediction_0"] += predict(
                    version_folder, test_path, prefix=prefix_from_name(model.name)
                )["prediction_0"]
            num_outfiles += 1

        out["prediction_0"] /= num_outfiles
        out.to_netcdf(save_dir / f"preds_{test_path.name}")


if __name__ == "__main__":
    landcover_mapper()
