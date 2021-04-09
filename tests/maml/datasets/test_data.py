from src.maml.datasets.data import FromPathsTestLoader
import numpy as np
import pickle
from dataclasses import dataclass


@dataclass
class TestInstance:
    labelled_array: np.ndarray = np.array([1, 2])


class TestFromPathsTestLoader:
    @staticmethod
    def test_test_set_is_unique(tmpdir):

        pos_path_ids = [1, 2, 3, 4, 5, 6]
        neg_path_ids = [7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12]

        pos_paths, neg_paths = [], []

        test_instance = TestInstance()

        for id in pos_path_ids:
            pos_path = tmpdir / f"pos_{id}"
            pos_paths.append(pos_path)
            with pos_path.open("wb") as f:
                pickle.dump(test_instance, f)

        for id in neg_path_ids:
            neg_path = tmpdir / f"neg_{id}"
            neg_paths.append(neg_path)
            with neg_path.open("wb") as f:
                pickle.dump(test_instance, f)

        loader = FromPathsTestLoader(
            neg_paths,
            pos_paths,
            True,
            cache=False,
            device=None,
            normalizing_dict={"mean": np.array([1, 2]), "std": np.array([1, 2])},
            num_test=2,
        )

        train_paths = loader.subset_to_tasks["training"].pickle_files
        test_paths = loader.subset_to_tasks["testing"].pickle_files

        assert len(set(test_paths).intersection(set(train_paths))) == 0
        assert len(set(test_paths)) == len(test_paths) == 4

        loader.cross_val(seed=0, negative_sample_ratio=1, sample_from=-1)
        train_paths = loader.subset_to_tasks["training"].pickle_files
        test_paths = loader.subset_to_tasks["testing"].pickle_files
        assert len(set(test_paths).intersection(set(train_paths))) == 0
        assert len(set(test_paths)) == len(test_paths) == 4

        loader.cross_val(seed=2, negative_sample_ratio=1, sample_from=-1)
        train_paths = loader.subset_to_tasks["training"].pickle_files
        test_paths = loader.subset_to_tasks["testing"].pickle_files
        assert len(set(test_paths).intersection(set(train_paths))) == 0
        assert len(set(test_paths)) == len(test_paths) == 4
