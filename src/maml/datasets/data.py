from pathlib import Path
import numpy as np
import random

import torch

from src.exporters.sentinel.cloudfree import BANDS
from src.processors import TogoProcessor

from .data_funcs import default_to_int
from .tasksets import CropNonCropFromPaths, KenyaCropTypeFromPaths, SingleTask
from src.utils import set_seed

from typing import Any, Dict, Tuple, List, Union


class BaseDataLoader:

    device: torch.device
    subset_to_tasks: Dict[str, Any]

    bands_to_remove = ["B1", "B10"]

    @staticmethod
    def add_noise(x: torch.Tensor, noise_factor: float) -> torch.Tensor:
        if noise_factor == 0:
            return x
        # expect input to be of shape [timesteps, bands]
        # and to be normalized with mean 0, std=1
        # if its not, it means no norm_dict was passed, so lets
        # just assume std=1
        noise = torch.normal(0, 1, size=x.shape).float() * noise_factor

        # the added noise is the same per band, so that the temporal relationships
        # are preserved
        # noise_per_timesteps = noise.repeat(x.shape[0], 1)
        return x + noise

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """This nested function situation is so that
        _remove_bands can be called from an unitialized
        dataset, speeding things up at inference while still
        keeping the convenience of not having to check if remove
        bands is true all the time.
        """

        if self.remove_bands:
            return self._remove_bands(x)
        else:
            return x

    @classmethod
    def _remove_bands(cls, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        indices_to_remove: List[int] = []
        for band in cls.bands_to_remove:
            indices_to_remove.append(BANDS.index(band))

        bands_index = 1 if len(x.shape) == 2 else 2
        indices_to_keep = [i for i in range(x.shape[bands_index]) if i not in indices_to_remove]
        if len(x.shape) == 2:
            # timesteps, bands
            return x[:, indices_to_keep]
        else:
            # batches, timesteps, bands
            return x[:, :, indices_to_keep]


class FromPathsDataLoader(BaseDataLoader):
    def __init__(
        self,
        label_to_tasks: Dict[str, Union[CropNonCropFromPaths, KenyaCropTypeFromPaths]],
        normalizing_dict: Dict,
        device: torch.device,
    ) -> None:
        self.label_to_tasks = label_to_tasks
        self.normalizing_dict = normalizing_dict
        self.device = device

    @property
    def task_labels(self) -> List[str]:
        return self.label_to_tasks.keys()

    def task_k(self, label: str) -> int:
        return self.label_to_tasks[label][1].task_k

    def sample_task(
        self, task_label: str, k: int, noise_factor: float
    ) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:

        weight, task = self.label_to_tasks[task_label]

        x, y = task.sample(k, deterministic=False)
        x = self.remove_bands(self._normalize(x))

        return (
            weight,
            (
                self.add_noise(torch.from_numpy(x).float(), noise_factor).to(self.device),
                torch.from_numpy(y).float().to(self.device),
            ),
        )

    def sample(
        self, k: int, noise_factor: float
    ) -> Dict[int, Tuple[float, Tuple[torch.Tensor, torch.Tensor]]]:

        output_dict = {}
        for label in self.label_to_tasks.keys():

            output_dict[label] = self.sample_task(label, k, noise_factor)
        return output_dict


class TestBaseLoader(BaseDataLoader):
    def len_train(self) -> int:
        task = self.subset_to_tasks["training"]
        return len(task)

    def _sample(
        self, k: int, subset: str, deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task = self.subset_to_tasks[subset]
        x, y = task.sample(k, deterministic=deterministic)
        x = self.remove_bands(self._normalize(x))
        return (
            torch.from_numpy(x).float().to(self.device),
            torch.from_numpy(y).float().to(self.device),
        )

    def sample_train(
        self, num_samples: int, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # we want the same K samples to be returned every time.
        # in addition, if we have K1 > K2, then we want the samples
        # returned by test.sample(k2) to be a subset of test.sample(k1)

        # however, when num_samples = -1, then we want to sample from
        # all the training data, which is why we leave the option for
        # non-deterministic sampling. This should only be true when num_samples=-1, allowing
        # comparisons of different num_samples inputs when its not true.
        return self._sample(num_samples, "training", deterministic)

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        task = self.subset_to_tasks["testing"]

        # cache is True for testing
        x, y = task.x, task.y
        x = self.remove_bands(self._normalize(x))

        return (
            torch.from_numpy(x).float().to(self.device),
            torch.from_numpy(y).float().to(self.device),
        )

    def cross_val(self, seed: int, negative_sample_ratio: float, sample_from: int) -> None:
        raise NotImplementedError


class TogoMetaLoader(TestBaseLoader):

    test_dataset = TogoProcessor.dataset
    eval_dataset = TogoProcessor.evaluation_dataset

    def __init__(
        self,
        data_folder: Path,
        remove_b1_b10: bool,
        cache: bool,
        device: torch.device,
        normalizing_dict: Dict = None,
    ) -> None:
        self.data_folder = data_folder
        self.remove_b1_b10 = remove_b1_b10
        self.device = device

        assert normalizing_dict is not None, "Normalizing dict can't be computed from the test set"
        self.normalizing_dict = normalizing_dict

        self.subset_to_tasks = {
            subset: SingleTask(
                data_folder=data_folder,
                datasets=[self.test_dataset] if subset != "testing" else [self.eval_dataset],
                cache=True if subset == "testing" else cache,
                class_func=default_to_int,
            )
            for subset in ["training", "testing"]
        }

    def cross_val(self, seed: int, negative_sample_ratio: float, sample_from: int) -> None:
        print("No negative sampling for the Togo dataset")
        set_seed(seed)

        if sample_from == -1:
            # we shuffle the training indices, so that when sample_train is called
            # (which is deterministic) it will return different samples
            random.shuffle(self.subset_to_tasks["training"].positive_indices)
            random.shuffle(self.subset_to_tasks["training"].negative_indices)
        else:
            positive_indices = self.subset_to_tasks["training"].positive_indices
            negative_indices = self.subset_to_tasks["training"].negative_indices

            sample_from_pos = positive_indices[: sample_from // 2]
            sample_from_neg = negative_indices[: sample_from // 2]

            random.shuffle(sample_from_pos)
            random.shuffle(sample_from_neg)

            self.subset_to_tasks["training"].positive_indices = (
                sample_from_pos + positive_indices[sample_from // 2 :]
            )
            self.subset_to_tasks["training"].negative_indices = (
                sample_from_neg + negative_indices[sample_from // 2 :]
            )


class FromPathsTestLoader(TestBaseLoader):
    def __init__(
        self,
        negative_paths: List[Path],
        positive_paths: List[Path],
        remove_b1_b10: bool,
        cache: bool,
        device: torch.device,
        normalizing_dict: Dict = None,
        num_test: int = 10,
    ) -> None:
        self.remove_b1_b10 = remove_b1_b10
        self.device = device
        self.cache = cache

        assert normalizing_dict is not None, "Normalizing dict can't be computed from the test set"
        self.normalizing_dict = normalizing_dict

        self.positive_paths = positive_paths
        self.negative_paths = negative_paths

        self.num_test = num_test

        pos_train, pos_test = self.split_train_test(self.positive_paths, self.num_test)
        neg_train, neg_test = self.split_train_test(self.negative_paths, self.num_test)

        self.subset_to_tasks = {
            "testing": KenyaCropTypeFromPaths(
                negative_paths=neg_test, positive_paths=pos_test, cache=True,
            ),
            "training": KenyaCropTypeFromPaths(
                negative_paths=neg_train, positive_paths=pos_train, cache=self.cache,
            ),
        }

    @staticmethod
    def split_train_test(
        relevant_paths: List[Path], num_test: int
    ) -> Tuple[List[Path], List[Path]]:
        print(f"Using {num_test} samples per class for testing")
        # because we can duplicate paths, its important to make sure that
        # a) the test set only contains unique instances
        # b) none of those unique instances are also in the training set
        random.shuffle(relevant_paths)

        test_set = relevant_paths[:num_test]
        true_num_test = num_test
        while len(set(test_set)) != len(test_set):
            # first, remove duplicates
            test_set = list(set(test_set))
            # then, append more paths from the other set
            while len(test_set) < num_test:
                test_set.append(relevant_paths[true_num_test])
                true_num_test += 1

        assert len(test_set) == len(set(test_set)) == num_test

        # now, remove any elements in the test set from the
        # training set
        training_set = relevant_paths[true_num_test:]
        training_set = [path for path in training_set if path not in test_set]
        return training_set, test_set

    def cross_val(self, seed: int, negative_sample_ratio: float, sample_from: int) -> None:
        r"""
        When we have a very small test set, we use will
        train the model multiple times using
        different train / test distributions. This will
        allow us to understand the "true" performance of the
        model, reducing the noisiness of having a tiny test set
        """
        if sample_from != -1:
            raise RuntimeError("sample_from not implemented for test paths datasets")
        set_seed(seed)

        pos_train, pos_test = self.split_train_test(self.positive_paths, self.num_test)
        neg_train, neg_test = self.split_train_test(self.negative_paths, self.num_test)

        if negative_sample_ratio < 1:
            num_samples = int(negative_sample_ratio * len(neg_train))
            # its fine to do this, because the negative paths were shuffled
            # in split_train_test
            print(f"Using {num_samples} negative examples out of {len(neg_train)}")
            neg_train = neg_train[:num_samples]

        self.subset_to_tasks = {
            "testing": KenyaCropTypeFromPaths(
                negative_paths=neg_test, positive_paths=pos_test, cache=True,
            ),
            "training": KenyaCropTypeFromPaths(
                negative_paths=neg_train, positive_paths=pos_train, cache=self.cache,
            ),
        }
