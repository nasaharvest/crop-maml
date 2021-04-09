from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.exporters import GeoWikiExporter
from src.processors import (
    KenyaPVProcessor,
    KenyaNonCropProcessor,
    EthiopiaProcessor,
    SudanProcessor,
    MaliProcessor,
    LEMProcessor,
)
from src.exporters.sentinel.cloudfree import BANDS

from ..datasets.data_funcs import to_crop_noncrop

from typing import cast, Tuple, Optional, List, Dict, Sequence


class PretrainingData(Dataset):

    bands_to_remove = ["B1", "B10"]

    def __init__(
        self,
        data_folder: Path,
        subset: str,
        remove_b1_b10: bool,
        cache: bool = True,
        include_sudan_ethiopia: bool = False,
        normalizing_dict: Optional[Dict] = None,
    ) -> None:

        self.normalizing_dict: Optional[Dict] = normalizing_dict
        self.data_folder = data_folder
        self.features_dir = data_folder / "features"
        self.cache = cache

        assert subset in ["training", "validation", "testing"]
        self.subset_name = subset

        self.remove_b1_b10 = remove_b1_b10

        self.pickle_files = []
        normalizing_dicts = []

        datasets = [
            GeoWikiExporter.dataset,
            KenyaPVProcessor.dataset,
            KenyaNonCropProcessor.dataset,
            MaliProcessor.dataset,
            LEMProcessor.dataset,
        ]

        if include_sudan_ethiopia:
            datasets.extend([EthiopiaProcessor.dataset, SudanProcessor.dataset])
        for dataset in datasets:
            ds_pickle_files, normalizing_dict = self.load_files_and_normalizing_dicts(
                self.features_dir / dataset, self.subset_name
            )

            ds_pickle_files = self.check_for_nones(ds_pickle_files)

            self.pickle_files.extend(ds_pickle_files)
            normalizing_dicts.append((len(ds_pickle_files), normalizing_dict))

        if self.normalizing_dict is None:
            self.normalizing_dict = self.adjust_normalizing_dict(normalizing_dicts)

        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        if self.cache:
            self.x, self.y = self.to_array()

    @staticmethod
    def check_for_nones(pickle_files: List[Path]) -> List[Path]:
        print("Checking labels")
        output_files = []
        for filepath in tqdm(pickle_files):
            with filepath.open("rb") as f:
                target_datainstance = pickle.load(f)
            if to_crop_noncrop(target_datainstance) is not None:
                output_files.append(filepath)
        return output_files

    @staticmethod
    def load_files_and_normalizing_dicts(
        features_dir: Path, subset_name: str, file_suffix: str = "pkl"
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:
        pickle_files = list((features_dir / subset_name).glob(f"*.{file_suffix}"))

        # try loading the normalizing dict. By default, if it exists we will use it
        if (features_dir / "normalizing_dict.pkl").exists():
            with (features_dir / "normalizing_dict.pkl").open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

    @staticmethod
    def adjust_normalizing_dict(
        dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    ) -> Optional[Dict[str, np.ndarray]]:

        for _, single_dict in dicts:
            if single_dict is None:
                return None

        dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

        new_total = sum([x[0] for x in dicts])

        new_mean = sum([single_dict["mean"] * length for length, single_dict in dicts]) / new_total

        new_variance = (
            sum(
                [
                    (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2) * length
                    for length, single_dict in dicts
                ]
            )
            / new_total
        )

        return {"mean": new_mean, "std": np.sqrt(new_variance)}

    def _normalize(self, array: np.ndarray, surrounding: bool) -> np.ndarray:
        if surrounding:
            norm_dict = self.surrounding_normalizing_dict
        else:
            norm_dict = self.normalizing_dict

        if norm_dict is None:
            return array
        else:
            return (array - norm_dict["mean"]) / norm_dict["std"]

    def __len__(self) -> int:
        return len(self.pickle_files)

    def to_array(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.x is not None:
            assert self.y is not None
            return self.x, self.y
        else:
            x_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            print("Loading data into memory")
            for i in tqdm(range(len(self))):
                x, y = self[i]
                x_list.append(x)
                y_list.append(y)

            return torch.stack(x_list), torch.stack(y_list)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache and (self.x is not None):
            return (
                cast(torch.Tensor, self.x)[index],
                cast(torch.Tensor, self.y)[index],
            )

        target_file = self.pickle_files[index]

        # first, we load up the target file
        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        label = to_crop_noncrop(target_datainstance)
        x = self.remove_bands(
            x=self._normalize(target_datainstance.labelled_array, surrounding=False)
        )

        return (
            torch.from_numpy(x).float(),
            torch.tensor(label).float(),
        )
