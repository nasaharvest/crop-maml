from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import json
import random

from .data_funcs import default_to_int, to_crop_noncrop
from .utils import adjust_normalizing_dict, unique_sample
from src.utils.regions import BoundingBox, STR2BB
from src.utils.countries import countries, country_to_bbox, find_country

from src.exporters import GeoWikiExporter
from src.processors import (
    KenyaPVProcessor,
    KenyaNonCropProcessor,
    EthiopiaProcessor,
    SudanProcessor,
    MaliProcessor,
    LEMProcessor,
)

from src.engineer import (
    GeoWikiDataInstance,
    PVKenyaDataInstance,
    BrazilEngineer,
    BrazilDataInstance,
)

from typing import cast, Callable, Dict, List, Tuple, Optional


class BaseTask:

    positive_indices: List[int]
    negative_indices: List[int]

    sampled_positive_indices: List[int] = []
    sampled_negative_indices: List[int] = []

    def __init__(self, negative_paths: List[Path], positive_paths: List[Path], cache: bool) -> None:

        self.pickle_files = positive_paths + negative_paths

        self.positive_indices = list(range(len(positive_paths)))
        self.negative_indices = list(
            range(len(positive_paths), len(positive_paths) + len(negative_paths))
        )
        self.y_vals = [1] * len(positive_paths) + [0] * len(negative_paths)

        self.cache = cache

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        if cache:
            x_list: List[np.ndarray] = []
            y_list: List[int] = []

            for i in range(len(self)):
                x, y = self[i]
                x_list.append(x)
                y_list.append(y)
            self.x, self.y = np.stack(x_list), np.array(y_list)

    @classmethod
    def load_normalizing_dicts(
        cls, data_folder: Path
    ) -> List[Tuple[int, Optional[Dict[str, np.ndarray]]]]:
        # the weighting gets tricky here, so lets just do a weighting by overall
        # datset size

        length_nd = []
        for dataset in cls.datasets:
            length = len(list((data_folder / "features" / dataset).glob("*/*.pkl")))

            with (data_folder / "features" / dataset / "normalizing_dict.pkl").open("rb") as f:
                norm_dict = pickle.load(f)
            length_nd.append((length, norm_dict))
        return length_nd

    @property
    def task_k(self) -> int:
        return min(len(self.positive_indices), len(self.negative_indices))

    def __len__(self) -> int:
        return len(self.pickle_files)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        if (self.cache) & (self.x is not None):
            return (
                cast(np.ndarray, self.x)[index],
                cast(int, self.y[index]),
            )

        target_file = self.pickle_files[index]

        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        return target_datainstance.labelled_array, self.y_vals[index]

    def sample(self, k: int, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # we will sample to get half positive and half negative
        # examples
        output_x: List[np.ndarray] = []
        output_y: List[np.ndarray] = []

        if k == -1:
            k = self.task_k
        else:
            k = min(k, self.task_k)

        if deterministic:
            pos_indices = self.positive_indices[:k]
            neg_indices = self.negative_indices[:k]
        else:
            # first, check if we should clear the already sampled indices
            if (len(self.positive_indices) - len(self.sampled_positive_indices)) < k:
                self.sampled_positive_indices = []

            if (len(self.negative_indices) - len(self.sampled_negative_indices)) < k:
                self.sampled_negative_indices = []

            # only show datapoints which haven't been shown yet (in this task-epoch)
            pos_to_sample = [
                x for x in self.positive_indices if x not in self.sampled_positive_indices
            ]
            neg_to_sample = [
                x for x in self.negative_indices if x not in self.sampled_negative_indices
            ]

            try:
                pos_indices = unique_sample(self.pickle_files, pos_to_sample, k)
            except RuntimeError as e:
                print(e)
                pos_indices = random.sample(pos_to_sample, k)

            try:
                neg_indices = unique_sample(self.pickle_files, neg_to_sample, k)
            except RuntimeError as e:
                print(e)
                neg_indices = random.sample(neg_to_sample, k)

            self.sampled_positive_indices.extend(pos_indices)
            self.sampled_negative_indices.extend(neg_indices)

        # returns a list of [pos_index, neg_index, pos_index, neg_index, ...]
        indices = [val for pair in zip(pos_indices, neg_indices) for val in pair]
        for index in indices:
            x, y = self[index]
            output_x.append(x)
            output_y.append(y)

        return np.stack(output_x, axis=0), np.array(output_y)


class MaliCropTypeFromPaths(BaseTask):

    # non crop vs. crop is covered by the
    # CropNonCropFromPaths class. We include the dataset
    # so non crop can be in the negative_indices, but we won't
    # include it in the positive indices
    datasets = [
        GeoWikiExporter.dataset,
        MaliProcessor.dataset,
    ]

    @classmethod
    def make_train_val(
        cls,
        data_folder: Path,
        val_size: float,
        min_k: int,
        cache: bool,
        negative_crop_sampling: int,
        test_crop: Optional[str] = None,
    ):

        crop_to_path: Dict[str, Tuple[List[Path], List[Path]]] = {}

        classes_to_index_path = (
            data_folder / "features" / MaliProcessor.dataset / "classes_to_index.json"
        )
        crop_to_int = json.load(classes_to_index_path.open("r"))

        all_crops = list(crop_to_int.keys())

        for crop in all_crops:
            # the non crop v. rest will be in the countries
            # dataset
            if crop != "Non Crop":
                crop_to_path[f"mali_{crop}"] = ([], [])

        mali_bbox = country_to_bbox(find_country(countries, "Mali"))

        for dataset in cls.datasets:
            all_files = []
            for subset in ["training", "validation", "testing"]:
                # The test set is defined seperately, because
                # in very data-imbalanced classes the test set
                # might not contain enough of the class-of-interest
                # to be representative
                all_files.extend(list((data_folder / "features" / dataset / subset).glob("*.pkl")))
            for filepath in tqdm(all_files):
                with filepath.open("rb") as f:
                    datainstance = pickle.load(f)

                if isinstance(datainstance, GeoWikiDataInstance):
                    if (datainstance.crop_probability < 0.5) & (datainstance.isin(mali_bbox)):
                        crop_label = "Non Crop"
                    else:
                        # We don't know what crop type the GeoWiki
                        # crop tasks are, so we will ignore them.
                        continue
                else:
                    crop_label = datainstance.crop_label

                for crop_type in crop_to_path.keys():
                    if crop_type == f"mali_{crop_label}":
                        crop_to_path[crop_type][1].append(filepath)
                    else:
                        if crop_label != "Non Crop":
                            for _ in range(negative_crop_sampling):
                                crop_to_path[crop_type][0].append(filepath)
                        else:
                            crop_to_path[crop_type][0].append(filepath)

        train_dict, val_dict, test_paths = {}, {}, None

        for crop_label, (neg_paths, pos_paths) in crop_to_path.items():
            if test_crop is not None:
                if crop_label == f"mali_{test_crop}":
                    test_paths = (neg_paths, pos_paths)
                    continue

            task_k = min(len(pos_paths), len(neg_paths))
            if task_k >= min_k:
                randint = np.random.rand()

                taskset = MaliCropTypeFromPaths(
                    positive_paths=pos_paths, negative_paths=neg_paths, cache=cache
                )

                if randint < val_size:
                    val_dict[crop_label] = (1, taskset)
                else:
                    train_dict[crop_label] = (1, taskset)
        return train_dict, val_dict, test_paths


class BrazilCropTypeFromPaths(BaseTask):

    # we don't include GeoWiki since we have multiple
    # non-crop datasets, and its not clear which the
    # GeoWiki labels would fall under
    datasets = [LEMProcessor.dataset, GeoWikiExporter.dataset]

    @classmethod
    def make_train_val(
        cls,
        data_folder: Path,
        val_size: float,
        min_k: int,
        cache: bool,
        negative_crop_sampling: int,
        test_crop: Optional[str] = None,
    ):
        crop_to_path: Dict[str, Tuple[List[Path], List[Path]]] = {}

        classes_to_index_path = (
            data_folder / "features" / LEMProcessor.dataset / "classes_to_index.json"
        )
        crop_to_int = json.load(classes_to_index_path.open("r"))

        all_crops = list(crop_to_int.keys())

        for crop in all_crops:
            # the non crop v. rest will be in the countries
            # dataset
            crop_to_path[f"brazil_{crop}"] = ([], [])

        # The LEM+ points are all in West Bahia - this centers the
        # GeoWiki points around the relevant area
        bahia = STR2BB["West Bahia"]

        for dataset in cls.datasets:
            all_files = []
            for subset in ["training", "validation", "testing"]:
                # The test set is defined seperately, because
                # in very data-imbalanced classes the test set
                # might not contain enough of the class-of-interest
                # to be representative
                all_files.extend(list((data_folder / "features" / dataset / subset).glob("*.pkl")))
            for filepath in tqdm(all_files):
                with filepath.open("rb") as f:
                    datainstance = pickle.load(f)

                if isinstance(datainstance, BrazilDataInstance):
                    crop_label = datainstance.crop_label

                    for crop_type in crop_to_path.keys():
                        if crop_type == f"brazil_{crop_label}":
                            crop_to_path[crop_type][1].append(filepath)
                        else:
                            # if the crop label is a crop, and the crop_type we are adding it
                            # it to is also a crop, oversample. Otherwise, don't.
                            if (crop_label not in BrazilEngineer.non_crop_classes) and (
                                crop_type[7:] not in BrazilEngineer.non_crop_classes
                            ):
                                for _ in range(negative_crop_sampling):
                                    crop_to_path[crop_type][0].append(filepath)
                            else:
                                crop_to_path[crop_type][0].append(filepath)

                elif isinstance(datainstance, GeoWikiDataInstance):

                    if datainstance.isin(bahia):
                        is_crop = datainstance.crop_probability >= 0.5

                        for crop_type in crop_to_path.keys():
                            crop_label = crop_type[7:]

                            if crop_label in BrazilEngineer.non_crop_classes:
                                # if the crop label is a non crop class, add
                                # the crop GeoWiki labels
                                if is_crop == 1:
                                    crop_to_path[crop_type][0].append(filepath)
                            else:
                                # if the crop label is a crop class, add
                                # the non crop GeoWiki labels
                                if is_crop == 0:
                                    crop_to_path[crop_type][0].append(filepath)

        train_dict, val_dict, test_paths = {}, {}, None

        for crop_label, (neg_paths, pos_paths) in crop_to_path.items():
            if test_crop is not None:
                if crop_label == f"brazil_{test_crop}":
                    test_paths = (neg_paths, pos_paths)
                    continue

            task_k = min(len(pos_paths), len(neg_paths))
            if task_k >= min_k:
                randint = np.random.rand()

                taskset = BrazilCropTypeFromPaths(
                    positive_paths=pos_paths, negative_paths=neg_paths, cache=cache
                )

                if randint < val_size:
                    val_dict[crop_label] = (1, taskset)
                else:
                    train_dict[crop_label] = (1, taskset)
        return train_dict, val_dict, test_paths


class KenyaCropTypeFromPaths(BaseTask):

    # non crop vs. crop is covered by the
    # CropNonCropFromPaths class. We include the dataset
    # so non crop can be in the negative_indices, but we won't
    # include it in the positive indices
    datasets = [
        GeoWikiExporter.dataset,
        KenyaNonCropProcessor.dataset,
        KenyaPVProcessor.dataset,
    ]

    @classmethod
    def make_train_val(
        cls,
        data_folder: Path,
        val_size: float,
        min_k: int,
        cache: bool,
        negative_crop_sampling: int,
        test_crop: Optional[str] = None,
    ):

        crop_to_path: Dict[str, Tuple[List[Path], List[Path]]] = {}

        pv_classes_to_index_path = (
            data_folder / "features" / KenyaPVProcessor.dataset / "classes_to_index.json"
        )
        crop_to_int = json.load(pv_classes_to_index_path.open("r"))

        all_crops = list(crop_to_int.keys())
        all_crops = [crop for crop in all_crops if crop != "non_crop"]

        for crop in all_crops:
            crop_to_path[crop] = ([], [])

        datasets = cls.datasets

        kenya_bbox = country_to_bbox(find_country(countries, "Kenya"))

        for dataset in datasets:
            all_files = []
            for subset in ["training", "validation", "testing"]:
                # The test set is defined seperately, because
                # in very data-imbalanced classes the test set
                # might not contain enough of the class-of-interest
                # to be representative
                all_files.extend(list((data_folder / "features" / dataset / subset).glob("*.pkl")))
            for filepath in tqdm(all_files):
                with filepath.open("rb") as f:
                    datainstance = pickle.load(f)

                if isinstance(datainstance, GeoWikiDataInstance):
                    if (datainstance.crop_probability < 0.5) & datainstance.isin(kenya_bbox):
                        crop_label = "non_crop"
                    else:
                        # We don't know what crop type the GeoWiki
                        # crop tasks are, so we will ignore them.
                        continue
                elif isinstance(datainstance, PVKenyaDataInstance):
                    crop_label = datainstance.crop_label
                    if "intercrop" in crop_label:
                        # since we care about the maize map, we will explicitly
                        # ignore intercrop_maize, which can be confusing
                        continue
                else:
                    crop_label = "non_crop"

                for crop_type in crop_to_path.keys():
                    if crop_type == crop_label:
                        crop_to_path[crop_type][1].append(filepath)
                    else:
                        if crop_label != "non_crop":
                            for _ in range(negative_crop_sampling):
                                crop_to_path[crop_type][0].append(filepath)
                        else:
                            crop_to_path[crop_type][0].append(filepath)

        train_dict, val_dict, test_paths = {}, {}, None

        for crop_label, (neg_paths, pos_paths) in crop_to_path.items():
            if test_crop is not None:
                if crop_label == test_crop:
                    test_paths = (neg_paths, pos_paths)
                    continue

            task_k = min(len(pos_paths), len(neg_paths))
            if task_k >= min_k:
                randint = np.random.rand()

                taskset = KenyaCropTypeFromPaths(
                    positive_paths=pos_paths, negative_paths=neg_paths, cache=cache
                )

                # randint is [0,1) - if val_size is 0, we want to ensure
                # no tasks will go into the validation set. This is why
                # its this way instead of randint < val_size
                if randint > (1 - val_size):
                    val_dict[crop_label] = (1, taskset)
                else:
                    train_dict[crop_label] = (1, taskset)
        return train_dict, val_dict, test_paths


class CropNonCropFromPaths(BaseTask):

    datasets = [
        GeoWikiExporter.dataset,
        KenyaPVProcessor.dataset,
        KenyaNonCropProcessor.dataset,
        EthiopiaProcessor.dataset,
        SudanProcessor.dataset,
        MaliProcessor.dataset,
        LEMProcessor.dataset,
    ]

    @classmethod
    def make_train_val(
        cls,
        data_folder: Path,
        val_size: float,
        min_k: int,
        cache: bool,
        test_country: Optional[str] = None,
    ):

        # {country_name: positive_paths, negative_paths}
        country_to_path: Dict[str, Tuple[float, List[Path], List[Path]]] = {}

        datasets = cls.datasets
        for dataset in datasets:
            all_files = []
            for subset in ["training", "validation", "testing"]:
                # The test set is defined seperately, because
                # in very data-imbalanced classes the test set
                # might not contain enough of the class-of-interest
                # to be representative
                all_files.extend(list((data_folder / "features" / dataset / subset).glob("*.pkl")))
            for filepath in tqdm(all_files):
                with filepath.open("rb") as f:
                    datainstance = pickle.load(f)

                y = to_crop_noncrop(datainstance)

                for country in countries:

                    country_name = country.name
                    country_bbox = country_to_bbox(country)

                    if datainstance.isin(country_bbox):
                        if country_name not in country_to_path:

                            country_to_path[country_name] = ([], [], 1)

                        country_to_path[country_name][y].append(filepath)

        train_dict, val_dict = {}, {}

        for country, (neg_paths, pos_paths, country_weight) in country_to_path.items():
            if test_country is not None:
                if country == test_country:
                    continue

            if (len(pos_paths) >= min_k) and (len(neg_paths) >= min_k):
                randint = np.random.rand()

                taskset = CropNonCropFromPaths(
                    positive_paths=pos_paths, negative_paths=neg_paths, cache=cache
                )

                # randint is [0,1) - if val_size is 0, we want to ensure
                # no tasks will go into the validation set. This is why
                # its this way instead of randint < val_size
                if randint > (1 - val_size):
                    val_dict[country] = (country_weight, taskset)
                else:
                    train_dict[country] = (country_weight, taskset)
        return train_dict, val_dict


class SingleTask(BaseTask):
    def __init__(
        self,
        data_folder: Path,
        datasets: List[str],
        cache: bool,
        class_func: Optional[Callable] = None,
        geographic_filter: Optional[BoundingBox] = None,
    ) -> None:

        self.data_folder = data_folder
        self.cache = cache

        if class_func is None:
            # then, we will assume the binary case
            class_func = default_to_int
        self.class_func = class_func
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

        files_and_nds: List[Tuple] = []

        for dataset in datasets:
            files_and_nds.append(
                self.load_files_and_normalizing_dicts(self.data_folder / "features" / dataset)
            )

        pickle_files: List[Path] = []
        for files, _ in files_and_nds:
            pickle_files.extend(files)
        self.pickle_files = pickle_files

        self.normalizing_dict = adjust_normalizing_dict([(len(x[0]), x[1]) for x in files_and_nds])

        self.positive_indices: List[int] = []
        self.negative_indices: List[int] = []

        x_list: List[np.ndarray] = []
        y_list: List[int] = []
        for i in tqdm(range(len(self))):
            target_file = self.pickle_files[i]

            with target_file.open("rb") as f:
                target_datainstance = pickle.load(f)

            y = self.class_func(target_datainstance)

            if geographic_filter is not None:
                if not target_datainstance.isin(geographic_filter):
                    continue
            x = target_datainstance.labelled_array
            if y == 1:
                self.positive_indices.append(i)
            elif y == 0:
                self.negative_indices.append(i)
            else:
                pass
            if cache:
                # TODO figure out indexing without needing
                # to cache the whole dataset multiple times
                x_list.append(x)
                y_list.append(y)
        if cache:
            self.x, self.y = np.stack(x_list), np.array(y_list)
        print(
            f"Filtered to {len(self.negative_indices) + len(self.positive_indices)} "
            f"from {len(self)}"
        )

    @staticmethod
    def load_files_and_normalizing_dicts(
        features_dir: Path, file_suffix: str = "pkl"
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:
        # glob IS deterministic, but the files are unordered. Note that here we take all the files
        # in train / val / test since the splitting happens on a task level, not an instance level
        pickle_files = list(features_dir.glob(f"*/*{file_suffix}"))

        # try loading the normalizing dict. By default, if it exists we will use it
        if (features_dir / "normalizing_dict.pkl").exists():
            with (features_dir / "normalizing_dict.pkl").open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

    def __len__(self) -> int:
        return len(self.pickle_files)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:

        if (self.cache) & (self.x is not None):
            return (
                cast(np.ndarray, self.x)[index],
                cast(int, self.y)[index],
            )

        target_file = self.pickle_files[index]

        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        return target_datainstance.labelled_array, self.class_func(target_datainstance)
