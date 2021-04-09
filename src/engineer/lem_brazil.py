from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime
import json


from src.processors import LEMProcessor
from src.exporters import BrazilSentinelExporter
from .base import BaseEngineer, BaseDataInstance

from typing import Optional


@dataclass
class BrazilDataInstance(BaseDataInstance):
    is_crop: int
    crop_int: int
    crop_label: str


class BrazilEngineer(BaseEngineer):

    sentinel_dataset = BrazilSentinelExporter.dataset
    dataset = LEMProcessor.dataset

    non_crop_classes = [
        "Cerrado",  # natural land in Brazil
        "Pasture",
        "Conversion area",  # Cerrado which has been cleared but is not yet cropland
        "Uncultivated soil",
    ]

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

        self.labels_2019 = self.labels[~self.labels["2019_crop"].isna()]
        self.labels_2019["label"] = self.labels_2019["2019_crop"]
        self.labels_2020 = self.labels[~self.labels["2020_crop"].isna()]
        self.labels_2020["label"] = self.labels_2020["2020_crop"]

        unique_classes = set(self.labels["2019_crop"].unique())
        unique_classes.update(set(self.labels["2020_crop"].unique()))

        # if we don't know what it is, we will ignore it
        unique_classes.remove("Not identified")
        unique_classes.remove(None)

        self.classes_to_index = {crop: idx for idx, crop in enumerate(unique_classes)}

        json.dump(self.classes_to_index, (self.savedir / "classes_to_index.json").open("w"))

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        lem = data_folder / "processed" / LEMProcessor.dataset / "data.geojson"
        assert lem.exists(), "LEM processor must be run to load labels"
        return geopandas.read_file(lem)

    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        add_ndwi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
    ) -> Optional[BrazilDataInstance]:

        da = self.load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        year = int(path_to_file.name.split("_")[0].split("-")[1])
        if year == 2019:
            labels = self.labels_2019
        elif year == 2020:
            labels = self.labels_2020
        else:
            print(f"Unrecognized year {year}")
            return [None, None]

        overlap = labels[
            (
                (labels.lon <= max_lon)
                & (labels.lon >= min_lon)
                & (labels.lat <= max_lat)
                & (labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            return None
        else:
            label_lat = overlap.iloc[0].lat
            label_lon = overlap.iloc[0].lon

            label_type = overlap.iloc[0].label
            if label_type == "Not identified":
                print("Unidentified crop. Skipping")
                return (None, None)

            is_crop = 1
            if label_type in self.non_crop_classes:
                is_crop = 0

            label_int = self.classes_to_index[label_type]

            closest_lon = self.find_nearest(da.x, label_lon)
            closest_lat = self.find_nearest(da.y, label_lat)

            labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

            if add_ndvi:
                labelled_np = self.calculate_ndvi(labelled_np)
            if add_ndwi:
                labelled_np = self.calculate_ndwi(labelled_np)

            labelled_array = self.maxed_nan_to_num(
                labelled_np, nan=nan_fill, max_ratio=max_nan_ratio
            )

            if (not is_test) and calculate_normalizing_dict:
                self.update_normalizing_values(self.normalizing_dict_interim, labelled_array)

            if labelled_array is not None:
                return BrazilDataInstance(
                    label_lat=label_lat,
                    label_lon=label_lon,
                    instance_lat=closest_lat,
                    instance_lon=closest_lon,
                    labelled_array=labelled_array,
                    is_crop=is_crop,
                    crop_int=label_int,
                    crop_label=label_type,
                )
            else:
                return None
