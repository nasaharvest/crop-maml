from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime
import numpy as np
import json


from src.processors import MaliProcessor
from src.exporters import MaliSentinelExporter
from .base import BaseEngineer, BaseDataInstance

from typing import Optional


@dataclass
class MaliDataInstance(BaseDataInstance):
    is_crop: int
    crop_int: int
    crop_label: str


class MaliEngineer(BaseEngineer):

    sentinel_dataset = MaliSentinelExporter.dataset
    dataset = MaliProcessor.dataset

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

        unique_classes = self.labels.label.unique()

        self.classes_to_index = {
            crop: idx for idx, crop in enumerate(unique_classes[unique_classes != np.array(None)])
        }

        json.dump(self.classes_to_index, (self.savedir / "classes_to_index.json").open("w"))

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        mali = data_folder / "processed" / MaliProcessor.dataset / "data.geojson"
        assert mali.exists(), "Sudan processor must be run to load labels"
        return geopandas.read_file(mali)

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
    ) -> Optional[MaliDataInstance]:

        da = self.load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        # we do year - 1, since the end of the timeseries is April, label_year + 1
        year = int(path_to_file.name.split("_")[0].split("-")[1]) - 1
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
                & (self.labels.year == year)
            )
        ]
        if len(overlap) == 0:
            return None
        else:
            label_lat = overlap.iloc[0].lat
            label_lon = overlap.iloc[0].lon

            label_type = overlap.iloc[0].label
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
                return MaliDataInstance(
                    label_lat=label_lat,
                    label_lon=label_lon,
                    instance_lat=closest_lat,
                    instance_lon=closest_lon,
                    labelled_array=labelled_array,
                    is_crop=int(overlap.iloc[0].is_crop),
                    crop_int=label_int,
                    crop_label=label_type,
                )
            else:
                return None
