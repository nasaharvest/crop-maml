from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import geopandas
from datetime import datetime
import json

from typing import Optional


from src.processors import KenyaNonCropProcessor
from src.exporters import KenyaNonCropSentinelExporter
from .base import BaseEngineer, BaseDataInstance
from .pv_kenya import PVKenyaEngineer


@dataclass
class KenyaNonCropDataInstance(BaseDataInstance):
    crop_int: Optional[int]
    is_crop: bool = False
    crop_label: str = "non_crop"


class KenyaNonCropEngineer(BaseEngineer):

    sentinel_dataset = KenyaNonCropSentinelExporter.dataset
    dataset = KenyaNonCropProcessor.dataset

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

        classes_to_index_path = (
            data_folder / "features" / PVKenyaEngineer.dataset / "classes_to_index.json"
        )
        if classes_to_index_path.exists():
            with classes_to_index_path.open("r") as f:
                c2i = json.load(f)

            max_idx = max(c2i.values())
            self.non_crop_index: Optional[int] = max_idx + 1
        else:
            self.non_crop_index = None
        print(
            f"Using crop_int value: {self.non_crop_index if self.non_crop_index is not None else 'None'}"
        )

    @staticmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        pv_kenya = data_folder / "processed" / KenyaNonCropProcessor.dataset / "data.geojson"
        assert pv_kenya.exists(), "Kenya Non Crop processor must be run to load labels"
        return geopandas.read_file(pv_kenya)

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
    ) -> Optional[KenyaNonCropDataInstance]:

        da = self.load_tif(path_to_file, days_per_timestep=days_per_timestep, start_date=start_date)

        # first, we find the label encompassed within the da

        min_lon, min_lat = float(da.x.min()), float(da.y.min())
        max_lon, max_lat = float(da.x.max()), float(da.y.max())
        overlap = self.labels[
            (
                (self.labels.lon <= max_lon)
                & (self.labels.lon >= min_lon)
                & (self.labels.lat <= max_lat)
                & (self.labels.lat >= min_lat)
            )
        ]
        if len(overlap) == 0:
            return None
        else:
            label_lat = overlap.iloc[0].lat
            label_lon = overlap.iloc[0].lon

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
                return KenyaNonCropDataInstance(
                    label_lat=label_lat,
                    label_lon=label_lon,
                    instance_lat=closest_lat,
                    instance_lon=closest_lon,
                    labelled_array=labelled_array,
                    crop_int=self.non_crop_index,
                )
            else:
                return None
