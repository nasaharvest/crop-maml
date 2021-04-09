import geopandas
import pandas as pd
from pathlib import Path

from .base import BaseProcessor

from typing import List


class MaliProcessor(BaseProcessor):

    dataset = "mali"

    @staticmethod
    def process_non_crop(raw_folder: Path) -> geopandas.GeoDataFrame:

        df = geopandas.read_file(raw_folder / "mali_noncrop_2019")

        df["is_crop"] = 0
        df["lon"] = df.geometry.centroid.x
        df["lat"] = df.geometry.centroid.y

        df["label"] = "Non Crop"
        df["year"] = 2019

        return df[["is_crop", "geometry", "lat", "lon", "label", "year"]]

    @staticmethod
    def process_segou(raw_folder: Path) -> geopandas.GeoDataFrame:
        df = geopandas.read_file(raw_folder / "segou_bounds_07212020")

        df["is_crop"] = 1
        df["lon"] = df.geometry.centroid.x
        df["lat"] = df.geometry.centroid.y
        df["label"] = df["2018_main_"]
        df["year"] = 2018

        df_2019 = df.copy()
        df_2019["year"] = 2019
        df_2019["label"] = df["2019_main_"]

        combined_df = pd.concat([df, df_2019])

        return combined_df[["is_crop", "geometry", "lat", "lon", "label", "year"]]

    def process(self) -> None:

        output_dfs: List[geopandas.GeoDataFrame] = []

        output_dfs.append(self.process_non_crop(self.raw_folder))
        output_dfs.append(self.process_segou(self.raw_folder))

        df = pd.concat(output_dfs)
        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
