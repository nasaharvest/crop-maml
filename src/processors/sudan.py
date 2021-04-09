import geopandas
import pandas as pd
from pathlib import Path

from .base import BaseProcessor

from typing import List


class SudanProcessor(BaseProcessor):

    dataset = "sudan"

    @staticmethod
    def process_single_shapefile(filepath: Path) -> geopandas.GeoDataFrame:
        df = geopandas.read_file(filepath)
        is_crop = 1
        if "non" in filepath.name.lower():
            is_crop = 0

        df["is_crop"] = is_crop

        df["lon"] = df.geometry.centroid.x
        df["lat"] = df.geometry.centroid.y

        df["org_file"] = filepath.name

        return df[["is_crop", "geometry", "lat", "lon", "org_file"]]

    def process(self) -> None:

        output_dfs: List[geopandas.GeoDataFrame] = []

        shapefiles = [
            self.raw_folder / "sudan_crop",
            self.raw_folder / "sudan_non_crop",
            # self.raw_folder / "sudan_crop_v2_ag"
        ]

        for filepath in shapefiles:
            output_dfs.append(self.process_single_shapefile(filepath))

        df = pd.concat(output_dfs)
        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
