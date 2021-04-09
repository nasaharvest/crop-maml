import geopandas
import pandas as pd
from pathlib import Path

from .base import BaseProcessor

from typing import List


class EthiopiaProcessor(BaseProcessor):

    dataset = "ethiopia"

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
            self.raw_folder / "ethiopia_crop.shp",
            self.raw_folder / "ethiopia_noncrop.shp",
            self.raw_folder / "ethiopia_crop_gt",
            self.raw_folder / "ethiopia_non_crop_gt",
        ]

        for filepath in shapefiles:
            output_dfs.append(self.process_single_shapefile(filepath))

        df = pd.concat(output_dfs)
        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
