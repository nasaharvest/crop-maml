from datetime import date, timedelta
import pandas as pd
from pathlib import Path
import geopandas
from shapely.geometry import Polygon, MultiPolygon

import ee
from .base import BaseSentinelExporter

from typing import Optional


class BrazilMunicipalitiesExporter(BaseSentinelExporter):
    r"""
    This is useful if you are trying to export
    full regions for predictions
    """
    raw_dataset = "brazil_municipalities"
    dataset = "earth_engine_region_brazil_municipalities"

    def __init__(self, data_folder: Path = Path("data"), identifier: Optional[str] = None) -> None:
        if identifier is not None:
            self.dataset = f"{self.dataset}_{identifier}"
        super().__init__(data_folder)

    def load_labels(self) -> pd.DataFrame:
        return geopandas.read_file(self.raw_folder / self.raw_dataset).to_crs(epsg=4326)

    def export_for_region(
        self,
        end_date: date,
        municipality_name: str = "LUÍS EDUARDO MAGALHÃES",
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        checkpoint: bool = True,
        monitor: bool = True,
        fast: bool = True,
    ):
        start_date = end_date - num_timesteps * timedelta(days=days_per_timestep)

        region_row = self.labels[self.labels["NM_MUNICIP"] == municipality_name]
        if len(region_row) == 0:
            raise ValueError(f"No municipality named {municipality_name}")
        else:
            # turn it into a GEE polygon
            geometry = region_row.iloc[0].geometry
            region_id = region_row.iloc[0].CD_GEOCMU

            if isinstance(geometry, Polygon):
                regions = [ee.Geometry.Polygon(list(geometry.exterior.coords))]
            elif isinstance(geometry, MultiPolygon):
                regions = [
                    ee.Geometry.Polygon(list(geom.exterior.coords)) for geom in geometry.geoms
                ]

        for idx, region in enumerate(regions):
            self._export_for_polygon(
                polygon=region,
                polygon_identifier=f"{idx}-{region_id}",
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
