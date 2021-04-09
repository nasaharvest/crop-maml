from datetime import date, timedelta
import pandas as pd
from pathlib import Path

from .base import BaseSentinelExporter
from .utils import bounding_box_to_earth_engine_bounding_box
from src.utils import STR2BB

from typing import Optional


class RegionalExporter(BaseSentinelExporter):
    r"""
    This is useful if you are trying to export
    full regions for predictions
    """
    dataset = "earth_engine_region"

    def __init__(self, data_folder: Path = Path("data"), identifier: Optional[str] = None) -> None:
        if identifier is not None:
            self.dataset = f"{self.dataset}_{identifier}"
        super().__init__(data_folder)

    def load_labels(self) -> pd.DataFrame:
        # We don't need any labels for this exporter,
        # so we can return an empty dataframe
        return pd.DataFrame()

    def export_for_region(
        self,
        region_name: str,
        end_date: date,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        checkpoint: bool = True,
        monitor: bool = True,
        metres_per_polygon: Optional[int] = 10000,
        fast: bool = True,
    ):
        start_date = end_date - num_timesteps * timedelta(days=days_per_timestep)

        region = bounding_box_to_earth_engine_bounding_box(STR2BB[region_name])

        if metres_per_polygon is not None:

            regions = region.to_polygons(metres_per_patch=metres_per_polygon)

            for idx, region in enumerate(regions):
                self._export_for_polygon(
                    polygon=region,
                    polygon_identifier=f"{idx}-{region_name}",
                    start_date=start_date,
                    end_date=end_date,
                    days_per_timestep=days_per_timestep,
                    checkpoint=checkpoint,
                    monitor=monitor,
                    fast=fast,
                )
        else:
            self._export_for_polygon(
                polygon=region.to_ee_polygon(),
                polygon_identifier=region_name,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
