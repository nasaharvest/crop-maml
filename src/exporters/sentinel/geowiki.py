import pandas as pd
import xarray as xr
from datetime import date

from .base import BaseSentinelExporter
from src.exporters import GeoWikiExporter

from typing import Optional


class GeoWikiSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_geowiki"

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        geowiki = self.data_folder / "processed" / GeoWikiExporter.dataset / "data.nc"
        assert geowiki.exists(), "GeoWiki processor must be run to load labels"
        return xr.open_dataset(geowiki).to_dataframe().dropna().reset_index()

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        start_date: date = date(2017, 3, 28),
        end_date: date = date(2018, 3, 28),
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:
        assert start_date >= self.min_date, f"Sentinel data does not exist before {self.min_date}"

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points, surrounding_metres=surrounding_metres,
        )

        for idx, bounding_box in enumerate(bounding_boxes_to_download):
            self._export_for_polygon(
                polygon=bounding_box.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=end_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
