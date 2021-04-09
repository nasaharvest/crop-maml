import pandas as pd
import geopandas
from datetime import timedelta, date

from .base import BaseSentinelExporter
from src.processors.kenya_non_crop import KenyaNonCropProcessor

from typing import Optional


class KenyaNonCropSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_kenya_non_crop"

    # data collection date
    data_date = date(2020, 4, 16)

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        non_crop = self.data_folder / "processed" / KenyaNonCropProcessor.dataset / "data.geojson"
        assert non_crop.exists(), "Kenya non crop processor must be run to load labels"
        return geopandas.read_file(non_crop)[["lat", "lon"]]

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points, surrounding_metres=surrounding_metres,
        )

        start_date = self.data_date - num_timesteps * timedelta(days=days_per_timestep)

        for idx, bounding_info in enumerate(bounding_boxes_to_download):

            self._export_for_polygon(
                polygon=bounding_info.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=self.data_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
