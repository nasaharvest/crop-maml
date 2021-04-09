import geopandas
from datetime import timedelta, date

from .base import BaseSentinelExporter
from src.processors import LEMProcessor

from .utils import EEBoundingBox, bounding_box_from_centre


from typing import List, Tuple


class BrazilSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_lem_brazil"

    # data collection date, to be consistent
    # with the non crop data
    data_month = 4
    data_day = 16

    def load_labels(self) -> geopandas.GeoDataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        lem = self.data_folder / "processed" / LEMProcessor.dataset / "data.geojson"
        assert lem.exists(), "Mali processor must be run to load labels"
        return geopandas.read_file(lem)[["lat", "lon", "2019_crop", "2020_crop"]]

    @staticmethod
    def bboxes_per_year(
        df: geopandas.GeoDataFrame, box_date: date, surrounding_metres: int
    ) -> List[Tuple[EEBoundingBox, date]]:
        boxes: List[Tuple[EEBoundingBox, date]] = []
        for _, row in df.iterrows():
            boxes.append(
                (
                    bounding_box_from_centre(
                        mid_lat=row["lat"],
                        mid_lon=row["lon"],
                        surrounding_metres=surrounding_metres,
                    ),
                    box_date,
                )
            )
        return boxes

    def labels_to_bounding_boxes(self, surrounding_metres: int) -> List[Tuple[EEBoundingBox, date]]:

        crop_2019 = self.labels[~self.labels["2019_crop"].isna()]

        boxes: List[Tuple[EEBoundingBox, date]] = []
        boxes.extend(
            self.bboxes_per_year(
                crop_2019, date(2020, self.data_month, self.data_day), surrounding_metres,
            )
        )

        return boxes

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        fast: bool = True,
    ) -> None:
        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            surrounding_metres=surrounding_metres,
        )

        for idx, (bounding_info, box_date) in enumerate(bounding_boxes_to_download):

            start_date = box_date - num_timesteps * timedelta(days=days_per_timestep)

            self._export_for_polygon(
                polygon=bounding_info.to_ee_polygon(),
                # this is helpful to match it to the year in the processed file
                polygon_identifier=f"{idx}-{box_date.year - 1}",
                start_date=start_date,
                end_date=box_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
