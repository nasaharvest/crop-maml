import geopandas
from datetime import timedelta, date
from tqdm import tqdm

from .base import BaseSentinelExporter
from src.processors import MaliProcessor

from .utils import EEBoundingBox, bounding_box_from_centre


from typing import Optional, List, Tuple


class MaliSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_mali"

    # data collection date, to be consistent
    # with the non crop data
    data_month = 4
    data_day = 16

    def load_labels(self) -> geopandas.GeoDataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        mali = self.data_folder / "processed" / MaliProcessor.dataset / "data.geojson"
        assert mali.exists(), "Mali processor must be run to load labels"
        return geopandas.read_file(mali)[["lat", "lon", "year"]]

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[Tuple[EEBoundingBox, date]]:

        output: List[Tuple[EEBoundingBox, date]] = []

        for _, row in tqdm(self.labels.iterrows()):

            output.append(
                (
                    bounding_box_from_centre(
                        mid_lat=row["lat"],
                        mid_lon=row["lon"],
                        surrounding_metres=surrounding_metres,
                    ),
                    # labels collected for 2018 are the 2018-2019 season
                    date(int(row["year"]) + 1, self.data_month, self.data_day),
                )
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

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

        for idx, (bounding_info, box_date) in enumerate(bounding_boxes_to_download):

            start_date = box_date - num_timesteps * timedelta(days=days_per_timestep)

            self._export_for_polygon(
                polygon=bounding_info.to_ee_polygon(),
                polygon_identifier=f"{idx}-{box_date.year}",
                start_date=start_date,
                end_date=box_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
                fast=fast,
            )
