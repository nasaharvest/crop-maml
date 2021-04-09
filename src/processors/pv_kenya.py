import geopandas
from pyproj import Transformer
from tqdm import tqdm

from typing import Set

from .base import BaseProcessor


class KenyaPVProcessor(BaseProcessor):
    dataset = "plant_village_kenya"

    @staticmethod
    def _remove_overlapping(
        df: geopandas.GeoDataFrame, overlapping_threshold: float
    ) -> geopandas.GeoDataFrame:

        rows_to_remove: Set[int] = set()

        org_len = len(df)
        df = df[df.geometry.is_valid]
        print(f"Removed {org_len - len(df)} invalid geometries")

        for idx_1, row_1 in tqdm(df.iterrows(), total=len(df)):
            for idx_2, row_2 in df.iloc[idx_1 + 1 :].iterrows():
                if row_1.geometry.intersects(row_2.geometry):
                    max_intersection_area = overlapping_threshold * min(
                        row_1.geometry.area, row_2.geometry.area
                    )
                    if row_1.geometry.intersection(row_2.geometry).area >= max_intersection_area:
                        rows_to_remove.add(idx_1)
                        rows_to_remove.add(idx_2)

        cleaned_df = df.drop(rows_to_remove)

        print(f"New df has len {len(cleaned_df)}, from {len(df)}")
        return cleaned_df

    @staticmethod
    def _fix_incorrect_year(df: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        r"""
        From discussions with the Plant Village team, rows where
        the planting date is in late 2019 and the harvest date is in
        early 2019 stem from a problem with the application used to
        input the data - the harvest year should be 2020
        """
        for row_idx in range(len(df)):
            if df.iloc[row_idx].harvest_da < df.iloc[row_idx].planting_d:
                harvest_date = df.iloc[row_idx].harvest_da
                # confirm the month is January, which would be
                # expected in a roll over situation
                if harvest_date.split("-")[1] == "01":
                    year = int(harvest_date.split("-")[0])
                    df.harvest_da.iloc[row_idx] = f"{year + 1}{harvest_date[4:]}"
        return df

    def process(self, only_2020: bool = True, overlapping_threshold: float = 0.1) -> None:

        df = geopandas.read_file(self.raw_folder / "field_boundaries_pv_04282020.shp")

        org_len = len(df)
        df = df[df.geometry.is_valid]
        print(f"Removed {org_len - len(df)} invalid geometries")

        org_len = len(df)
        df = df[df.harvest_da != "nan"]
        df = df[df.planting_d != "nan"]

        df = df[df.harvest_da != "NaT"]
        df = df[df.planting_d != "NaT"]
        print(f"Removed {org_len - len(df)} NaN harvest or planting dates")

        org_len = len(df)
        print("Fixing incorrect years")
        df = self._fix_incorrect_year(df)
        df = df[df.harvest_da > df.planting_d]
        print(f"Removed {org_len - len(df)} rows where the harvest happened before planting")

        if only_2020:
            org_len = len(df)
            df = df[df.harvest_da.str.contains("2020")]
            print(f"Removed {org_len - len(df)} rows where harvest didn't happen in 2020")

            # there should be no overlapping polygons, since these are all from
            # the same harvest
            df = self._remove_overlapping(df, overlapping_threshold)

        x = df.geometry.centroid.x.values
        y = df.geometry.centroid.y.values

        transformer = Transformer.from_crs(crs_from=32636, crs_to=4326)

        lat, lon = transformer.transform(xx=x, yy=y)
        df["lat"] = lat
        df["lon"] = lon

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
