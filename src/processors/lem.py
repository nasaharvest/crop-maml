import geopandas
import pandas as pd

from .base import BaseProcessor


class LEMProcessor(BaseProcessor):
    dataset = "lem_brazil"

    def process(self) -> None:

        df = geopandas.read_file(self.raw_folder)

        org_len = len(df)

        df = df[df.geometry.is_valid]
        print(f"Removed {org_len - len(df)} invalid geometries")

        org_len = len(df)
        # check for datapoints where the crops were consistent over a year
        grouped_months = {
            2019: [
                "Oct_2019",
                "Nov_2019",
                "Dec_2019",
                "Jan_2020",
                "Feb_2020",
                "Mar_2020",
                "Apr_2020",
            ],
            2020: ["May_2020", "Jun_2020", "Jul_2020", "Aug_2020", "Sep_2020"],
        }

        year_dfs = []
        for year, group in grouped_months.items():
            # construct the condition
            condition = df[group[0]] == df[group[1]]
            for val in group[2:]:
                condition &= df[group[0]] == df[val]

            year_df = df[condition]
            year_df[f"{year}_crop"] = df[group[0]]
            year_dfs.append(year_df)

        df = pd.concat(year_dfs)
        print(f"Removed {org_len - len(df)} changing crop types - {len(df)} points remaining")

        x = df.geometry.centroid.x.values
        y = df.geometry.centroid.y.values

        df["lat"] = y
        df["lon"] = x

        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")
