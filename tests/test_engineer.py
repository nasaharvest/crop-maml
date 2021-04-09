from src.engineer.base import BaseEngineer
from src.engineer import GeoWikiEngineer
from src.exporters import GeoWikiExporter, GeoWikiSentinelExporter

import numpy as np
import xarray as xr


class TestEngineer:
    def test_sentinel_files_correctly_retrieved(self, tmp_path):

        # setup the files
        sentinel_hub_path = tmp_path / "raw" / GeoWikiSentinelExporter.dataset
        sentinel_hub_path.mkdir(parents=True)

        num_files = 10
        for i in range(num_files):

            (sentinel_hub_path / f"{i}.tif").touch()

            # we want to make sure the engineer doesn't pick these guys up
            (sentinel_hub_path / f"{i}.npy").touch()

        class MockEngineer:
            sentinel_dataset = GeoWikiSentinelExporter.dataset

        engineer = MockEngineer()
        sentinel_files = GeoWikiEngineer.get_geospatial_files(engineer, tmp_path)
        assert len(sentinel_files) == num_files

        for i in range(num_files):
            assert (sentinel_hub_path / f"{i}.tif") in sentinel_files

    def test_find_nearest(self):
        array = np.array([1, 2, 3, 4, 5])
        target = 1.1
        assert BaseEngineer.find_nearest(array, target) == 1

    def test_mixed_nan_to_num(self):

        array = np.array([1, np.nan, 1, np.nan])

        # case 1 - the array has less nans than would be triggered
        num_array = BaseEngineer.maxed_nan_to_num(array, nan=0, max_ratio=0.75)
        assert (num_array == np.array([1, 0, 1, 0])).all()

        # case 2 - the array has more nans than would be triggered
        num_array = BaseEngineer.maxed_nan_to_num(array, nan=0, max_ratio=0.25)
        assert num_array is None

    def test_normalizing_dict_correctly_calculated(self, tmp_path):

        # setup the files)
        processed_folder = tmp_path / "processed" / GeoWikiExporter.dataset
        processed_folder.mkdir(parents=True)
        xr.Dataset({"FEATURES": np.ones(10)}).to_netcdf(processed_folder / "data.nc")

        array_1 = np.array([[1, 2, 3], [2, 2, 4]])
        array_2 = np.array([[2, 2, 4], [1, 2, 3]])

        engineer = GeoWikiEngineer(tmp_path)

        engineer.update_normalizing_values(engineer.normalizing_dict_interim, array_1)
        engineer.update_normalizing_values(engineer.normalizing_dict_interim, array_2)

        normalizing_values = engineer.calculate_normalizing_dict(engineer.normalizing_dict_interim)

        assert (normalizing_values["mean"] == np.array([1.5, 2, 3.5])).all()
        assert (
            normalizing_values["std"]
            == np.array([np.std([1, 2, 2, 1], ddof=1), 0, np.std([3, 4, 4, 3], ddof=1)])
        ).all()
