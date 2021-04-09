from src.maml.datasets.tasksets import BaseTask

import pytest


class TestBaseTask:
    @pytest.mark.parametrize("cache", [True, False])
    def test_pos_neg_paths(self, monkeypatch, cache, tmpdir):

        neg_paths = ["a0", "a1", "a2"]
        pos_paths = [f"b{i}" for i in range(1000)]

        def __getitem__(self, index: int):
            return self.pickle_files[index], self.y_vals[index]

        monkeypatch.setattr(BaseTask, "__getitem__", __getitem__)

        task = BaseTask(negative_paths=neg_paths, positive_paths=pos_paths, cache=cache)

        for idx, path in enumerate(task.pickle_files):

            if path.startswith("a"):
                assert idx in task.negative_indices
                assert task.y_vals[idx] == 0
            elif path.startswith("b"):
                assert idx in task.positive_indices
                assert task.y_vals[idx] == 1
