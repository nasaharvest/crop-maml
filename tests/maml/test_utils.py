from src.maml.utils import sample
from src.maml.datasets.utils import unique_sample

import random
import torch


def test_unique_sample():

    random.seed(0)

    vals = [1, 1, 2, 2, 3, 3]
    indices_to_sample = list(range(len(vals)))

    k = 3
    output_indices = unique_sample(vals, indices_to_sample, k=k)

    selected_vals = [vals[i] for i in output_indices]
    assert len(set(selected_vals)) == len(selected_vals)
    assert len(selected_vals) == k
    for i in [1, 2, 3]:
        assert i in selected_vals


def test_sample():

    batch = (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5]))

    k = 2
    sample_1, state = sample(batch, k=k)

    assert len(sample_1[0]) == k
    assert (sample_1[0] == sample_1[1]).all()

    sample_2, state = sample(batch, k=k, state=state)
    assert (sample_2[0] == sample_2[1]).all()
    assert len(set(sample_2[0].tolist()).intersection(set(sample_1[0].tolist()))) == 0
