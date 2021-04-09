import numpy as np
import torch
import random
from collections import defaultdict

from typing import cast, Sequence, Tuple, Dict, List, Optional


def unique_sample(vals: List, valid_indices: List[int], k: int) -> List[int]:

    # so we don't mess with the original
    valid_indices = valid_indices.copy()

    MAX_RETRIES = 3

    # This function behaves like random.sample, but if there
    # are duplicate in the list being sampled (vals) it will
    # ensure the subsample returns no duplicate values.

    # first, draw a sample as normal
    selected_indices = random.sample(valid_indices, k)

    # then, check the values which have been selected
    selected_vals = [vals[i] for i in selected_indices]
    if len(set(selected_vals)) == len(selected_vals):
        return selected_indices

    # otherwise, there are duplicates. We will need to find them
    duplicates = defaultdict(list)
    for index, element in enumerate(selected_vals):
        # this is a bit confusing - index here is the value of
        # the element in selected_indices, not its index (since
        # each value represents an index)
        duplicates[element].append(selected_indices[index])

    for element, indices in duplicates.items():
        if len(indices) > 1:
            # this element is duplicated
            for duplicated_index in indices[1:]:
                selected_indices.remove(duplicated_index)

                # add a new index to replace the removed one
                num_tries = 0
                new_index = random.sample(valid_indices, 1)[0]
                while vals[new_index] in set(selected_vals):
                    valid_indices.remove(new_index)
                    new_index = random.sample(valid_indices, 1)[0]
                    num_tries += 1
                    if num_tries >= MAX_RETRIES:
                        raise RuntimeError("Max retries exceeded!")
                selected_indices.append(new_index)
    return selected_indices


def adjust_normalizing_dict(
    dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
) -> Optional[Dict[str, np.ndarray]]:

    for _, single_dict in dicts:
        if single_dict is None:
            return None

    dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

    new_total = sum([x[0] for x in dicts])

    new_mean = sum([single_dict["mean"] * length for length, single_dict in dicts]) / new_total

    new_variance = (
        sum(
            [
                (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2) * length
                for length, single_dict in dicts
            ]
        )
        / new_total
    )

    return {"mean": new_mean, "std": np.sqrt(new_variance)}


def split_tensors(
    x: torch.Tensor, y: torch.Tensor, val_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    indices = torch.rand(x.shape[0]).to(x.device)

    train_indices = indices >= val_ratio
    val_indices = indices < val_ratio

    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]
