"""
Since we are doing 2-way meta-learning, all these functions will take
as input data instances and return 1 or 0
"""
from src.engineer.base import BaseDataInstance
from src.engineer.pv_kenya import PVKenyaDataInstance


def default_to_int(datainstance: BaseDataInstance) -> int:
    assert hasattr(datainstance, "is_crop")
    return datainstance.is_crop


def to_crop_noncrop(datainstance: BaseDataInstance) -> int:
    if hasattr(datainstance, "is_crop"):
        return datainstance.is_crop
    elif hasattr(datainstance, "crop_probability"):
        return datainstance.crop_probability >= 0.5
    elif isinstance(datainstance, PVKenyaDataInstance):
        # it is a crop
        return 1
    else:
        raise RuntimeError(f"Unexpected datainstance, {type(datainstance)}")
