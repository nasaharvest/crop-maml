from pathlib import Path
import pickle
import xarray as xr
import json
import torch
from tqdm import tqdm
import numpy as np

from .lstm import Classifier
from .utils import tif_to_np, preds_to_xr
from .datasets.data import BaseDataLoader

from typing import Dict, List, Optional


def predict(
    path_to_version_folder: Path,
    path_to_file: Path,
    batch_size: int = 64,
    add_ndvi: bool = True,
    add_ndwi: bool = False,
    nan_fill: float = 0,
    days_per_timestep: int = 30,
    use_gpu: bool = True,
    prefix: Optional[str] = None,
) -> xr.Dataset:

    device: torch.Device = torch.device("cpu")
    if use_gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("No GPU - not using one")
        device = torch.device("cuda" if use_cuda else "cpu")

    with (path_to_version_folder / "normalizing_dict.pkl").open("rb") as f:
        normalizing_dict = pickle.load(f)

    # load the model info
    with (path_to_version_folder / "model_info.json").open("r") as f:
        model_info = json.load(f)

    model = _load_model(path_to_version_folder, model_info, device, prefix)

    model.eval()

    input_data = tif_to_np(
        path_to_file,
        add_ndvi=add_ndvi,
        add_ndwi=add_ndwi,
        nan=nan_fill,
        normalizing_dict=normalizing_dict,
        days_per_timestep=days_per_timestep,
    )

    predictions: List[np.ndarray] = []
    cur_i = 0

    pbar = tqdm(total=input_data.x.shape[0] - 1)
    while cur_i < (input_data.x.shape[0] - 1):

        batch_x_np = input_data.x[cur_i : cur_i + batch_size]
        if model_info["remove_b1_b10"]:
            batch_x_np = BaseDataLoader._remove_bands(batch_x_np)
        batch_x = torch.from_numpy(batch_x_np).float()

        if use_gpu and (device is not None):
            batch_x = batch_x.to(device)

        with torch.no_grad():
            batch_preds = model.forward(batch_x).cpu().numpy()

        predictions.append(batch_preds)
        cur_i += batch_size
        pbar.update(batch_size)

    all_preds = np.concatenate(predictions, axis=0)
    if len(all_preds.shape) == 1:
        all_preds = np.expand_dims(all_preds, axis=-1)

    return preds_to_xr(all_preds, lats=input_data.lat, lons=input_data.lon,)


def _load_model(
    path_to_version_folder: Path, model_info: Dict, device: torch.device, prefix: Optional[str],
) -> Classifier:

    classifier = Classifier(
        input_size=model_info["input_size"],
        classifier_vector_size=model_info["classifier_vector_size"],
        classifier_dropout=model_info["classifier_dropout"],
        classifier_base_layers=model_info["classifier_base_layers"],
        num_classification_layers=model_info["num_classification_layers"],
    )

    filename = "test_model.pth"
    if prefix is not None:
        filename = f"{prefix}_{filename}"

    classifier.load_state_dict(torch.load(path_to_version_folder / filename, map_location=device))

    return classifier
