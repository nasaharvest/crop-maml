from pathlib import Path
import json
import pickle
import dill
import warnings

import torch
from torch import nn
from torch import optim

import learn2learn as l2l
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import subprocess

from .lstm import Classifier
from .datasets import (
    TogoMetaLoader,
    FromPathsDataLoader,
    CropNonCropFromPaths,
    KenyaCropTypeFromPaths,
    FromPathsTestLoader,
    MaliCropTypeFromPaths,
    BrazilCropTypeFromPaths,
    TestBaseLoader,
)
from .datasets.utils import adjust_normalizing_dict

from src.utils import set_seed
from src.maml.utils import sample

from typing import Dict, Tuple, Optional, List


class Learner:
    """
    The MAML learner.

    :param data_folder: The path to the data folder. Training arrays
        should be stored in data_folder / features
    :param k: The number of positive and negative samples to use per task
        for the inner loop training
    :param update_val_size: The number of positive and negative samples to use
        per task for the outer loop training
    :param classifier_vector_size: The size of the LSTM hidden vector
    :param classifier_dropout: Dropout to use between LSTM timesteps
    :param classifier_base_layers: Number of LSTM layers to use
    :param remove_b1_b10: Whether to remove B1 and B10 bands from the timeseries
    :param cache: Whether to save arrays in memory to speed training up
    :param use_cuda: If available, whether to use a GPU
    :param val_size: The ratio of tasks to use as validation tasks
    :param negative_crop_sampling: For crop type tasks, oversample other crops in
        the negative class by this much
    :param validate_on_crop_types: Whether to include crop type tasks in the validation
        tasks. If False, they will all be used for training
    """

    def __init__(
        self,
        data_folder: Path,
        k: int = 10,
        update_val_size: int = 8,
        classifier_vector_size: int = 128,
        classifier_dropout: float = 0.2,
        classifier_base_layers: int = 1,
        num_classification_layers: int = 2,
        remove_b1_b10: bool = True,
        cache: bool = True,
        use_cuda: bool = True,
        val_size: float = 0.2,
        negative_crop_sampling: int = 10,
        validate_on_crop_types: bool = False,
    ) -> None:

        # update val size needs to be divided by 2 since
        # k is multiplied by 2 (k = number of positive / negative vals)
        min_total_k = k + (update_val_size // 2)

        self.model_info: Dict = {
            "k": k,
            "update_val_size": update_val_size,
            "classifier_vector_size": classifier_vector_size,
            "classifier_dropout": classifier_dropout,
            "classifier_base_layers": classifier_base_layers,
            "num_classification_layers": num_classification_layers,
            "remove_b1_b10": remove_b1_b10,
            "negative_crop_sampling": negative_crop_sampling,
            "git-describe": subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8"),
        }

        set_seed()

        self.data_folder = data_folder
        if use_cuda and torch.cuda.is_available():
            self.use_cuda = use_cuda
            self.device = torch.device("cuda")
        else:
            self.use_cuda = False
            self.device = torch.device("cpu")
        print(f"Using {self.device}")

        normalizing_dicts: List[Tuple[int, Optional[Dict[str, np.ndarray]]]] = []
        self.train_k, self.train_tasks, self.val_tasks = [], {}, {}

        self.test_datasets = {}

        country_train_tasks, country_val_tasks = CropNonCropFromPaths.make_train_val(
            data_folder, val_size=val_size, min_k=min_total_k, cache=True, test_country="Togo",
        )

        self._load_dataset(country_train_tasks, country_val_tasks)
        normalizing_dicts.extend(CropNonCropFromPaths.load_normalizing_dicts(data_folder))

        (crop_train_tasks, crop_val_tasks, test_paths,) = KenyaCropTypeFromPaths.make_train_val(
            data_folder,
            val_size=val_size if validate_on_crop_types else 0,
            min_k=min_total_k,
            cache=True,
            test_crop="common_beans",
            negative_crop_sampling=negative_crop_sampling,
        )
        self._load_dataset(crop_train_tasks, crop_val_tasks, prefix="Kenya")
        normalizing_dicts.extend(KenyaCropTypeFromPaths.load_normalizing_dicts(data_folder))

        (mali_crop_train_tasks, mali_crop_val_tasks, _,) = MaliCropTypeFromPaths.make_train_val(
            data_folder,
            val_size=val_size if validate_on_crop_types else 0,
            min_k=min_total_k,
            cache=True,
            negative_crop_sampling=negative_crop_sampling,
            test_crop=None,
        )
        normalizing_dicts.extend(MaliCropTypeFromPaths.load_normalizing_dicts(data_folder))
        self._load_dataset(mali_crop_train_tasks, mali_crop_val_tasks, prefix="Mali")

        (
            brazil_crop_train_tasks,
            brazil_crop_val_tasks,
            brazil_test_paths,
        ) = BrazilCropTypeFromPaths.make_train_val(
            data_folder,
            val_size=val_size if validate_on_crop_types else 0,
            min_k=min_total_k,
            cache=True,
            negative_crop_sampling=negative_crop_sampling,
            test_crop="Coffee",
        )
        normalizing_dicts.extend(BrazilCropTypeFromPaths.load_normalizing_dicts(data_folder))
        self._load_dataset(brazil_crop_train_tasks, brazil_crop_val_tasks, prefix="Brazil")

        mean_train_k = np.mean(self.train_k)
        median_train_k = np.median(self.train_k)
        print(f"Mean training k: {mean_train_k}, median: {median_train_k}")

        self.model_info["training_ks"] = self.train_k
        self.model_info["mean_training_k"] = mean_train_k
        self.model_info["median_training_k"] = median_train_k

        normalizing_dict = adjust_normalizing_dict(normalizing_dicts)

        self.train_dl = FromPathsDataLoader(
            label_to_tasks=self.train_tasks, normalizing_dict=normalizing_dict, device=self.device,
        )

        self.val_dl = FromPathsDataLoader(
            label_to_tasks=self.val_tasks, normalizing_dict=normalizing_dict, device=self.device,
        )

        self.test_datasets = {
            "Togo": TogoMetaLoader(
                data_folder=data_folder,
                remove_b1_b10=remove_b1_b10,
                cache=cache,
                normalizing_dict=self.train_dl.normalizing_dict,
                device=self.device,
            )
        }

        self.test_datasets["common_beans"] = FromPathsTestLoader(
            negative_paths=test_paths[0],
            positive_paths=test_paths[1],
            remove_b1_b10=remove_b1_b10,
            normalizing_dict=self.train_dl.normalizing_dict,
            num_test=10,
            cache=cache,
            device=self.device,
        )

        self.test_datasets["coffee"] = FromPathsTestLoader(
            negative_paths=brazil_test_paths[0],
            positive_paths=brazil_test_paths[1],
            remove_b1_b10=remove_b1_b10,
            normalizing_dict=self.train_dl.normalizing_dict,
            num_test=10,
            cache=cache,
            device=self.device,
        )
        sample = self.train_dl.sample(k=1, noise_factor=0)
        input_size = sample[list(sample.keys())[0]][1][0].shape[1]

        self.model_info["input_size"] = input_size

        self.model = Classifier(
            input_size=input_size,
            classifier_base_layers=classifier_base_layers,
            classifier_dropout=classifier_dropout,
            classifier_vector_size=classifier_vector_size,
            num_classification_layers=num_classification_layers,
        ).to(self.device)

        self.loss = nn.BCELoss(reduction="mean")

        self.results_dict = {
            "meta_train": [],
            "meta_val": [],
            "meta_val_auc": [],
            "meta_train_auc": [],
        }

        self.train_info: Dict = {}
        self.maml: Optional[l2l.algorithms.MAML] = None

        model_folder = self.data_folder / "maml_models"
        model_folder.mkdir(exist_ok=True)

        model_increment = 0
        model_name = f"version_{model_increment}"

        while (model_folder / model_name).exists():
            model_increment += 1
            model_name = f"version_{model_increment}"

        self.version_folder = model_folder / model_name
        self.version_folder.mkdir()

        with (self.version_folder / "model_info.json").open("w") as f:
            json.dump(self.model_info, f)

    def _load_dataset(self, train_tasks: Dict, val_tasks: Dict, prefix: str = "") -> None:

        self.train_tasks.update(train_tasks)
        self.val_tasks.update(val_tasks)

        self.train_k.extend([task[1].task_k for task in train_tasks.values()])

        print(f"{prefix} Using {len(train_tasks)} train, " f"{len(val_tasks)} val tasks")

        if prefix is not None:
            prefix = f"{prefix}_"
        self.model_info[f"{prefix}train_tasks"] = list(train_tasks.keys())
        self.model_info["{prefix}val_tasks"] = list(val_tasks.keys())

    def _fast_adapt(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        learner: nn.Module,
        k: int,
        val_size: int,
        calc_auc_roc: bool,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:

        batch_size = k * 2  # k is the number of positive, negative examples
        data, labels = batch

        max_train_index = int(data.shape[0] - val_size)
        adaptation_data, adaptation_labels = (
            data[:max_train_index],
            labels[:max_train_index],
        )
        evaluation_data, evaluation_labels = (
            data[max_train_index:],
            labels[max_train_index:],
        )

        train_auc_roc: Optional[float] = None
        valid_auc_roc: Optional[float] = None

        # Adapt the model
        num_adaptation_steps = len(adaptation_data) // batch_size  # should divide cleanly
        for i in range(num_adaptation_steps):
            x = adaptation_data[i * batch_size : (i + 1) * batch_size]
            y = adaptation_labels[i * batch_size : (i + 1) * batch_size]
            train_preds = learner(x).squeeze(dim=1)
            train_error = self.loss(train_preds, y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # currently getting a UserWarning from PyTorch here, but it can be ignored
                learner.adapt(train_error)
            if calc_auc_roc:
                train_auc_roc = roc_auc_score(y.cpu().numpy(), train_preds.detach().cpu().numpy())

        # Evaluate the adapted model
        if len(evaluation_data) > 0:
            preds = learner(evaluation_data).squeeze(dim=1)
            valid_error = self.loss(preds, evaluation_labels)
            if calc_auc_roc:
                valid_auc_roc = roc_auc_score(
                    evaluation_labels.cpu().numpy(), preds.detach().cpu().numpy()
                )
            else:
                valid_auc_roc = None
            return train_error, train_auc_roc, valid_error, valid_auc_roc
        return train_error, train_auc_roc, None, None

    def train(
        self,
        update_lr: float = 0.001,
        meta_lr: float = 0.001,
        min_meta_lr: float = 0.00001,
        max_adaptation_steps: int = 3,
        num_iterations: int = 2000,
        noise_factor: float = 0,
        save_best_val: bool = True,
        checkpoint_every: int = 20,
        save_train_task_results: bool = True,
        schedule: bool = False,
    ) -> None:
        """
        Train the MAML model

        :param update_lr: The learning rate to use when learning a specific task
            (inner loop learning rate)
        :param meta_lr: The learning rate to use when updating the MAML model
            (outer loop learning rate)
        :param min_meta_lr: The minimum meta learning rate to use for the cosine
            annealing scheduler. Only used if `schedule == True`
        :param max_adaptation_steps: The maximum number of adaptation steps to be used
            per task. Each task will do as many adaptation steps as possible (up to this
            upper bound) given the number of unique positive and negative data instances
            it has
        :param num_iterations: The number of iterations to train the meta-model for. One
            iteration is a complete pass over all the tasks
        :param noise_factor: The amount of gaussian noise to add to the timeseries
        :param save_best_val: Whether to save the model with the best validation score,
            as well as the final model
        :param checkpoint_every: The model prints out training statistics every
            `checkpoint_every` iteration
        :param save_train_tasks_results: Whether to save the results for the training
            tasks to a json object
        :param schedule: Whether to use cosine annealing on the meta learning rate during
            training
        """

        self.train_info = {
            "update_lr": update_lr,
            "meta_lr": meta_lr,
            "max_adaptation_steps": max_adaptation_steps,
            "num_iterations": num_iterations,
            "noise_factor": noise_factor,
        }

        k = self.model_info["k"]
        update_val_size = self.model_info["update_val_size"]
        val_k = update_val_size // 2

        with (self.version_folder / "train_info.json").open("w") as f:
            json.dump(self.train_info, f)

        self.maml = l2l.algorithms.MAML(self.model, lr=update_lr, first_order=False)
        opt = optim.Adam(self.maml.parameters(), meta_lr)

        scheduler = None
        if schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=num_iterations, eta_min=min_meta_lr
            )

        best_val_score = np.inf

        val_labels = self.val_dl.task_labels
        self.val_results: Dict[str, Dict[str, List]] = {}
        self.train_results: Dict[str, Dict[str, List]] = {}

        labels = self.train_dl.task_labels

        for iteration_num in tqdm(range(num_iterations)):

            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0
            meta_train_auc = 0.0
            meta_valid_auc_roc = 0.0

            total_weight = 0.0

            for task_label in labels:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone()

                # find the maximum number of adaptation steps this task can have,
                # based on the task_k
                task_k = self.train_dl.task_k(task_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                weight, batch = self.train_dl.sample_task(
                    task_label, k=final_k, noise_factor=noise_factor
                )
                _, _, evaluation_error, train_auc = self._fast_adapt(
                    batch, learner, k=k, val_size=update_val_size, calc_auc_roc=True
                )

                if task_label not in self.train_results:
                    self.train_results[task_label] = {
                        "batch_size": [],
                        "AUC": [],
                        "loss": [],
                    }

                self.train_results[task_label]["batch_size"].append(len(batch[0]))
                self.train_results[task_label]["AUC"].append(train_auc)
                self.train_results[task_label]["loss"].append(evaluation_error.item())

                evaluation_error = evaluation_error * weight
                total_weight += weight

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_auc += train_auc

            for val_label in val_labels:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone()

                task_k = self.val_dl.task_k(val_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                weight, val_batch = self.val_dl.sample_task(
                    val_label, k=final_k, noise_factor=noise_factor
                )
                _, _, val_error, val_auc = self._fast_adapt(
                    val_batch, learner, k=k, val_size=update_val_size, calc_auc_roc=True
                )

                if val_label not in self.val_results:
                    self.val_results[val_label] = {
                        "batch_size": [],
                        "AUC": [],
                        "loss": [],
                    }

                self.val_results[val_label]["batch_size"].append(len(val_batch[0]))
                self.val_results[val_label]["AUC"].append(val_auc)
                self.val_results[val_label]["loss"].append(val_error.item())

                val_error = val_error * weight

                # note that backwards is not called
                meta_valid_error += val_error.item()
                meta_valid_auc_roc += val_auc

            # Print some metrics
            meta_batch_size = len(labels)
            meta_val_size = len(val_labels)
            self.results_dict["meta_train"].append((meta_train_error / meta_batch_size))
            self.results_dict["meta_val"].append((meta_valid_error / meta_val_size))
            self.results_dict["meta_val_auc"].append(meta_valid_auc_roc / meta_val_size)
            self.results_dict["meta_train_auc"].append(meta_train_auc / meta_batch_size)

            if iteration_num > 0:
                mean_mt = np.mean(self.results_dict["meta_train"][-checkpoint_every:])
                mean_mv = np.mean(self.results_dict["meta_val"][-checkpoint_every:])
                mean_mvauc = np.mean(self.results_dict["meta_val_auc"][-checkpoint_every:])
                mean_mtauc = np.mean(self.results_dict["meta_train_auc"][-checkpoint_every:])

                if mean_mv <= best_val_score:
                    best_val_score = mean_mv
                    if save_best_val:
                        self._checkpoint(iteration=iteration_num)

                if iteration_num % checkpoint_every == 0:
                    print(
                        f"Meta_train: {round(mean_mt, 3)}, meta_val: {round(mean_mv, 3)}, "
                        f"meta_train_auc: {round(mean_mtauc, 3)}, meta_val_auc: {round(mean_mvauc, 3)}"
                    )

            # Average the accumulated gradients and optimize
            for p in self.maml.parameters():
                p.grad.data.mul_(1.0 / total_weight)
            opt.step()

            if schedule:
                assert scheduler is not None
                scheduler.step()

        with (self.version_folder / "results.json").open("w") as rf:
            json.dump(self.results_dict, rf)

        with (self.version_folder / "val_results.json").open("w") as valf:
            json.dump(self.val_results, valf)

        if save_train_task_results:
            with (self.version_folder / "train_results.json").open("w") as trainf:
                json.dump(self.train_results, trainf)

        with (self.version_folder / "final_model.pkl").open("wb") as mf:
            dill.dump(self, mf)

    def _checkpoint(self, iteration: int) -> None:

        checkpoint_files = list(self.version_folder.glob("checkpoint*"))
        if len(checkpoint_files) > 0:
            for filepath in checkpoint_files:
                filepath.unlink()
        with (self.version_folder / f"checkpoint_iteration_{iteration}.pkl").open("wb") as f:
            dill.dump(self, f)

    def test(
        self,
        test_dataset_name: str,
        num_samples: int,
        train_k: Optional[int] = None,
        num_grad_steps: int = 2000,
        prefix: Optional[str] = None,
        save_state_dict: bool = True,
        seed: Optional[int] = None,
        test_mode: str = "maml",
        num_cross_val: int = 10,
    ) -> None:
        r"""
        Take the trained MAML model, and finetune it on one of the test datasets

        :param train_dataset_name: One of {"Togo", "common_beans", "coffee"}
        :num_samples: The number of samples to use from the test task
        :param train_k: The training k to use. If None, defaults to the k used when training
            the MAML model
        :param num_grad_steps: The number of gradient steps to finetune the model for
        :param prefix: The prefix to use when saving the model results
        :param save_state_dict: Whether to save the final state dict of the classifier
        :param seed: The random seed to use
        :test_mode: One of {"maml", "pretrained", "random"} - which weights to use when starting
            the finetuning. "pretrained" requires the pretrain.py script to have been run for this
            MAML version
        :param num_cross_val: The number of cross validation runs to do (i.e. how often this model
            should be finetuned with a different random seed)
        """
        if seed is not None:
            set_seed(seed)
        else:
            # use the default argument
            set_seed()

        assert test_mode in ["maml", "pretrained", "random"]

        test_dl = self.test_datasets[test_dataset_name]

        assert self.maml is not None, "This model must be trained before it can be tested!"

        test_results: Dict = {
            "grad_steps": [],
            "train_auc_roc": [],
            "test_auc_roc": [],
            "test_accuracy": [],
        }

        if train_k is None:
            train_k = self.model_info["k"]

        org_model = Classifier(
            input_size=self.model_info["input_size"],
            classifier_vector_size=self.model_info["classifier_vector_size"],
            classifier_dropout=self.model_info["classifier_dropout"],
            classifier_base_layers=self.model_info["classifier_base_layers"],
            num_classification_layers=self.model_info["num_classification_layers"],
        ).to(self.device)

        if test_mode == "maml":
            org_model.load_state_dict(self.maml.module.state_dict())
        elif test_mode == "pretrained":
            assert (
                self.version_folder / "pretrained.pth"
            ).exists(), "Pretrained model must be trained!"
            org_model.load_state_dict(torch.load(self.version_folder / "pretrained.pth"))
        else:
            assert test_mode == "random"
            # no weights loaded, we use the random weights

        for i in range(num_cross_val):
            model, test_results = self._test_loop(
                org_model,
                test_dl,
                dl_seed=i,
                num_grad_steps=num_grad_steps,
                num_samples=num_samples,
                train_k=train_k,
            )

            file_id = f"cv_{i}_{test_dataset_name}_{test_mode}"

            if self.version_folder is not None:
                filename = f"{file_id}_test_results.json"
                if prefix is not None:
                    filename = f"{prefix}_{filename}"
                with (self.version_folder / filename).open("w") as f:
                    json.dump(test_results, f)

                if save_state_dict:
                    filename = f"{file_id}_test_model.pth"
                    if prefix is not None:
                        filename = f"{prefix}_{filename}"
                    torch.save(model.state_dict(), self.version_folder / filename)

                    # the normalizing dict is for the entire model, not specific to the
                    # test dataset
                    with (self.version_folder / "normalizing_dict.pkl").open("wb") as f:
                        pickle.dump(test_dl.normalizing_dict, f)

            else:
                print("No version folder - how did that happen? Not saving test results")

    def _test_loop(
        self,
        org_model: Classifier,
        test_dl: TestBaseLoader,
        dl_seed: int,
        num_grad_steps: int,
        num_samples: int,
        train_k: int,
    ) -> Tuple[Classifier, Dict]:

        # we make a new model
        model = Classifier(
            input_size=self.model_info["input_size"],
            classifier_vector_size=self.model_info["classifier_vector_size"],
            classifier_dropout=self.model_info["classifier_dropout"],
            classifier_base_layers=self.model_info["classifier_base_layers"],
            num_classification_layers=self.model_info["num_classification_layers"],
        ).to(self.device)

        # then copy over the weights
        model.load_state_dict(org_model.state_dict())

        opt = optim.SGD(model.parameters(), lr=self.train_info["update_lr"])

        state: Optional[List[int]] = None  # used when sampling

        test_results: Dict = {
            "grad_steps": [],
            "train_auc_roc": [],
            "test_auc_roc": [],
            "test_accuracy": [],
        }

        test_dl.cross_val(dl_seed)

        # we divide by 2, because sample_train samples for both the
        # positive and negative values
        train_batch_total = None
        if num_samples != -1:
            train_batch_total = test_dl.sample_train(num_samples // 2)
            test_results["num_samples"] = train_batch_total[0].shape[0]
        else:
            print("Using all available training data!")
            test_results["num_samples"] = test_dl.len_train()

        print(f"Using {test_results['num_samples']} samples for training")

        test_x, test_y = test_dl.get_test_data()
        test_y_np = test_y.cpu().numpy()

        for i in tqdm(range(num_grad_steps)):

            if i != 0:
                model.train()
                opt.zero_grad()

                if num_samples != -1:
                    train_batch, state = sample(train_batch_total, train_k * 2, state)
                else:
                    train_batch = test_dl.sample_train(train_k, deterministic=False)

                train_x, train_y = train_batch
                preds = model(train_x).squeeze(dim=1)
                loss = self.loss(preds, train_y)

                loss.backward()
                opt.step()

                test_results["train_auc_roc"].append(
                    roc_auc_score(train_y.detach().cpu().numpy(), preds.detach().cpu().numpy())
                )

            model.eval()
            test_results["grad_steps"].append(i)
            with torch.no_grad():
                test_preds = model(test_x).cpu().numpy()

                test_results["test_auc_roc"].append(roc_auc_score(test_y_np, test_preds))

                test_preds_binary = (test_preds >= 0.5).astype(int)
                test_results["test_accuracy"].append(accuracy_score(test_y_np, test_preds_binary))

        return model, test_results
