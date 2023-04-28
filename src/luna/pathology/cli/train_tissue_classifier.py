import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import fire
import fsspec
import pandas as pd
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.optim as optim
from fsspec import open
from pyarrow.fs import copy_files
from ray import tune
from ray.air import session
from ray.air.config import RunConfig, ScalingConfig, TuneConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)
from torchvision import transforms

from luna.common.custom_logger import init_logger
from luna.common.utils import get_config, load_func, save_metadata, timed
from luna.pathology.analysis.ml import (
    BaseTorchClassifier,
    HDF5Dataset,
    TorchTileClassifierTrainer,
    get_group_stratified_sampler,
)

init_logger()
logger = logging.getLogger("train_tissue_classifier")


@timed
@save_metadata
def cli(
    tile_urlpath: str,
    label_set: Dict,
    label_col: str,
    stratify_col: str,
    num_splits: float,
    num_epochs: Union[int, Callable],
    batch_size: Union[int, Callable],
    learning_rate: Union[float, Callable],
    network: nn.Module,
    use_gpu: bool,
    num_cpus_per_worker: int,
    num_gpus_per_worker: int,
    num_gpus: Optional[int] = None,
    num_cpus: Optional[int] = None,
    output_urlpath: str = ".",
    num_samples: int = 1,
    num_workers: int = 1,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Trains a tissue classifier model based on tile dataset parquet table

    Args:
        tile_urlpath (str): url/path to input dataframe that contains a row for each tile
             and the cooresponding tissue type
        label_set (dict): dictionary that maps tissue types to numerical values
        label_col (str): name of the column in the tile_urlpath that contains the labels (tissue type)
            for each tile
        stratify_col (str): columnn in the tile_urlpath used to stratify the train/test splits,
            such as the patient id or slide id
        num_splits (int): (Optional) The number of folds, must at least be 2.
        num_epochs (Union[int, Callable]): number of epochs to train the model for. Can be either an integer value
            or a Ray.tune distribution (eg ray.tune.choice([10, 15])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        batch_size (Union[int, Callable]): batch size used in PyTorch dataloader. Can be either an integer value
            or a Ray.tune distribution (eg ray.tune.choice([32, 64])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        learning_rate (Union[float, Callable]): the learning rate used for the ADAM optimizer. an be either a float value
            or a Ray.tune distribution (eg ray.tune.loguniform([1.0e-4, 1.0e-1])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        network (nn.Module): The model architecture. Can either be defined in a seperate module, or be from torchvision.models
        use_gpu (bool): whether or not use use GPUs for model training. If set False, the num_gpu flag is ignored.
        num_cpus_per_worker (int): the number of cpus available to each worker. by default the number of workers is set to 1, meaning
            only one trial can be run at a time. If num_workers is increased, num_cpus_per_worker must refer to the number of cpus that
            can be used at the same time (ie if num_workers = 2 and num_cpus_per_worker=10, then 20 cores must be available), and must
            be less than or equal to num_cpus
        num_gpus_per_worker (int): the number of GPUs available for each worker. By default the number of workers is set to 1, meaning
            only one trial can be run at a time. If num_workers is increased, num_gpus_per_worker must refer to the number of GPUs that
            can be used at the same time (ie if num_workers = 2 and num_cpus_per_worker=1, then 2 GPUs must be available), and must
            be less than or equal to num_gpus
        num_gpus (int): total number of GPUs transparent to Ray
        num_cpus (int): total number of CPUs transparent to Ray
        output_urlpath (str): Output url/path. This is the location where Ray is going to save all associated metadata, logs, checkpoints and
            any artifacts from model training
        num_samples (int): This refers to the number of trials Ray is going to run, or how many times it's going to train a model with
            different parameters if performing hyperparameter tuning by setting a distribution on a passed argument, like learning_rate.
        num_workers (int): This refers to the number of workers that Ray will attempt to run in parallel. num_workers and other hardware
            resource arguments must be set in concert. Default = 1.
        storage_options (dict): options to pass to storage reading functions
        output_storage_options (dict): options to pass to storage writing functions

    Returns:
        dict: model fit metadata
    """
    config = get_config(vars())
    analysis = train_model(
        config["tile_urlpath"],
        config["label_set"],
        config["label_col"],
        config["stratify_col"],
        config["num_splits"],
        config["num_epochs"],
        config["batch_size"],
        config["learning_rate"],
        config["network"],
        config["use_gpu"],
        config["num_cpus_per_worker"],
        config["num_gpus_per_worker"],
        config["num_gpus"],
        config["num_cpus"],
        config["output_urlpath"],
        config["num_samples"],
        config["num_workers"],
        config["storage_options"],
        config["output_storage_options"],
    )

    # collect results from Ray
    # analysis interface currently in beta, interface is highly variable between Ray releases
    # can probably imporve output results once ray-train is non-beta.
    best_trial = analysis.get_best_result(metric="accuracy", mode="max")
    trial_df = analysis.get_dataframe()
    logger.info(trial_df)

    results = {
        "result_fpath": config[
            "output_urlpath"
        ],  # path to result folder generated by Ray
        "num_trials": len(trial_df.index),  # total number of trials run
        "best_trial": best_trial,  # df associated with bthe best trial
    }
    logger.info(f"Output: {config['output_urlpath']}")

    return results


class CustomReporter(tune.CLIReporter):
    """Custom RayTune CLI reporter
    Modified to log to data-processing in addition to stdout.
    """

    def report(self, trials, done, *sys_info):
        logger.info(self._progress_str(trials, done, *sys_info))


class TorchTileClassifier(BaseTorchClassifier):
    def setup(self, model):
        self.model = model


def train_func(config: Dict[str, any]):
    """Model training driver function.

    This function takes model training configurations and is called by ray.train.run to

    Args:
        config (dict[str, any]): a json-like dictionary containing parameters and configuration variables
            required for the model training procedure.
    """

    logger.info("Configuring model training driver function...")

    batch_size = config.get("batch_size")
    lr = config.get("learning_rate")
    epochs = config.get("num_epochs")
    n_workers = config.get("num_cpus_per_worker")
    tile_urlpath = config.get("tile_urlpath")
    num_splits = config.get("num_splits")
    label_set = config.get("label_set")
    label_col = config.get("label_col")
    stratify_by = config.get("stratify_col")
    network = config.get("network")
    checkpoint_urlpath = config.get("checkpoint_urlpath")
    storage_options = config.get("storage_options")
    output_storage_options = config.get("output_storage_options")

    with open(tile_urlpath, **storage_options) as of:
        df = (
            pd.read_parquet(of)
            .query("intersection_area > 0")
            .reset_index()
            .set_index("address")
        )

    # reset index
    df_nh = df.reset_index()
    # stratify by slide id while balancing regional_label
    train_sampler, val_sampler = get_group_stratified_sampler(
        df_nh, label_col, stratify_by, num_splits=num_splits
    )
    # replace string labels with categorical values
    df[label_col] = df[label_col].replace(label_set)
    dataset_local = HDF5Dataset(
        hdf5_manifest=df,
        label_cols=[label_col],
        using_ray=True,
        preprocess=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = DataLoader(
        dataset_local,
        batch_size=int(batch_size),
        sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset_local,
        batch_size=int(batch_size),
        sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
    )

    validation_loader = train.torch.prepare_data_loader(validation_loader)
    train_loader = train.torch.prepare_data_loader(train_loader)

    model = network(num_classes=len(label_set))

    model = train.torch.prepare_model(model)

    classifier = TorchTileClassifier(model=model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    train_metrics = MetricCollection(
        [
            Accuracy(),
            Precision(num_classes=len(label_set)),
            Recall(num_classes=len(label_set)),
            F1Score(num_classes=len(label_set)),
        ],
        prefix="train_",
    ).to(train.torch.get_device())

    val_metrics = MetricCollection(
        [
            Accuracy(),
            Precision(num_classes=len(label_set)),
            Recall(num_classes=len(label_set)),
            F1Score(num_classes=len(label_set)),
            ConfusionMatrix(num_classes=len(label_set)),
        ],
        prefix="val_",
    ).to(train.torch.get_device())

    classifier_trainer = TorchTileClassifierTrainer(
        classifier=classifier, criterion=criterion, optimizer=optimizer
    )
    logger.info("Starting training procedure")

    fs, checkpoint_path_prefix = fsspec.core.url_to_fs(
        checkpoint_urlpath, **output_storage_options
    )
    for ii in range(epochs):
        train_results = classifier_trainer.train_epoch(train_loader, train_metrics)
        val_results = classifier_trainer.validate_epoch(validation_loader, val_metrics)

        results = {**train_results, **val_results}
        session.report(results)
        path = Path(checkpoint_path_prefix) / f"checkpoint_{ii}.pt"
        with fs.open(path) as of:
            torch.save({"epoch": ii, "model_state_dict": model.state_dict()}, of)

    logger.info("Completed model training")


def trial_str_creator(trial):
    return "{}_{}_123".format(trial.trainable_name, trial.trial_id)


def train_model(
    tile_urlpath: str,
    label_set: Dict,
    label_col: str,
    stratify_col: str,
    num_splits: int,
    num_epochs: Union[int, Callable],
    batch_size: Union[int, Callable],
    learning_rate: Union[float, Callable],
    network: nn.Module,
    use_gpu: bool,
    num_cpus_per_worker: int,
    num_gpus_per_worker: int,
    num_gpus: int,
    num_cpus: int,
    output_urlpath: str,
    num_samples: int,
    num_workers: int = 1,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Trains a tissue classifier model based on tile dataset parquet table

    Args:
        tile_urlpath (str): url/path to input dataframe that contains a row for each tile
             and the cooresponding tissue type
        label_set (dict): dictionary that maps tissue types to numerical values
        label_col (str): name of the column in the tile_urlpath that contains the labels (tissue type)
            for each tile
        stratify_col (str): columnn in the tile_urlpath used to stratify the train/test splits,
            such as the patient id or slide id
        num_splits (int): (Optional) The number of folds, must at least be 2.
        num_epochs (Union[int, Callable]): number of epochs to train the model for. Can be either an integer value
            or a Ray.tune distribution (eg ray.tune.choice([10, 15])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        batch_size (Union[int, Callable]): batch size used in PyTorch dataloader. Can be either an integer value
            or a Ray.tune distribution (eg ray.tune.choice([32, 64])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        learning_rate (Union[float, Callable]): the learning rate used for the ADAM optimizer. an be either a float value
            or a Ray.tune distribution (eg ray.tune.loguniform([1.0e-4, 1.0e-1])). In the YAML config, this must be specified
            by setting the 'search type' and the 'search space'.
        network (nn.Module): The model architecture. Can either be defined in a seperate module, or be from torchvision.models
        use_gpu (bool): whether or not use use GPUs for model training. If set False, the num_gpu flag is ignored.
        num_cpus_per_worker (int): the number of cpus available to each worker. by default the number of workers is set to 1, meaning
            only one trial can be run at a time. If num_workers is increased, num_cpus_per_worker must refer to the number of cpus that
            can be used at the same time (ie if num_workers = 2 and num_cpus_per_worker=10, then 20 cores must be available), and must
            be less than or equal to num_cpus
        num_gpus_per_worker (int): the number of GPUs available for each worker. By default the number of workers is set to 1, meaning
            only one trial can be run at a time. If num_workers is increased, num_gpus_per_worker must refer to the number of GPUs that
            can be used at the same time (ie if num_workers = 2 and num_cpus_per_worker=1, then 2 GPUs must be available), and must
            be less than or equal to num_gpus
        num_gpus (int): total number of GPUs transparent to Ray
        num_cpus (int): total number of CPUs transparent to Ray
        output_urlpath (str): Output url/path. This is the location where Ray is going to save all associated metadata, logs, checkpoints and
            any artifacts from model training
        num_samples (int): This refers to the number of trials Ray is going to run, or how many times it's going to train a model with
            different parameters if performing hyperparameter tuning by setting a distribution on a passed argument, like learning_rate.
        num_workers (int): This refers to the number of workers that Ray will attempt to run in parallel. num_workers and other hardware
            resource arguments must be set in concert. Default = 1.
        storage_options (dict): options to pass to storage reading functions
        output_storage_options (dict): options to pass to storage writing functions

    Returns:
        ray.tune.result_grid.ResultGrid: model fit result
    """

    logger.info(
        f"Training a tissue classifier with: network={network}, batch_size={batch_size}, learning_rate={learning_rate}"
    )
    logger.info(
        f"Initilizing Ray Cluster, with: num_gpus={num_gpus}, num_workers={num_workers}"
    )

    output = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        dashboard_host="0.0.0.0",
        log_to_driver=True,
    )

    logger.info(f"View Ray Dashboard to see worker logs: {output['webui_url']}")
    logger.info("training model")

    # instantiaing Ray Trainer, setting resouce limits
    logger.info(
        f"Instantiating Ray Trainer with: num_cpus_per_worker={num_cpus_per_worker}, num_gpus_per_worker={num_gpus_per_worker}"
    )

    batch_size = ray.tune.choice(batch_size)[0]
    num_epochs = ray.tune.choice(num_epochs)[0]
    network = load_func(network)

    if len(learning_rate) == 2:
        learning_rate = ray.tune.loguniform(learning_rate[0], learning_rate[1])
    else:
        learning_rate = ray.tune.choice(learning_rate)[0]

    fs, output_path = fsspec.core.url_to_fs(output_urlpath, **output_storage_options)

    if fs.protocol == "file":
        local_output_path = output_path
    else:
        # store ray output in temp directory
        temp_dir = tempfile.TemporaryDirectory()
        local_output_path = temp_dir.name

    # model training configuration parameters
    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_cpus_per_worker": num_cpus_per_worker,
        "tile_urlpath": tile_urlpath,
        "label_set": label_set,
        "label_col": label_col,
        "stratify_col": stratify_col,
        "network": network,
        "num_splits": num_splits,
        "checkpoint_dir": local_output_path,
        "storage_options": storage_options,
    }

    cli_reporter = CustomReporter(max_report_frequency=180)
    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            resources_per_worker={
                "CPU": num_cpus_per_worker,
                "GPU": num_gpus_per_worker,
            },
            num_workers=num_workers,
            use_gpu=use_gpu,
        ),
        run_config=RunConfig(
            local_dir=local_output_path,
            progress_reporter=cli_reporter,
        ),
    )

    logger.info(f"Trainer logs will be logged in: {output_path}")
    os.environ["TUNE_RESULT_DIR"] = output_path

    tuner = ray.tune.Tuner(
        trainable=trainer,
        tune_config=TuneConfig(
            num_samples=num_samples,
        ),
    )
    # run distributed model training

    analysis = tuner.fit()

    ray.shutdown()

    logger.info("Finished training")

    # copy output
    if fs.protocol != "file":
        copy_files(local_output_path, output_path, destination_filesystem=fs)

    return analysis


if __name__ == "__main__":
    fire.Fire(cli)
