import os
import logging
import click

from typing import Union, Callable, Dict

import pandas as pd
import pyarrow.parquet as pq
import ray
import ray.train as train
import torch.nn as nn
import torch.optim as optim

from ray import tune
from ray.train import Trainer

from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1,
    ConfusionMatrix,
)

from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner, load_func
from luna.pathology.analysis.ml import (
    BaseTorchTileDataset,
    BaseTorchClassifier,
    TorchTileClassifierTrainer,
    get_group_stratified_sampler,
)

init_logger()
logger = logging.getLogger("train_tissue_classifier")


_params_ = [
    ("tile_dataset_fpath", str),
    ("output_dir", str),
    ("label_set", dict),
    ("label_col", str),
    ("stratify_col", str),
    ("num_splits", int),
    ("num_epochs", list),
    ("batch_size", list),
    ("learning_rate", list),
    ("network", str),  # string, but parsed as an object
    ("use_gpu", bool),
    ("num_cpus_per_worker", int),
    ("num_gpus_per_worker", int),
    ("num_gpus", int),
    ("num_cpus", int),
    ("num_workers", int),
    ("num_samples", int),
]


@click.command()
@click.argument("tile_dataset_fpath", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="Path to output directory to save results and logs from Ray",
)
@click.option(
    "-ls",
    "--label_set",
    required=False,
    help="Dictionary/json where keys coorespoond to tissue types and values coorespond to numerical values",
)
@click.option(
    "-lc",
    "--label_cols",
    required=False,
    help="Column name in the input dataframe cooresponding to the tissue type (eg. regional_label)",
)
@click.option(
    "-sc",
    "--stratify_col",
    required=False,
    help="Column name in the input dataframe used to stratify the training/validation datasets (eg. id_slide_container or patient_id)",
)
@click.option(
    "-nk",
    "--num_splits",
    required=False,
    help="The number of folds used for cross validation",
)
@click.option(
    "-ne",
    "--num_epochs",
    required=False,
    help="Number of epochs to train the model for. Can be either a fixed integer or a RayTune grid search",
)
@click.option(
    "-bx",
    "--batch_size",
    required=False,
    help="Batch size used train the model. Can be either a fixed integer or a RayTune grid search",
)
@click.option(
    "-lr",
    "--learning_rate",
    required=False,
    help="Learning rate used for the ADAM optimizer. Can be either a float or a RayTune distribution",
)
@click.option(
    "-nt",
    "--network",
    required=False,
    help="Neural network architecture. Can be either a nn.Module or a RayTune grid search",
)
@click.option(
    "-ug",
    "--use_gpu",
    required=False,
    help="Whether or not use use GPUs for model training",
)
@click.option(
    "-cw",
    "--num_cpus_per_workers",
    required=False,
    help="Number of CPUs transparent to each worker",
)
@click.option(
    "-gw",
    "--num_gpus_per_workers",
    required=False,
    help="Number of GPUs transparent to each worker. Can't be more than num_gpus",
)
@click.option(
    "-ng",
    "--num_gpus",
    required=False,
    help="Number of GPUs in total transparent to Ray",
)
@click.option(
    "-nc",
    "--num_cpus",
    required=False,
    help="Number of CPUs in total transparent to Ray",
)
@click.option(
    "-nw",
    "--num_workers",
    required=False,
    help="Total number of workers. Cooresponds to number of models to train concurrently.",
)
@click.option(
    "-ns",
    "--num_samples",
    required=False,
    help="number of trials to run",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """Train a tissue classifier model for all tiles in a slide

    \b
    Inputs:
        tile_dataset_fpath: path to tile dataset parquet table
    \b
    Outputs:
        ray ExperimentAnalysis dataframe and metadata saved to the output
    \b
    Example:
        train_tissue_classifier /tables/slides/slide_table
            -ne 5
            -nt torchvision.models.resnet18
            -nw 1
            -o results/train_tile_classifier_results

    """

    cli_runner(cli_kwargs, _params_, train_model)


class CustomReporter(tune.CLIReporter):
    """Custom RayTune CLI reporter
    Modified to log to data-processing in addition to stdout.
    """

    def report(self, trials, done, *sys_info):
        logger.info(self._progress_str(trials, done, *sys_info))


class TileDataset(BaseTorchTileDataset):
    def setup(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def preprocess(self, input_tile):
        return self.transform(input_tile)


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
    tile_dataset_fpath = config.get("tile_dataset_fpath")
    num_splits = config.get("num_splits")
    label_set = config.get("label_set")
    label_col = config.get("label_col")
    stratify_by = config.get("stratify_col")
    network = config.get("network")

    df = pq.ParquetDataset(tile_dataset_fpath).read().to_pandas()

    # reset index
    df_nh = df.reset_index()

    # stratify by slide id while balancing regional_label
    train_sampler, val_sampler = get_group_stratified_sampler(
        df_nh, label_col, stratify_by, num_splits=num_splits
    )

    dataset_local = TileDataset(
        tile_manifest=df, label_cols=[label_col], using_ray=True
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
            F1(num_classes=len(label_set)),
        ],
        prefix="train_",
    ).to(train.torch.get_device())

    val_metrics = MetricCollection(
        [
            Accuracy(),
            Precision(num_classes=len(label_set)),
            Recall(num_classes=len(label_set)),
            F1(num_classes=len(label_set)),
            ConfusionMatrix(num_classes=len(label_set)),
        ],
        prefix="val_",
    ).to(train.torch.get_device())

    classifier_trainer = TorchTileClassifierTrainer(
        classifier=classifier, criterion=criterion, optimizer=optimizer
    )

    logger.info("Starting training procedure")
    for ii in range(epochs):

        train_results = classifier_trainer.train_epoch(train_loader, train_metrics)
        val_results = classifier_trainer.validate_epoch(validation_loader, val_metrics)

        results = {**train_results, **val_results}
        train.report(**results)
        train.save_checkpoint(epoch=ii, model=model)

    logger.info("Completed model training")


def train_model(
    tile_dataset_fpath: str,
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
    num_gpus: int,
    num_cpus: int,
    output_dir: str,
    num_samples: int,
    num_workers: int = 1,
):
    """Trains a tissue classifier model based on tile dataset parquet table

    Args:
        tile_dataset_fpath (str): filepath to input dataframe that contains a row for each tile
             and the cooresponding tissue type
        label_set (dict): dictionary that maps tissue types to numerical values
        label_col (str): name of the column in the tile_dataset_fpath that contains the labels (tissue type)
            for each tile
        stratify_col (str): columnn in the tile_dataset_fpath used to stratify the train/test splits,
            such as the patient id or slide id
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
        output_dir (str): Output directory. This is the location where Ray is going to save all associated metadata, logs, checkpoints and
            any artifacts from model training
        num_samples (int): This refers to the number of trials Ray is going to run, or how many times it's going to train a model with
            different parameters if performing hyperparameter tuning by setting a distribution on a passed argument, like learning_rate.
        num_workers (int): This refers to the number of workers that Ray will attempt to run in parallel. num_workers and other hardware
            resource arguments must be set in concert. Default = 1.

    Returns:
        dict: metadata about function call
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
    print("training model")

    # instantiaing Ray Trainer, setting resouce limits
    logger.info(
        f"Instantiating Ray Trainer with: num_cpus_per_worker={num_cpus_per_worker}, num_gpus_per_worker={num_gpus_per_worker}"
    )

    trainer = Trainer(
        backend="torch",
        num_workers=num_workers,
        use_gpu=use_gpu,
        logdir=output_dir,
        resources_per_worker={
            "CPU": num_cpus_per_worker,
            "GPU": num_gpus_per_worker,
        },
    )

    batch_size = ray.tune.choice(batch_size)

    num_epochs = ray.tune.choice(batch_size)

    if len(learning_rate) == 2:
        learning_rate = ray.tune.loguniform(learning_rate[0], learning_rate[1])
    else:
        learning_rate = ray.tune.choice(learning_rate)

    network = load_func(network)

    logger.info(f"Trainer logs will be logged in: {output_dir}")
    os.environ["TUNE_RESULT_DIR"] = output_dir

    # model training configuration parameters
    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_cpus_per_worker": num_cpus_per_worker,
        "tile_dataset_fpath": tile_dataset_fpath,
        "label_set": label_set,
        "label_col": label_col,
        "stratify_col": stratify_col,
        "network": network,
        "num_splits": num_splits,
    }

    trainable = trainer.to_tune_trainable(train_func)

    cli_reporter = CustomReporter(max_report_frequency=180)

    # run distributed model training
    analysis = tune.run(
        trainable,
        config=config,
        local_dir=output_dir,
        mode="max",
        metric="val_Accuracy",
        progress_reporter=cli_reporter,
        log_to_file=True,
        num_samples=num_samples,
    )

    logger.info("Finished training")
    logger.info(f"{analysis.results_df}")

    # collect results from Ray
    # analysis interface currently in beta, interface is highly variable between Ray releases
    # can probably imporve output results once ray-train is non-beta.
    result_df = analysis.results_df
    trial_dfs = analysis.fetch_trial_dataframes()
    trial_df = pd.concat(trial_dfs.values(), ignore_index=True)
    best_trial = trial_df.iloc[pd.Series.idxmax(trial_df["val_Accuracy"])]

    results = {
        "result_fpath": output_dir,  # path to result folder generated by Ray
        "num_trials": len(result_df),  # total number of trials run
        "best_trial": best_trial,  # df associated with bthe best trial
    }

    ray.shutdown()

    return results


if __name__ == "__main__":

    cli()

    pass
