import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from fsspec import open
from PIL import Image

# from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import MetricCollection

from luna.pathology.common.utils import get_tile_array


class TorchTransformModel:
    def get_preprocess(self, **kwargs):
        """The transform model's preprocessing code

        Args:
            kwargs: Keyword arguements passed onto the subclass method
        """
        raise NotImplementedError(
            "get_preprocess() has not been implimented in the subclass!"
        )

    def transform(self, X: torch.Tensor):
        """Main transformer method, X -> X'

        Args:
            X (torch.Tensor): input tensor

        Returns:
            torch.tensor: Output tile as preprocessed tensor
        """
        raise NotImplementedError(
            "transform() has not been implimented in the subclass!"
        )

    pass


class HDF5Dataset(Dataset):
    """General dataset that uses a HDF5 manifest convention

    Applies preprocessing steps per instance, returning aggregate batches of data. Useful for training and inference.
    """

    def __init__(
        self, hdf5_manifest, preprocess=nn.Identity(), label_cols=[], using_ray=False
    ):
        """Initialize HD5FDataset

        Args:
            hdf5_manifest (pd.DataFrame): Dataframe of H5 data
            preprocess (transform): Function to apply to every bit of data
            label_cols (list[str]): (Optional) label columns to return as tensors, e.g. for training
            using_ray (bool): (Optional) Perform distributed dataloading with Ray for training
        """

        self.hdf5_manifest = hdf5_manifest
        self.label_cols = label_cols
        self.using_ray = using_ray
        self.preprocess = preprocess

    def __len__(self):
        return len(self.hdf5_manifest)

    def set_preprocess(self, preprocess):
        preprocess = preprocess

    def __repr__(self):
        return f"HD5FDataset with {len(self.hd5f_manifest)} tiles, indexed by {self.hd5f_manifest.index.names}, returning label columns: {self.label_cols}"

    def __getitem__(self, idx: int):
        """Tile accessor

        Loads a tile image from the tile manifest.  Returns a batch of the indices of the input dataframe, the tile data always.
        If label columns where specified, the 3rd position of the tuple is a tensor of the label data. If Ray is being used for
        model training, then only the image data and the label is returned.

        Args:
            idx (int): Integer index

        Returns:
            (optional str, torch.tensor, optional torch.tensor): tuple of the tile index and corresponding tile as a torch tensor, and metadata labels if specified, else the index
        """

        row = self.hdf5_manifest.iloc[idx]
        img = get_tile_array(row)

        if self.using_ray and not (len(self.label_cols)):
            raise ValueError(
                "If using Ray for training, you must provide a label column"
            )
        if len(self.label_cols):
            return self.preprocess(img), torch.tensor(row[self.label_cols]).squeeze()
        else:
            return self.preprocess(img), row.name


def post_transform_to_2d(input: np.array) -> np.array:
    """Convert input to a 2D numpy array on CPU

    Args:
        input (torch.tensor): tensor input of shape [B, *] where B is the batch dimension
    """
    if type(input) == torch.tensor:
        input = input.cpu.numpy()

    if not len(input.shape) == 2:
        warnings.warn(f"Reshaping model output (was {input.shape}) to 2D")
        input = np.reshape(input, (input.shape[0], -1))

    return input


class BaseTorchTileDataset(Dataset):
    """Base class for a tile dataset

    Impliments the usual torch dataset methods, and additionally provides a decoding of the binary tile data.
    PIL images can be further preprocessed before becoming torch tensors via an abstract preprocess method

    Will send the tensors to gpu if available, on the device specified by CUDA_VISIBLE_DEVICES="1"
    """

    def __init__(
        self,
        tile_manifest: pd.DataFrame = None,
        tile_urlpath: str = "",
        label_cols: List[str] = [],
        using_ray: bool = False,
        storage_options: dict = {},
        **kwargs,
    ):
        """Initialize BaseTileDataset

        Can accept either a tile dataframe or a path to tile data

        Args:
            tile_manifest (pd.DataFrame): Dataframe of tile data
            tile_path (str): Base path of tile data
            label_cols (list[str]): (Optional) label columns to return as tensors, e.g. for training
            using_ray (bool): (Optional) Perform distributed dataloading with Ray for training
        """

        if tile_manifest is not None:
            self.tile_manifest = tile_manifest
        elif tile_urlpath:
            with open(tile_urlpath, **storage_options) as of:
                self.tile_manifest = pd.read_csv(of).set_index("address")
        else:
            raise RuntimeError("Must specifiy either tile_manifest or tile_path")

        self.label_cols = label_cols
        self.using_ray = using_ray

        self.setup(**kwargs)

    def __len__(self):
        return len(self.tile_manifest)

    def __repr__(self):
        return f"TileDataset with {len(self.tile_manifest)} tiles, indexed by {self.tile_manifest.index.names}, returning label columns: {self.label_cols}"

    def __getitem__(self, idx: int):
        """Tile accessor

        Loads a tile image from the tile manifest.  Returns a batch of the indices of the input dataframe, the tile data always.
        If label columns where specified, the 3rd position of the tuple is a tensor of the label data. If Ray is being used for
        model training, then only the image data and the label is returned.

        Args:
            idx (int): Integer index

        Returns:
            (optional str, torch.tensor, optional torch.tensor): tuple of the tile index and corresponding tile as a torch tensor, and metadata labels if specified
        """

        row = self.tile_manifest.iloc[idx]
        img = Image.fromarray(get_tile_array(row))

        if self.using_ray:
            if not (len(self.label_cols)):
                raise ValueError(
                    "If using Ray for training, you must provide a label column"
                )
            return self.preprocess(img), torch.tensor(row[self.label_cols]).squeeze()

        if len(self.label_cols):
            return (
                row.name,
                self.preprocess(img),
                torch.tensor(row[self.label_cols].to_list()),
            )
        else:
            return row.name, self.preprocess(img)

    def setup(self, **kwargs):
        """Set additional attributes for dataset class

        Template/abstract method where a dataset is configured

        Args:
            kwargs: Keyword arguements passed onto the subclass method
        """
        raise NotImplementedError("setup() has not been implimented in the subclass!")

    def preprocess(self, input_tile: Image):
        """Preprocessing method called for each tile patch

        Loads a tile image from the tile manifest, must be manually implimented to accept a single PIL image and return a torch tensor.

        Args:
            input_tile (Image): Integer index

        Returns:
            torch.tensor: Output tile as preprocessed tensor
        """
        raise NotImplementedError(
            "preprocess() has not been implimented in the subclass!"
        )


class BaseTorchClassifier(nn.Module):
    def __init__(self, **kwargs):
        """Initialize BaseTorchClassifier

        A generic base class for a PyTorch classifier model. This serves as the base class inhereted
        for model training and inference.

        Will run on cuda if available, on the device specified by the CUDA_VISIBLE_DEVICES environment variable

        Args:
            kwargs: Keyward arguements passed onto the subclass method
        """

        super(BaseTorchClassifier, self).__init__()

        self.cuda_is_available = torch.cuda.is_available()

        self.setup(**kwargs)

        if self.cuda_is_available:
            self.cuda()

    def setup(self, **kwargs):
        """Set classifier modules

        Template/abstract method where individual modules that make up the forward pass are configured

        Args:
            kwargs: Keyword arguements passed onto the subclass method
        """
        raise NotImplementedError("setup() has not been implimented in the subclass!")


class BaseTorchTileClassifier(BaseTorchClassifier):
    def forward(self, index, tile_data):
        """Forward pass for base classifier class

        Loads a tile image from the tile manifest

        Args:
            index (list[str]): Tile address indicies with length B
            tile_data (torch.tensor): Input tiles of shape (B, *)

        Returns:
            pd.DataFrame: Dataframe of output features
        """
        if self.cuda_is_available:
            tile_data = tile_data.cuda()
        self.eval()
        with torch.no_grad():
            return pd.DataFrame(
                self.predict(tile_data).cpu().numpy(),
                index=index,
            )

    def setup(self, **kwargs):
        """Set classifier modules

        Template/abstract method where individual modules that make up the forward pass are configured

        Args:
            kwargs: Keyword arguements passed onto the subclass method
        """
        raise NotImplementedError("setup() has not been implimented in the subclass!")

    def predict(self, input_tiles: torch.tensor):
        """predict method

        Loads a tile image from the tile manifest, must be manually implimented to pass the input tensor through the modules specified in setup()

        Args:
            input_tiles (torch.tensor): Input tiles of shape (B, *)

        Returns:
            torch.tensor: 2D tensor with (B, C) where B is the batch dimension and C are output classes or features
        """
        raise NotImplementedError("predict() has not been implimented in the subclass!")


def get_group_stratified_sampler(
    df_nh: pd.DataFrame,
    label_col: str,
    group_col: str,
    num_splits: int = 5,
    random_seed: int = 42,
) -> Tuple[List, List]:
    """Generates sampler indices for torch DataLoader object that are
    stratified by a given group set (ie a column in a dataframe
    corresponding to patient identifiers), and balanced between target
    labels

    Args:
        df_nh (pd.DataFrame): A non-hierarchical/non-multi-indexed/flat dataframe
        label_col (str): The column name for the classes to balance across training and validation splits.
        group_col (str): The column name used to stratify the data (ie patient ids).
        num_splits (int): (Optional) The number of folds, must at least be 2.
    Returns:
        Tuple(List, List): a tuple of indices that correspond to training and validation samplers
    """

    cv = StratifiedGroupKFold(
        n_splits=num_splits, random_state=random_seed, shuffle=True
    )
    classes = df_nh[label_col]
    groups = df_nh[group_col]
    for fold_idx, (train_indices, val_indices) in enumerate(
        cv.split(df_nh, classes, groups)
    ):
        # check integrity. asserts that same group (ie patients) aren't in both
        # train and validation splits
        train_groups, val_groups = groups[train_indices], groups[val_indices]
        assert len(set(train_groups) & set(val_groups)) == 0

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return (train_sampler, val_sampler)
