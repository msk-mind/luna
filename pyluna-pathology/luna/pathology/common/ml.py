import pandas as pd
import numpy as np
import torch

from collections import Counter, defaultdict
from typing import Dict, Optional, Union, Tuple, List

from PIL import Image
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_random_state
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import MetricCollection

class BaseTorchTileDataset(Dataset):
    """Base class for a tile dataset
    
    Impliments the usual torch dataset methods, and additionally provides a decoding of the binary tile data.
    PIL images can be further preprocessed before becoming torch tensors via an abstract preprocess method

    Will send the tensors to gpu if available, on the device specified by CUDA_VISIBLE_DEVICES="1"
    """ 
    
    def __init__(self, tile_manifest=None, tile_path=None, label_cols=[], using_ray=False, **kwargs):
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
        elif tile_path is not None: 
            self.tile_manifest = pd.read_csv(tile_path + 'address.slice.csv').set_index("address")
            self.tile_manifest['data_path'] = tile_path + 'tiles.slice.pil'
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
        
        Loads a tile image from the tile manifest.  Returns a batch of the indicies of the input dataframe, the tile data always. 
        If label columns where specified, the 3rd position of the tuple is a tensor of the label data. If Ray is being used for 
        model training, then only the image data and the label is returned. 

        Args:
            idx (int): Integer index 

        Returns:
            (optional str, torch.tensor, optional torch.tensor): tuple of the tile index and corresponding tile as a torch tensor, and metadata labels if specified
        """ 
            
        row = self.tile_manifest.iloc[idx]
        with open(row.data_path, "rb") as fp:
            fp.seek(int(row.tile_image_offset))
            img = Image.frombytes(
                row.tile_image_mode,
                (int(row.tile_image_size_xy), int(row.tile_image_size_xy)),
                fp.read(int(row.tile_image_length)),
            )   

        if self.using_ray:
            if not(len(self.label_cols)):
                raise ValueError("If using Ray for training, you must provide a label column")
            return self.preprocess(img), torch.tensor(row[self.label_cols]).squeeze()

        if len(self.label_cols):                 
            return row.name, self.preprocess(img), torch.tensor(row[self.label_cols]).to_list()
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
        raise NotImplementedError("preprocess() has not been implimented in the subclass!")


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
                
        if self.cuda_is_available: self.cuda()

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
        if self.cuda_is_available: tile_data = tile_data.cuda()
        self.eval()
        with torch.no_grad():
            return pd.DataFrame(self.predict(tile_data).cpu().numpy(), index=index, )

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


class TorchTileClassifierTrainer(object):
    """Simple class to manage training and validation of a BaseTorchClassifier
    used for tile classification
    """
    def __init__(self, classifier: BaseTorchClassifier, criterion:nn.Module, optimizer: torch.optim):
        """Instantiate TorchTileClassifierTrainer instance 

        Args:
            classifier (BaseTorchClassifier): an instance of a BaseTorchClassifier
            criterion (nn.Module): the loss function optimized during training
            optimizer (torch.optim): torch optimizer object used to optimize the parameters of the BaseTorchClassifier

        """
        self.network = classifier.model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, dataloader: DataLoader, metrics:MetricCollection) -> Dict:
        """a simple PyTorch training loop that defines the optimization 
        procedure for a single epoch

        Args:
            dataloader (DataLoader): a PyTorch dataloader object for the training dataset
            metrics (MetricCollection): a collection of metrics, such as accuracy or recall,
                used to evaluate model performance during training
        Returns:
            Dict: A dictionary of metrics used for monitoring and model evaluation 
        """
        num_batches = len(dataloader)
        epoch_loss = 0 

        self.network.train()
        for epoch_iter, (inputs, labels) in enumerate(dataloader):

            # compute forward pass
            preds = self.network(inputs)

            # evaluate loss functional
            loss = self.criterion(preds, labels)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute metrics
            epoch_loss += loss.item()
            metrics(preds, labels)
 
        # aggregate metrics accros batches, push to CPU
        train_metrics = metrics.compute()
        train_metrics = {k: v.cpu().numpy() for k, v in train_metrics.items()}

        epoch_loss /= num_batches

        train_metrics['train_loss'] = epoch_loss

        return train_metrics
        
    def validate_epoch(self, dataloader: DataLoader, metrics: MetricCollection) -> Dict:
        """a simple PyTorch validation loop that defines the model validation procedure
        used during training. 

        Args:
            dataloader (DataLoader): a PyTorch dataloader object for the validation dataset
            metrics (MetricCollection):  a collection of torchmetrics used to evaluate model performance
        Returns:
            Dict: A dictionary of metrics used for monitoring and model evaluation 
        """
        print("starting validation")

        num_batches = len(dataloader)
        self.network.eval()
        loss = 0 

        with torch.no_grad():
            for epoch_iter, (inputs, labels) in enumerate(dataloader):

                # forward pass
                preds = self.network(inputs)

                # compute validation loss
                loss += self.criterion(preds, labels).item()
                
                # compute metrics 
                metrics(preds, labels)
        
        # aggregate metrics across batches, push to CPU
        val_metrics = metrics.compute()
        val_metrics = {k: v.cpu().numpy() for k, v in val_metrics.items()}

        loss /= num_batches

        val_metrics['val_loss'] = loss

        return val_metrics


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.
    See https://github.com/scikit-learn/scikit-learn/issues/13621 for a
    full discussion on group stratified cross validation.

    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).

    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class.
    """

    def __init__(self, n_splits:int=5, shuffle:bool=False, random_state:Union[int, None]=None):
        """Initilize a StratifiedGroupKFold iterator 
        
        Args:
            n_splits (int): Number of folds, must be at least 2. default=5.
            shuffle (bool): Whether to shuffle each class's samples before splitting into batches.
                Note that the samples within each split will not be shuffled. default=False
            random_state (Union[int, None]): when `shuffle` is true, `random_state` affects the ordering of the
                indices, which controls the randomness of each fold for each class.
                otherwise, leave `random_state` as `none`.
                pass an int for reproducible output across multiple function calls. default=None
        """
        super().__init__(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _iter_test_indices(self, X:pd.DataFrame, y:pd.DataFrame, groups:pd.DataFrame) -> List:
        """Computes test indicies based on group-K-fold cross validation. 
        Implementation based on this kaggle kernel:
        https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

        Args:
            X (pd.DataFrame): A non-hierarchical/non-multi-indexed/flat dataframe
            y (pd.DataFrame): The target classes for each sample to balance across
                training and validation splits. Should be a single-column. 
            groups (pd.DataFrame): the group used to stratify the data (ie patient ids).
                 Should be a single-column 
        Returns:    
            List: A list of indicies used for each of the k-folds of cross validation 
        """
        
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(
            groups_and_y_counts, key=lambda x: -np.std(x[1])
        ):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(
                        np.std(
                            [
                                y_counts_per_fold[j][label] / y_distr[label]
                                for j in range(self.n_splits)
                            ]
                        )
                    )
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group in enumerate(groups)
                if group in groups_per_fold[i]
            ]
            yield test_indices


def get_group_stratified_sampler(
    df_nh: pd.DataFrame,
    classes: pd.DataFrame,
    groups: pd.DataFrame,
    split: float = 0.2,
) -> Tuple[List, List]:
    """Generates sampler indicies for torch DataLoader object that are
    stratified by a given group set (ie a column in a dataframe
    cooresponding to patient identifiers), and balanced between target
    labels

    Args:
        df_nh (pd.DataFrame): A non-hierarchical/non-multi-indexed/flat dataframe
        classes (pd.DataFrame): The target classes for each sample to balance across
            training and validation splits. Should be a single-column. 
        groups (pd.DataFrame): the group used to stratify the data (ie patient ids). Should be a single-column 
        split (Float): (Optional) the train/val split. must yield a rational number when divided
        by zero in order to ensure proper balancing and stratification. Default=0.2
    Returns:
        Tuple(List, List): a tuple of indices that coorespond to training and validation samplers
    """
    #
    K = 1 / split
    if K % 1 != 0:
        print(K)
        raise ValueError("Invalid test/train ratio")
    else:
        K = int(K)

    cv = StratifiedGroupKFold(n_splits=K)
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