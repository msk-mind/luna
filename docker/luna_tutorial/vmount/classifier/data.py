"""data
utilities for data manipulation for model training
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_random_state
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class TileDataset(Dataset):
    """basic pandas dataframe tile dataset"""

    def __init__(self, df, label_set):
        self.tile_manifest = df
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.label_set = label_set

    def __len__(self):
        return len(self.tile_manifest)

    def __getitem__(self, idx):
        row = self.tile_manifest.iloc[idx]
        with open(row.data_path, "rb") as fp:
            fp.seek(int(row.tile_image_offset))
            img = Image.frombytes(
                row.tile_image_mode,
                (int(row.tile_image_size_xy), int(row.tile_image_size_xy)),
                fp.read(int(row.tile_image_length)),
            )

        return self.transform(img), self.label_set.get(row["regional_label"])


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

    :param n_splits: Number of folds. Must be at least 2
    :type n_s;ices: int, default=5
    :param shuffle: Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    :type shuffle: bool, default=False
    :param random_state: when `shuffle` is true, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        otherwise, leave `random_state` as `none`.
        pass an int for reproducible output across multiple function calls.
    :type random_state:int or RandomState instance, default=None
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    # Implementation based on this kaggle kernel:
    # https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def _iter_test_indices(self, X, y, groups):
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


def get_stratified_sampler(
    df_h: pd.DataFrame, index_level: int = 0, split: float = 0.8
):
    """generates sampler indicies for torch DataLoader object that are
    stratified by a given index set (ie a column in a dataframe
    cooresponding to patient identifiers)

    :param df_h: a hierarchical/multi-indexed dataframe
    :type df_h: pd.DataFrame
    :param index_level: the index in the heirarchical dataframe that is used to stratify
    :type indicies: int
    :param split: the train/val split
    :type split: float, optional

    :return: a tuple of indices that coorespond to training and validation samplers
    :rtype: Tuple(List, List)
    """
    # get a non-heirarchical/non-multiindexed version of the datafame
    df = df_h.reset_index()

    # get the index we want to stratify our dataset with
    index_set = np.array(df_h.index.levels[index_level])

    # randomly split index
    train_indices = np.random.choice(
        index_set, size=int(len(index_set) * split), replace=False
    )
    val_indices = np.setdiff1d(index_set, train_indices)

    # pull out stratified indices from a flat dataframe
    train_indices = df[
        np.in1d(df_h.index.get_level_values(index_level), train_indices)
    ].index
    val_indices = df[
        np.in1d(df_h.index.get_level_values(index_level), val_indices)
    ].index

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return (train_sampler, val_sampler)


def get_group_stratified_sampler(
    df_nh: pd.DataFrame,
    classes: pd.DataFrame,
    groups: pd.DataFrame,
    split: float = 0.2,
):
    """generates sampler indicies for torch DataLoader object that are
    stratified by a given group set (ie a column in a dataframe
    cooresponding to patient identifiers), and balanced between target
    labels

    :param df_nh: a non-hierarchical/non-multi-indexed/flat dataframe
    :type df: pd.DataFrame
    :param classes: the target classes for each sample to balance across
        training and validation splits
    :type classes: a single-column pd.DataFrame
    :param groups: the group used to stratify the data (ie patient ids)
    :type groups: a single-column pd.DataFrame
    :param split: the train/val split. must yield a rational number when divided
        by zero in order to ensure proper balancing and stratification
    :type split: float, optional
    :return: a tuple of indices that coorespond to training and validation samplers
    :rtype: Tuple(List, List)
    """
    #
    K = 1 / split
    if K % 1 != 0:
        print(K)
        print("Invalid test/train ratio")
        quit()
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


if __name__ == "__main__":

    main()

    pass
