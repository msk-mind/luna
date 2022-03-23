import pytest

from luna.pathology.analysis.ml import BaseTorchTileDataset, BaseTorchTileClassifier

import torch

from torch import nn

import numpy as np

# we are using sample PIL data
test_data = "pyluna-pathology/tests/luna/pathology/cli/testdata/data/save_tiles/123/123.tiles.csv"


# We need to impliment these
def test_ds_not_implimented():
    with pytest.raises(NotImplementedError):
        tile_dataset = BaseTorchTileDataset(tile_path=test_data)
        tile_dataset[0]


def test_clf_not_implimented():
    with pytest.raises(NotImplementedError):
        tile_classifier = BaseTorchTileClassifier()
        tile_classifier()


# Implimentation
class TileDataset(BaseTorchTileDataset):
    def setup(self):
        pass

    def preprocess(self, input_tile):
        return torch.tensor(np.array(input_tile)).flatten()


class MyClassifier(BaseTorchTileClassifier):
    def setup(self, num_classes):
        self.clf = nn.Linear(20, 10)
        self.sm = nn.Softmax(dim=1)

    def predict(self, input_tiles):
        return self.sm(self.clf(input_tiles))


# Test clf init
def test_init_clf():
    tile_classifier = MyClassifier(num_classes=4)
    out = tile_classifier(range(8), torch.rand(8, 20))
    assert len(out) == 8


# We need tiles for TileDataset
def test_wrong_args_ds():
    with pytest.raises(RuntimeError):
        TileDataset()


def test_init_ds_no_labels():
    tile_dataset = TileDataset(tile_path=test_data)
    assert len(tile_dataset[0]) == 2
    assert tile_dataset[0][1].shape == torch.Size([196608])
