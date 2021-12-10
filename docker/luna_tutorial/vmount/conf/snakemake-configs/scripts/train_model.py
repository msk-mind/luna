# todo: remove libs
import datetime
import sys
import os

from collections import Counter
from typing import List, Optional, Tuple

sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from PIL import Image
from pyarrow import fs
from tensorboard import notebook
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from classifier.model import Classifier
from classifier.data import (
    TileDataset,
    get_stratified_sampler,
    get_group_stratified_sampler,
)
from classifier.utils import set_seed, seed_workers

# constants
torch.set_default_tensor_type(torch.FloatTensor)

# define params
batch_size = 64
validation_split = 0.50
shuffle_dataset = False
random_seed = 42 
lr = 1e-3
epochs = 10
n_workers = 20

# set random seed
set_seed(random_seed)

# set output directory
output_dir = '/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/snakemake/model/outputs'

# load data
label_set = {
    "tumor":0,
    "vessels":1,
    "stroma":2
}

path_to_tile_manifest = '/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/snakemake/tiles/ov_tileset'
df = pq.ParquetDataset(path_to_tile_manifest).read().to_pandas()

df['regional_label'] = df['regional_label'].str.replace('arteries', 'vessels')
df['regional_label'] = df['regional_label'].str.replace('veins', 'vessels')

df['regional_label'] = df['regional_label'].str.replace('lympho_poor_tumor', 'tumor')
df['regional_label'] = df['regional_label'].str.replace('lympho_rich_tumor', 'tumor')

df['regional_label'] = df['regional_label'].str.replace('lympho_rich_stroma', 'stroma')

df = df[df['regional_label'] != 'adipocytes']

dataset_local = TileDataset(df, label_set)

# reset index
df_nh = df.reset_index()
# convert patient_ids to numerical values
# df_nh['patient_id'] = [id[2:] for id in df_nh['patient_id']]
# convert labels to ints
df_nh['regional_label'] = [label_set[label] for label in df_nh['regional_label']]

# stratify by patient id while balancing regional_label
train_sampler, val_sampler = get_group_stratified_sampler(
    df_nh, df_nh["regional_label"], df_nh["patient_id"], split=validation_split
)
print(next(iter(train_sampler)))
print(next(iter(val_sampler)))

train_loader = DataLoader(
        dataset_local,
        num_workers=n_workers,
        worker_init_fn=seed_workers,
        shuffle=shuffle_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

validation_loader = DataLoader(
        dataset_local,
        num_workers=n_workers,
        worker_init_fn=seed_workers,
        shuffle=shuffle_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )

data = {"train": train_loader, "val": validation_loader}

# define model
network = models.resnet18(num_classes=len(label_set))
optimizer = optim.Adam(network.parameters(), lr=lr)

# class balanced cross entropy 
train_labels = df_nh["regional_label"][train_sampler]
#class_weights = torch.Tensor([1/count for count in Counter(train_labels).values()])

#criterion = nn.CrossEntropyLoss(weight = class_weights.to(device='cuda'))
criterion = nn.CrossEntropyLoss() 

model = Classifier(
        network,
        criterion,
        optimizer,
        data,
        label_set,
        output_dir=output_dir,
    )

for n_epoch in range(epochs):
    print(n_epoch)
    _ = model.train(n_epoch)

    if n_epoch % 2 == 0:
        print("validating")
        _ = model.validate(n_epoch)
        model.save_ckpt(os.path.join(output_dir, 'ckpts'))

pass