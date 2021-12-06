"""model.py
generic classes and methods used to train and validate classifier models
"""
import os

from typing import List, Tuple, Dict

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from classifier.analyze import plot_cm, plot_roc, tensorboard_pr_curve


class Classifier(object):
    """Class representation of a neural network classifier. Essentially a wrapper
    for model instantiation, training and validation, with support for performance
    metric tracking.

    :param network: The neural network architecture used to define the model. Can
        be either user defined in a seperate file or a pre-defined torch.model
    :type network: torch.nn.Module
    :param criterion: The objective or loss function used to define the model. Can
        be either user defined or a torch.nn.Module.
    :type criterion: torch.nn.Module
    :param optimizer: The optimizer used to define the model
    :type optimizer: torch.optim
    :param data: Data used for training and validation. Should be a
        dictionary of two elements
    :type dataloader: Dict[str, torch.utils.data.DataLoader]
    :param device: the GPU device number used for training, or CPU. Defaults to
        'gpu:0'
    :type device: str, optional
    :param ckpt: A filepath to a checkpoint to resume training/validation from.
        Defaults to the empty string ""
    :type ckpt: str, optional
    :param output_dir: directory where log files and model checkpoints are saved to. Defaults to
        "./outputs/"
    :type output_dir: str, optional
    """

    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim,
        data: Dict[str, DataLoader],
        label_dict: Dict[str, int],
        device: str = "cuda:0",
        ckpt: str = "",
        output_dir: str = "./outputs/",
    ) -> None:
        """Class constructor method"""

        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.data = data
        self.label_dict = label_dict
        self.device = device
        self.ckpt = ckpt
        self.output_dir = output_dir 

        self.label_list = list(self.label_dict.keys())
        # initilize logging
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        images, labels = iter(self.data["train"]).next()
        self.writer.add_graph(self.network, images)

        # move network to device
        self.network.to(self.device)

        # requires training and validation data to be supplied
        try:
            assert len(self.data) == 2
            assert self.data["train"] is not None
        except AssertionError:
            print(
                "supply a dataloader in data['train']. If not performing validation, set data['val']=None"
            )
        # exit()

        # checkpoint loading and training resumption logic
        if self.ckpt is "":
            self.init_epoch = 0
            self.n_epoch = 0
        else:
            self.load_ckpt(self.ckpt)
            self.init_epoch = self.n_epoch

        pass

    def train(self, n_epoch: int):
        """Trains a model for a single epoch. Wraps boilerplate torch training
        loop, with additional support for logging

        :param n_epoch: The current epoch, passed from the 'outer' loop over
            the total number of epochs used during training.
        :type n_epoch: int
        :return: a tuple containing the training loss and accuracy for use in
            callbacks in the outer traiing loop
        :rtype: Tuple(float, float)
        """

        # explicitly put network into training mode
        self.network.train()
        # initilize running variables
        train_loss = 0.0
        n_correct = 0.0
        total = 0
        self.n_epoch = self.init_epoch + n_epoch

        for epoch_iter, (inputs, labels) in enumerate(self.data["train"]):

            # move data and label to target device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero out parameter gradients
            self.optimizer.zero_grad()

            # compute forward pass
            outputs = self.network(inputs)

            # evaluate loss
            loss = self.criterion(outputs, labels.long())

            # backward pass
            loss.backward()
            self.optimizer.step()

            # collect metrics
            train_loss += loss.item()
            n_correct += int((torch.argmax(outputs, 1) == labels).sum().item())
            total += inputs.size(0)

        # logging performance per epoch
        overall_acc = n_correct / total
        overall_loss = train_loss / len(self.data["train"])

        self.writer.add_scalar("training loss", overall_loss, self.n_epoch)
        self.writer.add_scalar("training accuracy", overall_acc, self.n_epoch)

        return (overall_acc, overall_loss)

    def validate(self, n_epoch: int):
        """Validates a model during training. This essentially wraps around
        boiler plate torch code with additional support for logging and
        performance metric computation

        :param n_epoch: The current epoch from the outer training loop. Not
            neccessary for this method to work, added so that self.train()
            and self.validate() have the same function signature.
        :type n_epoch: int
        :return: a tuple containing the validation loss and validation accuracy
            for use in extneral callbacks
        :rtype: Tuple(float, float)
        """

        # put network in evaluation mode
        self.network.eval()
        # initilize running variables
        val_loss = 0.0
        n_correct = 0.0
        total = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        class_probs = []
        class_labels = []
        self.n_epoch = self.init_epoch + n_epoch

        # freeze gradients
        with torch.no_grad():
            for epoch_iter, (inputs, labels) in enumerate(self.data["val"]):

                # move data to target device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # perform inference
                outputs = self.network(inputs)
                val_preds.extend(torch.argmax(outputs, 1).cpu().tolist())
                val_labels.extend(labels.data.int().cpu().tolist())
                val_probs.extend(F.softmax(outputs, 1).cpu().tolist())
                batch_loss = self.criterion(outputs, labels.long())

                # compute per-class probabilities
                class_probs.append([F.softmax(item, 0) for item in outputs])
                class_labels.append(labels)

                # collect metrics
                val_loss += batch_loss.item()
                n_correct += int(
                    (torch.argmax(outputs, 1) == labels).sum().item()
                )
                total += inputs.size(0)

            # logging performance per epoch
            overall_acc = n_correct / total
            overall_loss = val_loss / len(self.data["val"])
            self.writer.add_scalar(
                "validation accuracy", overall_acc, self.n_epoch
            )
            self.writer.add_scalar(
                "validation loss", overall_loss, self.n_epoch
            )

            # log custom figures to tensorboard
            # plot confusion matrix
            self.writer.add_figure(
                "validation confusion matrix",
                plot_cm(val_labels, val_preds, self.label_list),
                global_step=self.n_epoch,
            )
            # plot precision-recall curves
            class_probs = torch.cat(
                [torch.stack(batch) for batch in class_probs]
            )
            class_labels = torch.cat(class_labels)

            for label in range(len(self.label_list)):
                self.writer.add_pr_curve(
                    self.label_list[label],
                    class_labels == label,
                    class_probs[:, label],
                    global_step=self.n_epoch,
                )

            # if the classifier is binary, plot ROC curves
            if len(self.label_dict) == 2:
                val_probs = [val_prob[1] for val_prob in val_probs]
                self.writer.add_figure(
                    "receiver operating characteristic",
                    plot_roc(val_preds, val_labels, val_probs),
                    global_step=self.n_epoch,
                )

        return (overall_acc, overall_loss)

    def save_ckpt(self, dest_path: str = "./outputs/ckpts") -> None:
        """Utility function used to save model checkpoint. Automatically
        makes destination directory if it doesn't exit already

        :param dest_path: Filepath to directory to save model checkpoints to.
            Defaults to './outputs/ckpts'.
        :type dest_path: str, optional
        """

        model_states = {"net": self.network.state_dict()}
        optim_states = {"optim": self.optimizer.state_dict()}

        states = {
            "epoch": self.n_epoch,
            "model_states": model_states,
            "optim_states": optim_states,
        }
        os.makedirs(dest_path, exist_ok=True)
        file_path = os.path.join(dest_path, "ckpt_{}.pt".format(self.n_epoch))

        with open(file_path, mode="wb+") as f:
            torch.save(states, f)

        print("saved model checkpoint")

        pass

    def load_ckpt(self, file_path: str) -> None:
        """Utility function used to load model checkpoints.

        :param file_path: filename of saved checkpoint
        :type file_path: str
        """

        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            # re-instantiate persistent variables
            self.n_epoch = checkpoint["epoch"]
            self.network.load_state_dict(checkpoint["model_states"]["net"])
            self.optimizer.load_state_dict(checkpoint["optim_states"]["optim"])

            print("-> loaded ckpt {}".format(file_path))
        else:
            print("-> no ckpt at '{}'".format(file_path))

        pass


if __name__ == "__main__":

    pass
