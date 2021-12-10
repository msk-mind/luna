from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.utils.multiclass import unique_labels


def plot_cm(y_true, y_pred, classes):
    """plot confusion matrix to tensorboard
    TODO add type to docstring

    :param y_true: ground truth labels
    :param y_pred: prediction of the data
    :param classes: class labels
    :return: confusion matrix as a matplotlib/sns figure
    :rtype: matplotlib.fig
    """
    u_labels = unique_labels(y_true, y_pred)

    classes = np.array(classes)[u_labels.astype(int)]

    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "{:.0f}%".format(p)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "{:.0f}%".format(p)

    cm = pd.DataFrame(cm_perc, index=classes, columns=classes)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=annot, fmt="", ax=ax, square=True)

    plt.tight_layout()

    return fig


def tensorboard_pr_curve(
    writer, classes, class_index, probs, labels, global_step
):

    tensorboard_truth = labels == class_index
    tensorboard_probs = probs[:, class_index]

    writer.add_pr_curve(
        classes[class_index],
        tensorboard_truth,
        tensorboard_probs,
        global_step=global_step,
    )
    pass


def plot_roc(
    preds: list, labels: list, probs: list, title: Optional[str] = None
):
    """plot ROC curve to tensorboard

    :param preds: list of predictions for a given set of examples
    :type preds: List
    :param labels: list of labels for a given set of examples
    :type labels: List
    :param probs: list of probabilites for a set of examples
    :type probs: List
    :param title: title for the ROC plot
    :type title: str, optional
    :return: ROC plot figure
    :rtype: matplotlib.fig
    """
    matplotlib.rc_file_defaults()
    fp, tp, thresholds = roc_curve(labels, probs)

    roc_auc = auc(fp, tp)
    title = "Receiver Operating Characteristic"
    fig = plt.figure()
    lw = 2
    plt.plot(
        fp,
        tp,
        color="blue",
        lw=lw,
        label="ROC curve (area = {:.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="red", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    return fig


def main():

    pass


if __name__ == "__main__":

    main()

    pass
