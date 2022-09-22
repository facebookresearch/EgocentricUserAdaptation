"""
Workflow:
- Stored in WandB, copy-paste these:
# Check: TRANSFER_MATRIX/avg_AG/pred_action_batch/loss
# Check: TRANSFER_MATRIX/x_labels/avg_AG/pred_action_batch/loss/stream_users
# Check: TRANSFER_MATRIX/y_labels/avg_AG/pred_action_batch/loss/model_users


"""

# From: https://github.com/ContinualAI/avalanche/blob/b9f4514febad5fc22c2f719ec95b835b7f3476bf/avalanche/evaluation/metric_utils.py
from typing import Dict, Union, Iterable, Sequence, Tuple, TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_testing():
    matrix = [[-178.9048829317093, -155.3030999183655, -126.6202702999115, -135.898851108551, -128.5629695177078,
               -132.07769751548767, -100.8395688533783, -107.73759479522704, -115.61646418571472, -189.6961982011795],
              [-124.15000393390656, -99.23829202651976, -61.39166693687439, -128.95961923599242, -121.86845734119416,
               -91.71943640708923, -96.65158705711364, -115.63561420440674, -122.6344642162323, -126.8311171770096],
              [-169.21303265094758, -174.37426767349243, -107.34935812950134, -119.55384683609007, -150.44728729724883,
               -125.01250739097595, -131.6889371395111, -145.83118457794188, -149.94518580436707, -200.61360695362092],
              [-73.45332119464874, -88.19304971694946, -73.80433669090272, -73.71982698440551, -87.77376778125763,
               -85.23622794151306, -65.74598670005798, -66.9143102645874, -68.89353823661804, -92.78393318653109],
              [-68.57781689167022, -101.24137201309205, -78.72295432090759, -65.21985540390014, -68.86807129383087,
               -53.818164587020874, -49.70685839653015, -60.51835689544678, -54.72790675163269, -68.60921967029572],
              [-86.40143978595734, -104.62066888809204, -95.54140906333924, -117.68047189712524, -105.19867432117462,
               -73.71004347801208, -82.31068930625915, -98.29430255889892, -65.97422709465027, -143.99837868213655],
              [-149.67454845905303, -154.1528214454651, -96.17870230674744, -120.56947107315064, -155.53024475574497,
               -142.64678015708924, -95.4229413509369, -123.73767261505128, -133.9245032787323, -120.90486824512482],
              [-77.16889584064484, -89.81103601455689, -82.86612143516541, -72.84639673233032, -63.67603256702423,
               -72.19986090660095, -56.171891164779666, -68.87763614654541, -59.91962008476257, -110.68006660938264],
              [-153.17954723834993, -147.35554513931274, -108.74639105796814, -148.60546350479126, -121.63964188098907,
               -126.72505927085876, -110.77825293540954, -106.47584705352784, -103.10579142570496, -169.29172432422638],
              [-76.54474728107452, -73.86715745925903, -75.87611632347107, -69.18080339431762, -79.74386780261993,
               -82.85971655845643, -57.28234705924988, -62.083861351013184, -77.52493243217468, -48.42167046070099]]
    # matrix = [[-4.319421209013213,null,null,null,null,null,null,null,null,null],[null,-21.00615071993479,null,null,null,null,null,null,null,null],[null,null,1.7079791380011518,null,null,null,null,null,null,null],[null,null,null,-9.572440371261193,null,null,null,null,null,null],[null,null,null,null,-5.487690441165265,null,null,null,null,null],[null,null,null,null,null,-0.09211561922387128,null,null,null,null],[null,null,null,null,null,null,0.06582317764268202,null,null,null],[null,null,null,null,null,null,null,2.0436283127577215,null,null],[null,null,null,null,null,null,null,null,-15.469999925127638,null],[null,null,null,null,null,null,null,null,null,-2.742694706144467]]
    x_labels = ["104", "108", "24", "265", "27", "29", "30", "324", "421", "68"]
    y_labels = ["104", "108", "24", "265", "27", "29", "30", "324", "421", "68"]

    input_matrix = np.array(matrix, dtype=float)  # Square matrix
    assert len(input_matrix) == len(input_matrix[0]), "Must be square"

    fig = cm_image_creator(
        cm=input_matrix,
        display_labels_x=x_labels,
        xlabel="User Stream",
        display_labels_y=y_labels,
        ylabel="User Model",
        cmap="gist_heat",
        include_values=False,
        # values_format
    )

    plt.show()


def cm_image_creator(
        cm: np.ndarray,
        display_labels_x: Sequence = None,
        display_labels_y: Sequence = None,
        ylabel="True label",
        xlabel="Predicted label",
        include_values=False,
        xticks_rotation=0,
        yticks_rotation=0,
        values_format=None,
        cmap="viridis",
        image_title="",
):
    """
    The default Confusion Matrix image creator.
    Code adapted from
    `Scikit learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html>`_ # noqa
    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `False`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        float point value. Defaults to 0.
    :param yticks_rotation: Rotation of ytick labels. Valid values are
        float point value. Defaults to 0.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.
    """

    fig, ax = plt.subplots()

    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

    if display_labels_x is None:
        display_labels_x = np.arange(n_classes)

    if display_labels_y is None:
        display_labels_y = np.arange(n_classes)
    fig.colorbar(im_, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels_x,
        yticklabels=display_labels_y,
        ylabel=ylabel,
        xlabel=xlabel,
    )

    if image_title != "":
        ax.set_title(image_title)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plot_testing()
