import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


class Renderer:

    def __init__(self, n_shapes):
        self.__n_shapes = n_shapes
        self.__norm = self.__get_norm()
        self.__fmt = self.__get_formater()

    def render_board(self, board: np.ndarray):
        rows, cols = board.shape

        heatmap(board, list(range(rows)), list(range(cols)),
                cmap=plt.get_cmap("PiYG", self.__n_shapes),
                cbar_kw=dict(
                    ticks=np.arange(
                        0.5, self.__n_shapes + 0.5),
                    format=self.__fmt),
                norm=self.__norm)

    def __get_norm(self):
        ns = self.__n_shapes
        norm = matplotlib.colors.BoundaryNorm(
            np.linspace(0, ns, ns + 1), ns)
        return norm

    def __get_formater(self):
        fmt = matplotlib.ticker.FuncFormatter(lambda x, _: int(x + 0.5))
        return fmt
