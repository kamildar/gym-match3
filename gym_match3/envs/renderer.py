import numpy as np
import matplotlib
from matplotlib import colors
from gym_match3.envs.game import Board

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


class Renderer:

    def __init__(self, n_shapes):
        self.__n_shapes = n_shapes

    def render_board(self, board: Board):
        cmap = self.__get_cmap(board)
        norm = self.__get_norm()

        rows, cols = board.board_size

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(np.arange(0.5, cols + 0.5, step=1), minor=True)
        ax.set_yticks(np.arange(0.5, rows + 0.5, step=1), minor=True)
        plt.xticks(ticks=np.arange(1, cols + 1, step=1), labels=np.arange(0, cols, step=1))
        plt.yticks(ticks=np.arange(1, rows + 1, step=1), labels=np.arange(0, rows, step=1))
        ax.set_yticklabels(labels=np.arange(0, rows, step=1)[::-1])
        ax.xaxis.tick_top()
        plt.grid(which='minor', linewidth=2, c='white')
        plt.imshow(board.board,
                   extent=[0.5, 0.5 + cols, 0.5, 0.5 + rows],
                   cmap=cmap, norm=norm)

    @staticmethod
    def __get_cmap(board: Board):
        cmap = plt.get_cmap('tab20b').colors
        if np.any(board.board == board.immovable_shape):
            cmap = ['black'] + list(cmap)
        return colors.ListedColormap(cmap)

    def __get_norm(self):
        return colors.BoundaryNorm(np.arange(-1, self.__n_shapes), self.__n_shapes + 1)
