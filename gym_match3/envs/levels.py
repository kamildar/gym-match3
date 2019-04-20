from collections import namedtuple
import numpy as np

Level = namedtuple("Level", ["num", "h", "w", "n_shapes", "board"])


class Match3Levels:

    def __init__(self, h, w, immovable_shape=-1):
        self.__h = h
        self.__w = w
        self.__immovable_shape = immovable_shape

    def create_board(self, level: Level):
        empty_board = np.random.randint(0, level.n_shapes, size=(self.__h, self.__w))
        board = self.__put_immovable(empty_board, level)
        return board

    def __put_immovable(self, board, level):
        template = np.array(level.board)
        expanded_template = self.__expand_template(template)
        board[expanded_template == self.__immovable_shape] = -1
        return board

    def __expand_template(self, template):
        template_h, template_w = template.shape
        extra_h, extra_w = self.__calc_extra_dims(template_h, template_w)
        return np.pad(template, [extra_h, extra_w], mode='constant', constant_values=-1)

    def __calc_extra_dims(self, h, w):
        pad_h = self.__calc_padding(h, self.__h)
        pad_w = self.__calc_padding(w, self.__w)
        return pad_h, pad_w

    @staticmethod
    def __calc_padding(size, req_size):
        """
        calculate padding size for dimension
        :param size: int, size of level's dimension
        :param req_size: int, required size of dimension
        :return: tuple of ints with pad width
        """
        assert req_size >= size
        if req_size == size:
            pad = (0, 0)

        else:
            extra = req_size - size
            even = (extra % 2 == 0)

            if even:
                pad = (extra // 2, extra // 2)
            else:
                pad = (extra // 2 + 1, extra // 2)

        return pad


LEVELS = [
    Level(1, 5, 8, 6, [[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       ]),
    Level(2, 9, 9, 6, [[-1, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                       ]),
    Level(3, 7, 7, 6, [[-1, 0, 0, 0, 0, 0, -1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0, -1],
                       ]),
    Level(4, 9, 9, 6, [[-1, -1, 0, 0, 0, 0, 0, 0, -1, -1],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, -1, 0],
                       [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1],
                       ]),
    Level(5, 9, 6, 5, [[0, 0, 0, 0, 0, 0],
                       [0, -1, -1, -1, -1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, -1],
                       [0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, -1],
                       [0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, -1],
                       [0, 0, 0, 0, 0, 0],
                       ])

]
