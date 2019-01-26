import random
import copy
from itertools import product
import warnings
from abc import ABC, abstractmethod
import numpy as np


class OutOfBoardError(IndexError):
    pass


class AbstractPoint(ABC):

    @abstractmethod
    def get_coord(self) -> tuple:
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Point(AbstractPoint):
    """ pointer to coordinates on the board"""

    def __init__(self, row, col):
        self.__row = row
        self.__col = col

    def get_coord(self):
        return (self.__row, self.__col)

    def __add__(self, other):
        row1, col1 = self.get_coord()
        row2, col2 = other.get_coord()
        return Point(row1 + row2, col1 + col2)

    def __eq__(self, other):
        return self.get_coord() == other.get_coord()

    def __hash__(self):
        return hash(self.get_coord())


class Cell(Point):
    def __init__(self, shape, row, col):
        self.__shape = shape
        super().__init__(row, col)

    @property
    def shape(self):
        return self.__shape

    @property
    def point(self):
        return Point(*self.get_coord())

    def __eq__(self, other):
        eq_shape = self.shape == other.shape
        eq_points = super().__eq__(other)
        return (eq_shape and eq_points)

    def __hash__(self):
        return hash((self.shape, self.get_coord()))


class AbstractBoard(ABC):

    @property
    @abstractmethod
    def board(self):
        pass

    @property
    @abstractmethod
    def board_size(self):
        pass

    @property
    @abstractmethod
    def n_shapes(self):
        pass

    @abstractmethod
    def swap(self, point1: Point, point2: Point):
        pass

    @abstractmethod
    def set_board(self, board: np.ndarray):
        pass

    @abstractmethod
    def move(self, point: Point, direction: Point):
        pass

    @abstractmethod
    def shuffle(self, random_state=None):
        pass

    @abstractmethod
    def get_shape(self, point: Point):
        pass

    @abstractmethod
    def delete(self, points):
        pass

    @abstractmethod
    def get_line(self, ind):
        pass

    @abstractmethod
    def put_line(self, ind, line):
        pass

    @abstractmethod
    def put_mask(self, mask, shapes):
        pass


class Board(AbstractBoard):
    """ board for match3 game"""

    def __init__(self, rows, columns, n_shapes):
        self.__rows = rows
        self.__columns = columns
        self.__n_shapes = n_shapes
        self.__board = None  # np.ndarray

    def __getitem__(self, indx: Point):
        self.__check_board()
        self.__validate_points(indx)
        if isinstance(indx, Point):
            return self.board.__getitem__(indx.get_coord())
        else:
            raise ValueError('Only Point class supported for getting shapes')

    def __setitem__(self, value, indx: Point):
        self.__check_board()
        self.__validate_points(indx)
        if isinstance(indx, Point):
            self.__board.itemset(indx.get_coord(), value)
        else:
            raise ValueError('Only Point class supported for setting shapes')

    def __str__(self):
        return self.board.board

    @property
    def board(self):
        self.__check_board()
        return self.__board

    @property
    def board_size(self):
        rows, cols = None, None
        if self.__is_board_exist():
            rows, cols = self.board.shape
        else:
            rows, cols = self.__rows, self.__columns
        return rows, cols

    def set_board(self, board: np.ndarray):
        self.__validate_board(board)
        self.__board = board.astype(float)

    def shuffle(self, random_state=None):
        np.random.shuffle(self.__board)
        return self

    def __check_board(self):
        if not self.__is_board_exist():
            raise ValueError('Board is not created')

    @property
    def n_shapes(self):
        return self.__n_shapes

    def swap(self, point1: Point, point2: Point):
        point1_shape = self.get_shape(point1)
        point2_shape = self.get_shape(point2)
        self.put_shape(point2, point1_shape)
        self.put_shape(point1, point2_shape)

    def put_shape(self, shape, point: Point):
        self[point] = shape

    def move(self, point: Point, direction: Point):
        new_point = point + direction
        self.swap(point, new_point)

    def __is_board_exist(self):
        existance = (self.__board is not None)
        return existance

    def __validate_board(self, board: np.ndarray):
        self.__validate_max_shape(board)
        self.__validate_board_size(board)

    def __validate_board_size(self, board: np.ndarray):
        provided_board_shape = board.shape
        right_board_shape = self.board_size
        correct_shape = (provided_board_shape == right_board_shape)
        if not correct_shape:
            raise ValueError('Incorrect board shape: '
                             f'{provided_board_shape} vs {right_board_shape}')

    def __validate_max_shape(self, board: np.ndarray):
        provided_max_shape = np.nanmax(board)
        if np.isnan(provided_max_shape):
            return

        right_max_shape = self.n_shapes
        if provided_max_shape > right_max_shape:
            raise ValueError('Incorrect shapes of the board: '
                             f'{provided_max_shape} vs {right_max_shape}')

    def get_shape(self, point: Point):
        return self[point]

    def __validate_points(self, *args):
        for point in args:
            is_valid = self.__is_valid_point(point)
            if not is_valid:
                raise OutOfBoardError(f'Invalid point: {point.get_coord()}')

    def __is_valid_point(self, point: Point):
        row, col = point.get_coord()
        board_rows, board_cols = self.board_size
        correct_row = ((row + 1) <= board_rows) and (row >= 0)
        correct_col = ((col + 1) <= board_cols) and (col >= 0)
        return correct_row and correct_col

    def delete(self, points: set):
        coords = tuple(np.array([i.get_coord() for i in points]).T.tolist())
        self.__board[coords] = np.nan
        return self

    def get_line(self, ind, axis=1):
        return np.take(self.board, ind, axis=axis)

    def put_line(self, ind, line: list):
        # TODO: create board with putting lines
        # on arbitrary axis
        self.__validate_max_shape(line)
        self.__board[:, ind] = line
        return self

    def put_mask(self, mask, shapes):
        self.__validate_max_shape(shapes)
        self.__board[mask] = shapes
        return self


class RandomBoard(Board):

    def set_random_board(self, random_state=None):
        board_size = self.board_size

        np.random.seed(random_state)
        board = np.random.randint(
            low=0,
            high=self.n_shapes,
            size=board_size)
        self.set_board(board)
        return self


class CustomBoard(Board):

    def __init__(self, board: np.ndarray, n_shapes: int):
        columns, rows = board.shape
        super().__init__(columns, rows, n_shapes)
        self.set_board(board)


class AbstractSearcher(ABC):
    def __init__(self, board_ndim):
        self.__directions = self.__get_directions(board_ndim)

    def __get_directions(self, board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    @property
    def directions(self):
        return self.__directions

    def points_generator(self, board: Board):
        rows, cols = board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def axis_directions_gen(self):
        for axis_dirs in self.directions:
            yield axis_dirs

    def directions_gen(self):
        for axis_dirs in self.directions:
            for direction in axis_dirs:
                yield direction


class AbstractMatchesSearcher(ABC):

    @abstractmethod
    def scan_board_for_matches(self, board: Board):
        pass


class MatchesSearcher(AbstractSearcher):

    def __init__(self, length, board_ndim):
        self.__length = length
        super().__init__(board_ndim)

    def scan_board_for_matches(self, board: Board):
        matches = set()
        for point in self.points_generator(board):
            to_del = self.__get_match3_for_point(board, point)
            if to_del:
                matches.update(to_del)
        return matches

    def __get_match3_for_point(self, board: Board, point: Point):
        shape = board.get_shape(point)
        match3_list = []
        for neighbours in self.__generator_neighbours(board, point):
            filtered = self.__filter_cells_by_shape(shape, neighbours)
            if len(filtered) == (self.__length - 1):
                match3_list.extend(filtered)

        if len(match3_list) > 0:
            match3_list.append(Cell(shape, *point.get_coord()))

        return match3_list

    def __generator_neighbours(self, board: Board, point: Point):
        for axis_dirs in self.directions:
            new_points = [point + Point(*dir_) for dir_ in axis_dirs]
            try:
                yield [Cell(board.get_shape(new_p), *new_p.get_coord())
                       for new_p in new_points]
            except OutOfBoardError:
                continue
            finally:
                yield []

    @staticmethod
    def __filter_cells_by_shape(shape, *args):
        return list(filter(lambda x: x.shape == shape, *args))


class AbstractMovesSearcher(ABC):

    @abstractmethod
    def search_moves(self, board: Board):
        pass


class MovesSearcher(AbstractMovesSearcher, MatchesSearcher):

    def search_moves(self, board: Board):
        possible_moves = set()
        for point in self.points_generator(board):
            possible_moves_for_point = self.__search_moves_for_point(
                board, point)
            possible_moves.update(possible_moves_for_point)
        return possible_moves

    def __search_moves_for_point(self, board: Board, point: Point):
        # contain tuples of point and direction
        possible_moves = set()
        for direction in self.directions_gen():
            try:
                board.move(point, Point(*direction))
                matches = self.scan_board_for_matches(board)
                # inverse move
                board.move(point, Point(*direction))
            except OutOfBoardError:
                continue
            if len(matches) > 0:
                possible_moves.add((point, tuple(direction)))
        return possible_moves


class AbstractFiller(ABC):

    @abstractmethod
    def move_and_fill(self):
        pass


class Filler(AbstractFiller):

    def __init__(self, random_state=None):
        self.__random_state = random_state

    def move_and_fill(self, board: Board):
        self.__move_nans(board)
        self.__fill(board)

    def __move_nans(self, board: Board):
        _, cols = board.board_size
        for col_ind in range(cols):
            line = board.get_line(col_ind)
            if np.any(np.isnan(line)):
                new_line = self.__move_line(line)
                board.put_line(col_ind, new_line)
            else:
                continue

    def __move_line(self, line):
        is_nan_array = np.isnan(line)
        num_of_nan = is_nan_array.sum()
        argsort_line = np.zeros_like(line)
        argsort_line[is_nan_array] = np.inf
        argsort_line[~is_nan_array] = np.arange(
            len(argsort_line) - num_of_nan)[::-1]
        argsort_inds = np.argsort(argsort_line)[::-1]
        return line[argsort_inds]

    def __fill(self, board):
        is_nan_mask = np.isnan(board.board)
        num_of_nans = is_nan_mask.sum()

        np.random.seed(self.__random_state)
        new_shapes = np.random.randint(
            low=0, high=board.n_shapes, size=num_of_nans)
        board.put_mask(is_nan_mask, new_shapes)


class AbstractGame(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def move(self):
        pass


class Game(AbstractGame):

    def __init__(self, rows, columns, n_shapes, length, random_state=None):
        self.board = Board(
            rows=rows,
            columns=columns,
            n_shapes=n_shapes)
        self.__random_state = random_state
        self.__mtch_searcher = MatchesSearcher(length=length, board_ndim=2)
        self.__mv_searcher = MovesSearcher(length=length, board_ndim=2)
        self.__filler = Filler(random_state)

    def play(self, board: np.ndarray):
        self.start(board)
        while True:
            try:
                print(self.board.board)
                input_str = input()
                coords = input_str.split(', ')
                a, b, d0, d1 = [int(i) for i in coords]
                self.move(Point(a, b), Point(d0, d1))
            except KeyboardInterrupt:
                break

    def start(self, board: np.ndarray):
        self.board.set_board(board)
        self.__operate_untill_possible_moves()
        return self

    def move(self, point: Point, direction: Point):
        matches = self.__check_matches(
            point, direction)
        if len(matches) > 0:
            self.board.move(point, direction)
            self.board.delete(matches)
            self.__filler.move_and_fill(self.board)

            self.__operate_untill_possible_moves()
        return len(matches)

    def __check_matches(self, point: Point, direction: Point):
        tmp_board = self.__get_copy_of_board()
        tmp_board.move(point, direction)
        matches = self.__mtch_searcher.scan_board_for_matches(tmp_board)
        return matches

    def __get_copy_of_board(self):
        return copy.deepcopy(self.board)

    def __operate_untill_possible_moves(self):
        """
        scan board, then delete matches, move nans, fill
        repeat untill no matches and appear possible moves
        """
        self.__scan_del_mvnans_fill_untill()
        self.__shuffle_untill_possible()
        return self

    def __get_matches(self):
        return self.__mtch_searcher.scan_board_for_matches(self.board)

    def __get_possible_moves(self):
        return self.__mv_searcher.search_moves(self.board)

    def __scan_del_mvnans_fill_untill(self):
        matches = self.__get_matches()
        while len(matches) > 0:
            self.board.delete(matches)
            self.__filler.move_and_fill(self.board)
            matches = self.__get_matches()
        return self

    def __shuffle_untill_possible(self):
        possible_moves = self.__get_possible_moves()
        while len(possible_moves) == 0:
            self.board.shuffle(self.__random_state)
            self.__scan_del_mvnans_fill_untill()
            possible_moves = self.__get_possible_moves()
        return self


class RandomGame(Game):

    def start(self, *args, **kwargs):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board()
        super().start(tmp_board.board)


def main():
    import warnings
    # warnings.simplefilter("error", "RuntimeWarning")
    game = RandomGame(3, 3, length=3, n_shapes=4, random_state=1)
    game.play(None)


if __name__ == '__main__':
    main()
