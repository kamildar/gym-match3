import copy
from itertools import product
from functools import wraps
from abc import ABC, abstractmethod
import numpy as np


class OutOfBoardError(IndexError):
    pass


class ImmovableShapeError(ValueError):
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
        return self.__row, self.__col

    def __add__(self, other):
        row1, col1 = self.get_coord()
        row2, col2 = other.get_coord()
        return Point(row1 + row2, col1 + col2)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, constant):
        row, col = self.get_coord()
        return Point(row * constant, col * constant)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return -1 * other + self

    def __eq__(self, other):
        return self.get_coord() == other.get_coord()

    def __hash__(self):
        return hash(self.get_coord())

    def __str__(self):
        return str(self.get_coord())

    def __repr__(self):
        return self.__str__()


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
        return eq_shape and eq_points

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


def check_availability_dec(func):
    @wraps(func)
    def wrapped(self, *args):
        self._check_availability(*args)
        return func(self, *args)

    return wrapped


class Board(AbstractBoard):
    """ board for match3 game"""

    def __init__(self, rows, columns, n_shapes, immovable_shape=-1):
        self.__rows = rows
        self.__columns = columns
        self.__n_shapes = n_shapes
        self.__immovable_shape = immovable_shape
        self.__board = None  # np.ndarray

        if 0 <= immovable_shape < n_shapes:
            raise ValueError('Immovable shape has to be less or greater than n_shapes')

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
    def immovable_shape(self):
        return self.__immovable_shape

    @property
    def board(self):
        self.__check_board()
        return self.__board

    @property
    def board_size(self):
        if self.__is_board_exist():
            rows, cols = self.board.shape
        else:
            rows, cols = self.__rows, self.__columns
        return rows, cols

    def set_board(self, board: np.ndarray):
        self.__validate_board(board)
        self.__board = board.astype(float)

    def shuffle(self, random_state=None):
        moveable_mask = self.board != self.immovable_shape
        board_ravel = self.board[moveable_mask]
        np.random.seed(random_state)
        np.random.shuffle(board_ravel)
        self.put_mask(moveable_mask, board_ravel)

    def __check_board(self):
        if not self.__is_board_exist():
            raise ValueError('Board is not created')

    @property
    def n_shapes(self):
        return self.__n_shapes

    @check_availability_dec
    def swap(self, point1: Point, point2: Point):
        point1_shape = self.get_shape(point1)
        point2_shape = self.get_shape(point2)
        self.put_shape(point2, point1_shape)
        self.put_shape(point1, point2_shape)

    def put_shape(self, shape, point: Point):
        self[point] = shape

    def move(self, point: Point, direction: Point):
        self._check_availability(point)
        new_point = point + direction
        self.swap(point, new_point)

    def __is_board_exist(self):
        existence = (self.__board is not None)
        return existence

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
        if np.all(np.isnan(board)):
            return
        provided_max_shape = np.nanmax(board)

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

    def _check_availability(self, *args):
        for p in args:
            shape = self.get_shape(p)
            if shape == self.immovable_shape:
                raise ImmovableShapeError

    def delete(self, points: set):
        self._check_availability(*points)
        coordinates = tuple(np.array([i.get_coord() for i in points]).T.tolist())
        self.__board[coordinates] = np.nan
        return self

    def get_line(self, ind, axis=1):
        return np.take(self.board, ind, axis=axis)

    def put_line(self, ind, line: np.ndarray):
        # TODO: create board with putting lines on arbitrary axis
        self.__validate_line(ind, line)
        self.__validate_max_shape(line)
        self.__board[:, ind] = line
        return self

    def put_mask(self, mask, shapes):
        self.__validate_mask(mask)
        self.__validate_max_shape(shapes)
        self.__board[mask] = shapes
        return self

    def __validate_mask(self, mask):
        if np.any(self.board[mask] == self.immovable_shape):
            raise ImmovableShapeError

    def __validate_line(self, ind, line):
        immove_mask = self.board[:, ind] == self.immovable_shape
        new_immove_mask = np.array(line) == self.immovable_shape
        if not np.array_equal(immove_mask, new_immove_mask):
            raise ImmovableShapeError


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

    @staticmethod
    def __get_directions(board_ndim):
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

    @staticmethod
    def points_generator(board: Board):
        rows, cols = board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            if board[point] == board.immovable_shape:
                continue
            else:
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

    def search_moves(self, board: Board, all_moves=False):
        possible_moves = set()
        for point in self.points_generator(board):
            possible_moves_for_point = self.__search_moves_for_point(
                board, point)
            possible_moves.update(possible_moves_for_point)
            if len(possible_moves_for_point) > 0 and not all_moves:
                break
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
            except (OutOfBoardError, ImmovableShapeError):
                continue
            if len(matches) > 0:
                possible_moves.add((point, tuple(direction)))
        return possible_moves


class AbstractFiller(ABC):

    @abstractmethod
    def move_and_fill(self, board):
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
                new_line = self._move_line(line, board.immovable_shape)
                board.put_line(col_ind, new_line)
            else:
                continue

    @staticmethod
    def _move_line(line, immovable_shape):
        new_line = np.zeros_like(line)
        num_of_nans = np.isnan(line).sum()
        immov_mask = (line == immovable_shape)
        nans_mask = np.isnan(line)
        new_line[immov_mask] = immovable_shape

        num_putted = 0
        for ind, shape in enumerate(new_line):

            if shape != immovable_shape and num_putted < num_of_nans:
                new_line[ind] = np.nan
                num_putted += 1
                if num_putted == num_of_nans:
                    break

        spec_mask = nans_mask | immov_mask
        regular_values = line[~spec_mask]
        new_line[(new_line == 0)] = regular_values
        return new_line

    def __fill(self, board):
        is_nan_mask = np.isnan(board.board)
        num_of_nans = is_nan_mask.sum()

        np.random.seed(self.__random_state)
        new_shapes = np.random.randint(
            low=0, high=board.n_shapes, size=num_of_nans)
        board.put_mask(is_nan_mask, new_shapes)


class AbstractGame(ABC):

    @abstractmethod
    def start(self, board):
        pass

    @abstractmethod
    def swap(self, point, point2):
        pass


class Game(AbstractGame):

    def __init__(self, rows, columns, n_shapes, length,
                 all_moves=False,
                 immovable_shape=-1,
                 random_state=None):
        self.board = Board(
            rows=rows,
            columns=columns,
            n_shapes=n_shapes)
        self.__random_state = random_state
        self.__immovable_shape = immovable_shape
        self.__all_moves = all_moves
        self.__mtch_searcher = MatchesSearcher(length=length, board_ndim=2)
        self.__mv_searcher = MovesSearcher(length=length, board_ndim=2)
        self.__filler = Filler(random_state=random_state)

    def play(self, board: np.ndarray or None):
        self.start(board)
        while True:
            try:
                print(self.board.board)
                input_str = input()
                coords = input_str.split(', ')
                a, b, a1, b1 = [int(i) for i in coords]
                self.swap(Point(a, b), Point(a1, b1))
            except KeyboardInterrupt:
                break

    def start(self, board: np.ndarray or None):
        # TODO: check consistency of movable figures and n_shapes
        if board is None:
            rows, cols = self.board.board_size
            board = RandomBoard(rows, cols, self.board.n_shapes)
            board.set_random_board(random_state=self.__random_state)
            board = board.board
        self.board.set_board(board)
        self.__operate_until_possible_moves()

        return self

    def __start_random(self):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board(random_state=self.__random_state)
        super().start(tmp_board.board)

    def swap(self, point: Point, point2: Point):
        direction = point2 - point
        score = self.__move(point, direction)
        return score

    def __move(self, point: Point, direction: Point):
        score = 0

        matches = self.__check_matches(
            point, direction)
        if len(matches) > 0:
            score += len(matches)

            self.board.move(point, direction)
            self.board.delete(matches)
            self.__filler.move_and_fill(self.board)
            score += self.__operate_until_possible_moves()

        return score

    def __check_matches(self, point: Point, direction: Point):
        tmp_board = self.__get_copy_of_board()
        tmp_board.move(point, direction)
        matches = self.__mtch_searcher.scan_board_for_matches(tmp_board)
        return matches

    def __get_copy_of_board(self):
        return copy.deepcopy(self.board)

    def __operate_until_possible_moves(self):
        """
        scan board, then delete matches, move nans, fill
        repeat until no matches and appear possible moves
        """
        score = self.__scan_del_mvnans_fill_until()
        self.__shuffle_until_possible()
        return score

    def __get_matches(self):
        return self.__mtch_searcher.scan_board_for_matches(self.board)

    def __get_possible_moves(self):
        return self.__mv_searcher.search_moves(
            self.board,
            all_moves=self.__all_moves)

    def __scan_del_mvnans_fill_until(self):
        score = 0
        matches = self.__get_matches()
        score += len(matches)
        while len(matches) > 0:
            self.board.delete(matches)
            self.__filler.move_and_fill(self.board)
            matches = self.__get_matches()
            score += len(matches)
        return score

    def __shuffle_until_possible(self):
        possible_moves = self.__get_possible_moves()
        while len(possible_moves) == 0:
            self.board.shuffle(self.__random_state)
            self.__scan_del_mvnans_fill_until()
            possible_moves = self.__get_possible_moves()
        return self


class RandomGame(Game):

    def start(self, random_state=None, *args, **kwargs):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board(random_state=random_state)
        super().start(tmp_board.board)
