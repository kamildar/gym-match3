import unittest
import numpy as np
from gym_match3.envs.game import (Board,
                                  RandomBoard,
                                  CustomBoard,
                                  Point,
                                  Cell,
                                  AbstractSearcher,
                                  MatchesSearcher,
                                  Filler,
                                  Game,
                                  MovesSearcher,
                                  OutOfBoardError,
                                  ImmovableShapeError)
from gym_match3.envs.levels import (Match3Levels,
                                    Level)


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(columns=2, rows=2, n_shapes=3)
        board = np.array([
            [0, 1],
            [2, 0]
        ])
        self.board.set_board(board)

    def test_swap(self):
        """ test swapping two points """
        p1 = Point(0, 0)
        p2 = Point(0, 1)
        self.board.swap(p1, p2)
        correct = np.array([[1, 0], [2, 0]])
        self.assertEqual(
            correct.tolist(), self.board.board.tolist())

    def test_move(self):
        point = Point(1, 1)
        direction = Point(-1, 0)
        correct = np.array([
            [0, 0],
            [2, 1]
        ])
        self.board.move(point, direction)
        self.assertEqual(
            self.board.board.tolist(), correct.tolist()
        )

    def test_getting_validate_board_size(self):
        """ test getting board size """
        self.assertEqual(
            self.board.board_size, (2, 2))

    def test_setting_incorrect_shaped_board(self):
        """ test setting bad board  """
        new_board = np.array([[0, 0], [0, 10]])
        with self.assertRaises(ValueError):
            self.board.set_board(new_board)

    def test_setting_incorrect_sized_board(self):
        new_board = np.array([[0, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            self.board.set_board(new_board)

    def test_get_shape(self):
        """ test getting shape of points """
        points = [
            Point(0, 0),
            Point(1, 1)
        ]
        correct_answers = [
            self.board.board[0, 0],
            self.board.board[1, 1],
        ]
        board_answers = [self.board.get_shape(point)
                         for point in points]
        for ind, point in enumerate(points):
            with self.subTest(point=point.get_coord()):
                self.assertEqual(
                    correct_answers[ind], board_answers[ind]
                )

    def test_get_bad_shape(self):
        """ test getting shapes of incorrect points """
        points = [
            Point(-1, 0),
            Point(-1, -1),
            Point(1, -1),
            Point(10, 0),
            Point(0, 2),
        ]
        for point in points:
            with self.subTest(point=point.get_coord()):
                with self.assertRaises(OutOfBoardError):
                    self.board.get_shape(point)

    def test_delete(self):
        points = {Point(0, 0), Point(1, 1)}
        deleted_board = np.array([
            [np.nan, 1.],
            [2., np.nan]
        ], dtype='float')
        self.board.delete(points)
        self.assertTrue(
            np.allclose(
                self.board.board,
                deleted_board,
                equal_nan=True))

    def test_get_line(self):
        true = np.array([0, 2])
        answer = self.board.get_line(0)
        self.assertEqual(true.tolist(), answer.tolist())

    def test_put_line(self):
        true = self.board.board.copy()
        true[:, 0] = [2, 2]
        self.board.put_line(0, np.array([2, 2]))
        answer = self.board.board
        self.assertEqual(true.tolist(), answer.tolist())

    def test_put_mask(self):
        mask = np.ones_like(self.board.board, dtype=bool)
        mask[1, 1] = False
        true = np.array(
            [[1, 1], [1, 0]])
        self.board.put_mask(mask, [1, 1, 1])
        self.assertEqual(true.tolist(), self.board.board.tolist())


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.point1 = Point(1, 1)
        self.point2 = Point(-1, 0)

    def test_add(self):
        """ test summation of two points """
        true = Point(0, 1)
        result = self.point1 + self.point2
        self.assertEqual(true.get_coord(), result.get_coord())

    def test_mul(self):
        true = Point(-5, -10)
        result = Point(5, 10) * -1
        self.assertEqual(true, result)

    def test_sub(self):
        true = Point(-1, -1)
        result = Point(0, 0) - Point(1, 1)
        self.assertEqual(true, result)


class TestSearcher(unittest.TestCase):
    def setUp(self):
        self.board = RandomBoard(2, 2, 1).set_random_board(random_state=1)
        self.searcher = AbstractSearcher(2)

    def test_points_generator(self):
        answer = {i for i in self.searcher.points_generator(self.board)}
        true = {
            Point(0, 0),
            Point(0, 1),
            Point(1, 0),
            Point(1, 1)
        }
        self.assertEqual(true, answer)

    def test_axis_directions_gen(self):
        answer = sorted(sorted(i)
                        for i in self.searcher.axis_directions_gen())
        true = sorted([
            sorted([[0, 1], [0, -1]]),
            sorted([[1, 0], [-1, 0]])
        ])
        self.assertEqual(answer, true)

    def test_directions_gen(self):
        answer = {tuple(i) for i in self.searcher.directions_gen()}
        true = {
            (0, 1), (1, 0),
            (0, -1), (-1, 0)
        }
        self.assertEqual(answer, true)


class TestMatchesSearcher(unittest.TestCase):
    def setUp(self):
        self.board_3x3_zeros = Board(3, 3, 3)
        self.board_3x3_zeros.set_board(np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 2, 2]
        ]))

        self.board_3x3_seq = Board(3, 3, 9)
        self.board_3x3_seq.set_board(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]))

        self.board_4x4_big_angle = Board(columns=4, rows=4, n_shapes=6)
        self.board_4x4_big_angle.set_board(np.array([
            [0, 1, 2, 3],
            [0, 0, 5, 5],
            [0, 1, 3, 1],
            [0, 0, 0, 1]
        ]))
        self.searcher_three = MatchesSearcher(3, 2)

    @staticmethod
    def __to_coord(args):
        return sorted([i.get_coord() for i in args])

    def test_scan_board(self):
        zeros_answer = self.searcher_three.scan_board_for_matches(
            self.board_3x3_zeros)
        zeros_true = {
            Cell(0, 0, 0),
            Cell(0, 0, 1),
            Cell(0, 0, 2)
        }

        seq_answer = self.searcher_three.scan_board_for_matches(
            self.board_3x3_seq)
        seq_true = set()

        angle_answer = self.searcher_three.scan_board_for_matches(
            self.board_4x4_big_angle)
        angle_true = {
            Cell(0, 0, 0),
            Cell(0, 1, 0),
            Cell(0, 2, 0),
            Cell(0, 3, 0),
            Cell(0, 3, 1),
            Cell(0, 3, 2)
        }

        with self.subTest(object='zeros'):
            self.assertEqual(
                zeros_true,
                zeros_answer)

        with self.subTest(object='sequential'):
            self.assertEqual(
                seq_true,
                seq_answer)

        with self.subTest(object='angle'):
            self.assertEqual(
                angle_true,
                angle_answer)


class TestMovesSearcher(unittest.TestCase):
    def setUp(self):
        board = np.array([
            [1, 2, 0],
            [1, 3, 0],
            [3, 1, 2]
        ])
        self.board = CustomBoard(board=board, n_shapes=4)
        self.moves_searcher = MovesSearcher(
            length=3, board_ndim=2)

    def test_search_moves(self):
        true = {
            (Point(2, 0), (0, 1)),
            (Point(2, 1), (0, -1)),
        }
        answer = self.moves_searcher.search_moves(
            board=self.board, all_moves=True)
        self.assertEqual(true, answer)


class TestFiller(unittest.TestCase):
    def setUp(self):
        board = Board(3, 3, 5)
        board.set_board(np.array([
            [0, 1, 2],
            [np.nan, 0, 1],
            [np.nan, np.nan, np.nan]
        ]))
        self.board = board
        self.filler = Filler(1)

    def test_filler(self):
        self.filler.move_and_fill(board=self.board)
        true = np.array([
            [3., 4., 0.],
            [1., 1., 2.],
            [0., 0., 1.]
        ])
        answer = self.board.board
        self.assertEqual(true.tolist(), answer.tolist())


class TestGame(unittest.TestCase):

    def setUp(self):
        self.game = Game(rows=3, columns=3, n_shapes=(3 * 3),
                         length=3, random_state=1)
        board = np.array([
            [7, 1, 7],
            [1, 4, 6],
            [7, 1, 7]
        ])
        self.game.board.set_board(board)

    def test_bad_swap(self):
        old_board = self.game.board.board.copy()
        answer = self.game.swap(Point(0, 0), Point(0, 1))

        with self.subTest(returns_zero=True):
            self.assertTrue(answer == 0)

        with self.subTest(same_board=True):
            self.assertEqual(old_board.tolist(),
                             self.game.board.board.tolist())

    # def test_simple_way(self):
    #     true = np.array([
    #         [5., 8., 5.],
    #         [6., 5., 5.],
    #         [8., 4., 7.]
    #     ])
    #     self.game.swap(Point(1, 1), Point(1, 0))
    #     answer = self.game.board.board
    #     self.assertEqual(true.tolist(), answer.tolist())


class TestLevels(unittest.TestCase):

    def setUp(self):
        self.level = Level(5, 6, 3, [
            [-1, -1, 0],
            [0, 0, -1],
            [0, 0, -1]
        ])
        self.m3levels = Match3Levels([self.level], -1)

    def test_create_board(self):
        true = [
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1],
            [-1, -1, 0, 0, -1, -1],
            [-1, -1, 0, 0, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ]
        answer = self.m3levels.create_board(self.level)
        answer[answer != -1] = 0
        np.testing.assert_array_equal(true, answer)


class TestBoardImmove(TestBoard):
    def setUp(self):
        self.board = Board(
            columns=2, rows=2, n_shapes=3, immovable_shape=-1)
        board = np.array([
            [0, 1],
            [2, 0]
        ])
        self.board.set_board(board)

        self.board_wimove = Board(
            columns=2, rows=2, n_shapes=3, immovable_shape=-1)
        self.board_wimove.set_board(np.array([
            [-1, 0],
            [0, -1]
        ]))

    def test_availability(self):
        with self.assertRaises(ImmovableShapeError):
            self.board_wimove.move(Point(0, 0), Point(0, 1))

        with self.assertRaises(ImmovableShapeError):
            self.board_wimove.swap(Point(0, 0), Point(1, 1))

        with self.assertRaises(ImmovableShapeError):
            self.board_wimove.delete({Point(0, 0)})

    def test_putting(self):
        with self.assertRaises(ImmovableShapeError):
            mask = np.array([
                [True, False],
                [False, False]])
            self.board_wimove.put_mask(mask, [1])

        with self.assertRaises(ImmovableShapeError):
            ind = 0
            self.board_wimove.put_line(ind, np.array([1, 1]))

    def test_shuffle(self):
        self.board_wimove.shuffle()
        true = np.array([
            [True, False],
            [False, True]
        ])
        answer = self.board_wimove.board == -1
        np.testing.assert_array_equal(true, answer)


class TestSearcherImmove(TestSearcher):
    def setUp(self):
        self.board = Board(columns=3, rows=3, n_shapes=10, immovable_shape=-1)
        board = np.array([
            [-1, -1, -1],
            [-1, 0, -1],
            [-1, -1, 0]
        ])
        self.board.set_board(board)
        self.searcher = MatchesSearcher(3, 2)

    def test_points_generator(self):
        answer = {i for i in self.searcher.points_generator(self.board)}
        true = {
            Point(1, 1),
            Point(2, 2)
        }
        self.assertEqual(true, answer)


class TestMovesSearcherImmove(TestMovesSearcher):
    def setUp(self):
        board = np.array([
            [1, 2, 0],
            [1, 3, 0],
            [3, 1, 2]
        ])
        self.board = Board(columns=3, rows=3, n_shapes=4, immovable_shape=-1)
        self.board.set_board(board)

        board_wimmove = np.array([
            [-1, -1, 1],
            [1, 1, -1],
            [2, 1, 1]
        ])
        self.board_wimmove = Board(columns=3, rows=3, n_shapes=4, immovable_shape=-1)
        self.board_wimmove.set_board(board_wimmove)

        self.moves_searcher = MovesSearcher(
            length=3, board_ndim=2)

    def test_search_moves(self):
        true = {
            (Point(1, 0), (1, 0)),
            (Point(2, 0), (-1, 0)),
        }
        answer = self.moves_searcher.search_moves(
            board=self.board_wimmove, all_moves=True)
        self.assertEqual(true, answer)


    def test_big_board(self):
        board = np.array([
            [1, 2, 0],
            [1, 3, 0],
            [3, 1, 2]
        ])
        self.board = Board(columns=3, rows=3, n_shapes=4, immovable_shape=-1)
        self.board.set_board(board)



if __name__ == '__main__':
    unittest.main()


class TestFillerImmove(unittest.TestCase):
    def setUp(self) -> None:
        self.filler = Filler()

    def test_move_line(self):
        cases = [
            ([1, -1, 99, np.nan, np.nan, 1], [np.nan, -1, np.nan, 1, 99, 1]),
            ([1, -1, 99, 2, np.nan, 1], [np.nan, -1, 1, 99, 2, 1]),
            ([1, 1, 1, 1], [1, 1, 1, 1]),
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
            ([-1, -1, 0, np.nan], [-1, -1, np.nan, 0]),
            ([-1, -1, np.nan, np.nan], [-1, -1, np.nan, np.nan]),
            ([-1, -1, 2, 0], [-1, -1, 2, 0]),
        ]

        for line, correct in cases:
            with self.subTest(case=str(line)):
                new_line = self.filler._move_line(np.array(line), -1)
                np.testing.assert_array_equal(new_line, correct)
