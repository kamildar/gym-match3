import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_match3.envs.game import RandomGame, Point, OutOfBoardError
from gym_match3.envs.configurations import GameConfiguration
from gym_match3.envs.renderer import Renderer

from itertools import product

BOARD_NDIM = 2


class Match3Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.h = GameConfiguration.NUM_ROWS
        self.w = GameConfiguration.NUM_COLS
        self.n_shapes = GameConfiguration.NUM_SHAPES
        self.rollout_len = GameConfiguration.MOVES_PER_SESSION
        self.__episode_counter = 0

        self.__game = RandomGame(
            rows=GameConfiguration.NUM_ROWS,
            columns=GameConfiguration.NUM_COLS,
            n_shapes=GameConfiguration.NUM_SHAPES,
            length=GameConfiguration.MATCH_LENGTH)
        self.reset()
        self.renderer = Renderer(GameConfiguration.NUM_SHAPES)

        self.observation_space = spaces.Box(
            low=0,
            high=GameConfiguration.NUM_SHAPES,
            shape=self.__game.board.board_size,
            dtype=int)
        self.__match3_actions = self.__get_availiable_actions()
        self.action_space = spaces.Discrete(
            len(self.__match3_actions))

    def __get_directions(self, board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions

    def __points_generator(self):
        rows, cols = self.__game.board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            yield point

    def __get_availiable_actions(self):
        actions = set()
        directions = self.__get_directions(board_ndim=BOARD_NDIM)
        for point in self.__points_generator():
            for axis_dirs in directions:
                for dir_ in axis_dirs:
                    dir_p = Point(*dir_)
                    new_point = point + dir_p
                    try:
                        _ = self.__game.board[new_point]
                        actions.add(frozenset((point, new_point)))
                    except OutOfBoardError:
                        continue
        return list(actions)

    def __get_action(self, ind):
        return self.__match3_actions[ind]

    def step(self, action):
        # make action
        m3_action = self.__get_action(action)
        reward = self.__game.swap(*m3_action)

        ob = self.__get_board()

        self.__episode_counter += 1
        if self.__episode_counter >= self.rollout_len:
            episode_over = True
            self.__episode_counter = 0
            self.reset()
        else:
            episode_over = False

        return ob, reward, episode_over, {}

    def reset(self):
        self.__game.start(None)
        return self.__get_board()

    def __get_board(self):
        return self.__game.board.board

    def render(self, mode='human', close=False):
        self.renderer.render_board(self.__get_board())
