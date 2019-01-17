import gym
from gym import error, spaces, utils
from gym.utils import seeding

from game import RandomGame, Point
from configurations import GameConfiguration


class Match3Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__game = RandomGame(
            rows=GameConfiguration.NUM_ROWS,
            cols=GameConfiguration.NUM_COLS,
            n_shapes=GameConfiguration.NUM_SHAPES,
            length=GameConfiguration.MATCH_LENGTH)

    def step(self, action):
        """
        action is tuple of point and direction
        point is tuple (row_ind, col_ind)
        directions is tuple with only one -1 or 1:
            examples: (-1, 0), (0, 1)
        """
        episode_over = False

        # make action
        reward = self.__game.move(
            Point(action[0], action[1]),
            (action[2], action[3]))

        ob = self.__get_board()
        return ob, reward, episode_over, {}

    def reset(self):
        self.__game.start(None)
        return self.__get_board()

    def __get_board(self):
        return self.__game.board.board

    def render(self, mode='human', close=False):
        return None
