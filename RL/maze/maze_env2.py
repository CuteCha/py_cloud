import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from enum import Enum


class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class Maze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        self.n_rows, self.n_cols = self._maze.shape
        self.rat = rat
        self.target = (self.n_rows - 1, self.n_cols - 1)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if self._maze[self.rat] == 0.0:
            raise Exception("Invalid Rat Location: must sit on a free cell")

        self.mark_rat = 0.2
        self.mark_target = 0.8
        self.mark_visited = 0.6
        self.action = Action

        self.min_reward = -0.5 * self._maze.size
        self.total_reward = 0

        self.maze = None
        self.state = None

        self.visited = set()
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        row, col = rat
        self.maze[row, col] = self.mark_rat
        self.state = (row, col, 'start')
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        n_row, n_col, n_mode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            n_mode = 'blocked'
        elif action in valid_actions:
            n_mode = 'valid'
            if action == self.action.LEFT:
                n_col -= 1
            elif action == self.action.UP:
                n_row -= 1
            if action == self.action.RIGHT:
                n_col += 1
            elif action == self.action.DOWN:
                n_row += 1
        else:  # invalid action, no change in rat position
            n_mode = 'invalid'

        # new state
        self.state = (n_row, n_col, n_mode)

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward

        return self.observe(), reward, self.game_status()

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        if rat_row == self.n_rows - 1 and rat_col == self.n_cols - 1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def draw_env(self):
        canvas = np.copy(self.maze)
        # clear all visual marks
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = self.mark_rat
        return canvas

    def observe(self):
        canvas = self.draw_env()
        env_state = canvas.reshape((1, -1))
        return env_state

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        if rat_row == self.n_rows - 1 and rat_col == self.n_cols - 1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell

        actions = [self.action.LEFT, self.action.UP, self.action.RIGHT, self.action.DOWN]

        if row == 0:
            actions.remove(self.action.UP)
        elif row == self.n_rows - 1:
            actions.remove(self.action.DOWN)

        if col == 0:
            actions.remove(self.action.LEFT)
        elif col == self.n_cols - 1:
            actions.remove(self.action.RIGHT)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(self.action.UP)
        if row < self.n_rows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(self.action.DOWN)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(self.action.LEFT)
        if col < self.n_cols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(self.action.RIGHT)

        return actions

    def show_trace(self):
        plt.grid('on')
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, self.n_rows, 1))
        ax.set_yticks(np.arange(0.5, self.n_cols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        canvas = np.copy(self.maze)
        for row, col in self.visited:
            canvas[row, col] = self.mark_visited
        rat_row, rat_col, _ = self.state
        canvas[rat_row, rat_col] = self.mark_rat  # rat cell
        canvas[self.n_rows - 1, self.n_cols - 1] = 0.8  # cheese cell
        plt.imshow(canvas, interpolation='none', cmap=cm.RdYlGn)
        plt.show()


def debug():
    maze = [
        [1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 0., 1., 0., 1.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 1., 1., 0., 1., 1., 1.]
    ]
    q_maze = Maze(maze)
    for _ in range(3):
        canvas, reward, game_over = q_maze.act(Action.DOWN)
        print("reward=", reward)
        q_maze.show_trace()


if __name__ == '__main__':
    debug()
