import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from maze_env import Qmaze, DOWN


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.2  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.8  # cheese cell
    # img = plt.imshow(canvas, interpolation='none', cmap='gray')
    # img = plt.imshow(canvas, interpolation='none', cmap=cm.RdYlGn)
    # img = plt.imshow(canvas, interpolation='none', cmap="PuOr")
    plt.imshow(canvas, interpolation='none', cmap=cm.RdYlGn)
    plt.show()
    # return img


def main():
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
    qmaze = Qmaze(maze)
    canvas, reward, game_over = qmaze.act(DOWN)
    print("reward=", reward)
    show(qmaze)


if __name__ == '__main__':
    main()
