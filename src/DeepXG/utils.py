import math

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from constants import *
from math import asin, sin, atan2, degrees


def get_shot_angle(shot_x: float, shot_y: float):

    deg2 = (360 + degrees(atan2(GOAL_POSTS[0][0] - shot_x, GOAL_POSTS[0][1] - shot_y))) % 360
    deg1 = (360 + degrees(atan2(GOAL_POSTS[1][0] - shot_x, GOAL_POSTS[1][1] - shot_y))) % 360
    angle = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)
    print(f"{angle}")
    return angle


def get_shot_distance(shot_x: float, shot_y: float):
    return np.sqrt((shot_x-GOAL_CENTER[0])**2 + (shot_y-GOAL_CENTER[1])**2)


def get_pass_shot_zones(x, y):
    pitch = Pitch(line_color='black', pitch_type='statsbomb')
    ret = pitch.bin_statistic(x, y, None, 'count', bins=[10, 8])
    zones = [(ret["binnumber"][0][i], ret["binnumber"][1][i]) for i in range(len(ret["binnumber"][0]))]
    return zones




def random_draw():
    pitch = Pitch(pitch_color="green")
    fig, ax = pitch.grid(nrows=1, ncols=1, grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0, figheight=12)
    pitch.lines(xstart=80, ystart=20, xend=120, yend=35, ax=ax["pitch"])
    pitch.lines(xstart=80, ystart=20, xend=120, yend=45, ax=ax["pitch"])
    plt.show()


if __name__ == "__main__":
    get_shot_angle(80, 15)
    random_draw()