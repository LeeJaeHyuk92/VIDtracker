# Date: Wednesday 26 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: helper functions

import random
import math
import numpy as np

RAND_MAX = 2147483647

def sample_rand_uniform():
    """TODO: Docstring for sample_rand_uniform.

    :arg1: TODO
    :returns: TODO

    """
    return (random.randint(0, RAND_MAX) + 1) * 1.0 / (RAND_MAX + 2)

def sample_exp_two_sides(lambda_):
    """TODO: Docstring for sample_exp_two_sides.
    :returns: TODO

    """

    pos_or_neg = random.randint(0, RAND_MAX)
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = sample_rand_uniform()

    return math.log(rand_uniform) / (lambda_ * pos_or_neg)


def sample_exp_two_sides_shift(lambda_):
    """TODO: Docstring for sample_exp_two_sides.
    :returns: TODO

    """

    pos_or_neg = random.randint(0, RAND_MAX)
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = sample_rand_uniform()

    z = math.log(rand_uniform)
    if z < -0.3:
        z = 0.3

    # return math.log(rand_uniform) / (lambda_ * pos_or_neg)
    return z / pos_or_neg


def show_images(images, targets, bbox_gt_scaleds):
    import matplotlib.pyplot as plt
    import math
    import cv2

    fig = plt.figure(figsize=(4, 20))
    rows = len(images)
    cols = 2
    i = 0
    for image, target, bbox_gt_scaled in zip(images, targets, bbox_gt_scaleds):
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, (227, 227), interpolation=cv2.INTER_CUBIC)

        bbox_gt_scaled.unscale(image)
        bbox_gt_scaled.x1, bbox_gt_scaled.x2, bbox_gt_scaled.y1, bbox_gt_scaled.y2 = \
            int(bbox_gt_scaled.x1), int(bbox_gt_scaled.x2), int(bbox_gt_scaled.y1), int(bbox_gt_scaled.y2)
        image = cv2.rectangle(image, (bbox_gt_scaled.x1, bbox_gt_scaled.y1), (bbox_gt_scaled.x2, bbox_gt_scaled.y2), (0, 255, 0), 2)

        # debug
        # print(int(bbox_gt_scaled.x1), int(bbox_gt_scaled.x2), int(bbox_gt_scaled.y1), int(bbox_gt_scaled.y2))

        fig.add_subplot(rows, cols, i+1)
        plt.imshow(image)
        fig.add_subplot(rows, cols, i+2)
        plt.imshow(target)
        i += 2
    plt.show()