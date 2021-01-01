"""
Augmentor - Flip and Rotate Only
"""

from augmentor import *


def get_augmentation(**kwargs):
    augs = list()
    augs.append(FlipRotate())
    return Compose(augs)

Augmentor = get_augmentation
