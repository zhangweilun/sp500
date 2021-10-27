from enum import Enum

WINDOW_SIZE = 5
FEATURE_NUMS = 5
EPOCHS = 150
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10
MAX_ACC = 0.0


class Env(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3
