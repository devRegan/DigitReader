import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = (28, 28)
MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100

NORMALIZE = True
AUTO_INVERT = True

CONFIDENCE_THRESHOLD = 0.25

NUMBER_TO_WORD = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
}