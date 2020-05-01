import logging

from plantpathology.utils import *
from plantpathology.models import *

datasets = ["train", "test"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    train_df, test_df = list(map(lambda basename: pd.read_csv("%s.csv" % basename), datasets))


