import sys
import os

import logging

import numpy as np
import pandas as pd


import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.regularizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras.losses import * 
    import keras.preprocessing.image.iterator
    keras.preprocessing.image.iterator.BatchFromFilesMixin.set_processing_attrs = plantpathology.keras_custom_patch.set_processing_attrs
    from keras.preprocessing.image import ImageDataGenerator

    from keras import backend as K
    from keras.utils import generic_utils

logger = logging.getLogger(__name__)

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    
def steps_from_gen(generator):
    return generator.n // generator.batch_size



class TrisplitImageDataGenerator(ImageDataGenerator):
    def __init__(self, *argv, testing_split = 0, **kwargs):
        super().__init__(*argv, **kwargs)
        if testing_split and not 0 < testing_split < 1:
            raise ValueError(
                "`testing_split` must be strictly between 0 and 1. "
                " Received: %s" % testing_split)
        self._testing_split = testing_split
        if not 0 <= self._validation_split + self._testing_split < 1:
            raise ValueError("(validation_split + testing_split) must lies between [0, 1), currently: %")

