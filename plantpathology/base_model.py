import os

from datetime import datetime
from plantpathology.utils import *

class CustomEarlyStopping(EarlyStopping):

    def __init__(self, target=None, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        current = self.get_monitor_value(logs)
        if not self.target or self.monitor_op(self.target, self.best):
            super().on_epoch_end(epoch, logs)

class BaseModel:
    def __init__(self,
                 input_shape = (299, 299, 3), # (2048, 1365, 3),
                 output_shape = (4, ),
                 batch_size = None,
                 epochs = 1000,
                 verbose = 2,
                 validation_split = 0.3,
                 testing_split = 0.1,
                 use_multiprocessing = False,
                 compiled = False,
                 *argv, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.compiled = compiled
        self.epochs = epochs
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.validation_split = validation_split
        self.testing_split = testing_split
        self.data_generator = TrisplitImageDataGenerator(validation_split=self.validation_split,
                                                         testing_split=self.testing_split,
                                                         # rescale=1./255.,
                                                         preprocessing_function = self.preprocessing_function)
        self.__dict__.update(kwargs)

        self.init()
        self.model = self.prepare_model()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        try:
            self.load_weights()
        except:
            raise ImportError("Could not load pretrained model weights")

        if not self.compiled:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
            print ("compiled: %s" % self.__class__.__name__)

        # self.model.summary()

    @property
    def name(self):
        return self.__class__.__name__

    def init(self):
        pass

    @property
    def optimizer(self):
        return Adadelta()

    @property
    def loss(self):
        return "binary_crossentropy"

    @property
    def preprocessing_function(self):
        """ To be overriden """
        return None

    @property
    def weight_filename(self):
        return "%s.h5" % self.name

    def load_weights(self, filename = None):
        if filename is None:
            filename = self.weight_filename

        if os.path.exists(filename):
            self.model.load_weights(filename, by_name=True, skip_mismatch=True)

    def save_weights(self):
        self.model.save_weights(self.weight_filename)

    @property
    def metrics(self):
        return [AUC()]


    @property
    def use_earlystopping(self):
        return False

    @property
    def main_metric(self):
        return "auc_2"

    @property
    def metric_mode(self):
        return "max"

    @property
    def earlystopping(self):
       return CustomEarlyStopping(monitor="val_%s" % self.main_metric, # use validation accuracy for stopping
                                  mode = self.metric_mode,
                                  # min_delta = 0.0001,
                                  patience = 20, 
                                  verbose = self.verbose,
                                  target = 0.9)

    @property
    def modelcheckpoint(self):
        return ModelCheckpoint(os.path.join(self.logdir, "epoch{epoch:03d}-%(metric)s{%(metric)s:.3f}-val_%(metric)s{val_%(metric)s:.3f}.h5") % dict(metric = self.main_metric), monitor="val_%s" % self.main_metric, save_weights_only=True, save_best_only=True, mode = self.metric_mode) # period=3, 


    @property
    def callbacks(self):
        callbacks = [
            self.modelcheckpoint,
            TensorBoard(log_dir=self.logdir, write_graph = True),
            TerminateOnNaN(),
        ]
        if self.use_earlystopping:
            callbacks.append(self.earlystopping)

        return callbacks

    @property
    def logdir(self):
        return "logs/%s/%s" % (self.__class__.__name__, datetime.now().strftime("%Y%m%d-%H%M%S"))

    def prepare_model(self):
        raise NotImplementedError("prepare_model must be overrided by subclass")

    def fit(self, train_X, train_Y, validation_X, validation_Y):
        history = self.model.fit(train_X, train_Y,
                                 validation_data = (validation_X, validation_Y),
                                 batch_size = self.batch_size,
                                 epochs = self.epochs,
                                 callbacks = self.callbacks,
                                 verbose = self.verbose,
                                 use_multiprocessing = self.use_multiprocessing)
        self.save_weights()
        return history

    def flow_from_dataframe(self, dataframe, subset = None, class_mode = "raw", directory = "preprocessed"): # "images"):
        return self.data_generator.flow_from_dataframe(dataframe = dataframe,
                                                       subset = subset,
                                                       directory = directory,
                                                       x_col = "image_id",
                                                       y_col = ["healthy", "multiple_diseases", "rust", "scab"],
                                                       # has_ext = False,
                                                       class_mode = class_mode,
                                                       target_size = self.input_shape[:2])

    def fit_df(self, train_df, validation_df, **kwargs):
        train_generator = self.flow_from_dataframe(train_df, **kwargs) # , "training", **kwargs)
        validation_generator = self.flow_from_dataframe(validation_df, **kwargs) # "validation", **kwargs)


        history = self.model.fit_generator(generator = train_generator,
                                           steps_per_epoch = steps_from_gen(train_generator),
                                           validation_data = validation_generator,
                                           validation_steps = steps_from_gen(validation_generator),
                                           epochs = self.epochs,
                                           callbacks = self.callbacks,
                                           verbose = self.verbose)
                                           # batch_size = self.batch_size)
        self.save_weights()
        return history


    def evaluate(self, test_X, test_Y):
        return self.model.evaluate(test_X, test_Y,
                                   batch_size = self.batch_size,
                                   verbose = self.verbose,
                                   use_multiprocessing = self.use_multiprocessing)

    def evaluate_df(self, df, **kwargs):
        test_generator = self.flow_from_dataframe(df, **kwargs) # , "testing", **kwargs)
        return self.model.evaluate_generator(generator = test_generator,
                                             steps = steps_from_gen(test_generator),
                                             callbacks = self.callbacks,
                                             verbose = self.verbose)

    def predict(self, X, *argv, **kwargs):
        return self.model.predict(X, *argv, **kwargs)

    def predict_df(self, df, **kwargs):
        test_generator = self.flow_from_dataframe(df, class_mode = None, **kwargs)
        return self.model.predict_generator(test_generator,
                                            steps = steps_from_gen(test_generator),
                                            callbacks = self.callbacks,
                                            verbose = self.verbose)
