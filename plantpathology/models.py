from plantpathology.base_model import *

class BaselineMLP(BaseModel):
    @property
    def use_earlystopping(self):
        return True

    @property
    def kernel_regularizer(self):
        return lr(0.0002)

    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
         
