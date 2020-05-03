from plantpathology.base_model import *

class BaselineCNN(BaseModel):
    @property
    def use_earlystopping(self):
        return True

    @property
    def kernel_regularizer(self):
        return l2(0.0002)

    @property
    def preprocessing_function(self):
        return inception_v3.preprocess_input # inception_resnet_v2.preprocess_input

    @property
    def dropout_rate(self):
        return 0.3

    def prepare_model(self):
        # base_model = inception_resnet_v2.InceptionResNetV2(include_top = False,
        base_model = inception_v3.InceptionV3(include_top = False,
                                              weights = "imagenet",
                                              input_shape = self.input_shape)
        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False

        inputs = base_model.output

        # Dimensionality Reduction
        conv_1_reduce = Conv2D(64, (1, 1), padding = "valid", activation = "relu", kernel_regularizer = self.kernel_regularizer)(inputs)
        conv_1 = Conv2D(192, (3, 3), padding = "same", activation = "relu", kernel_regularizer = self.kernel_regularizer)(conv_1_reduce)
        conv_1_bn = BatchNormalization()(conv_1)
        conv_1_activation = Activation("relu")(conv_1_bn)
        
        pool_1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "valid")(conv_1_activation)

        flatten = Flatten()(pool_1) # inputs)

        # fc_1 = Dense(1024, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_1")(flatten)
        # dropout_1 = Dropout(self.dropout_rate)(fc_1)

        # fc_2 = Dense(512, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_2")(dropout_1)
        # dropout_2 = Dropout(self.dropout_rate)(fc_2)

        # fc_3 = Dense(256, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_3")(dropout_2)
        fc_3 = Dense(256, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_3")(flatten) # dropout_2)
        dropout_3 = Dropout(self.dropout_rate)(fc_3)

        fc_4 = Dense(128, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_4")(dropout_3)
        dropout_4 = Dropout(self.dropout_rate)(fc_4)

        fc_5 = Dense(64, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_5")(dropout_4)
        dropout_5 = Dropout(self.dropout_rate)(fc_5)

        fc_6 = Dense(32, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_6")(dropout_5)
        dropout_6 = Dropout(self.dropout_rate)(fc_6)

        fc_7 = Dense(16, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_7")(dropout_6)
        dropout_7 = Dropout(self.dropout_rate)(fc_7)

        fc_8 = Dense(self.output_shape[0], kernel_regularizer = self.kernel_regularizer, name = "fc_8")(dropout_7)
        outputs = Activation("softmax", name = "output")(fc_8)

        return Model(base_model.input, outputs, name = self.name)



