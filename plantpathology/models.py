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

    def prepare_base_model(self):
        base_model = inception_v3.InceptionV3(include_top = False,
                                              weights = "imagenet",
                                              input_shape = self.input_shape)
        base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False

        return base_model


    def prepare_model(self, inputs):
        # Dimensionality Reduction
        conv_1_reduce = Conv2D(64, (1, 1), padding = "valid", activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "conv_1_reduce")(inputs)
        conv_1 = Conv2D(192, (3, 3), padding = "same", activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "conv_1")(conv_1_reduce)
        conv_1_bn = BatchNormalization(name = "conv_1_bn")(conv_1)
        conv_1_activation = Activation("relu", name = "conv_1_activation")(conv_1_bn)
        
        pool_1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "valid", name = "maxpool_1")(conv_1_activation)

        flatten = Flatten()(pool_1) # inputs)

        # fc_1 = Dense(1024, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_1")(flatten)
        # dropout_1 = Dropout(self.dropout_rate)(fc_1)

        # fc_2 = Dense(512, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_2")(dropout_1)
        # dropout_2 = Dropout(self.dropout_rate)(fc_2)

        # fc_3 = Dense(256, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_3")(dropout_2)
        fc_3 = Dense(256, activation = "relu", kernel_regularizer = self.kernel_regularizer, name = "fc_3")(flatten)
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

        return outputs

class SENetTrial(BaselineCNN):
    @property
    def squeeze_excite_ratio(self):
        return 16

    def prepare_model(self, inputs):
        inputs_attention = self.se_block(inputs)
        return super().prepare_model(inputs_attention)

    def se_block(self, inputs):
        initial_input = inputs
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = initial_input._keras_shape[channel_axis]

        global_pool = GlobalAveragePooling2D()(inputs)
        reshape = Reshape((1, 1, filters))(global_pool)
        fc_1 = Dense(filters // self.squeeze_excite_ratio, activation = "relu", kernel_initializer = "he_normal", use_bias = False, name = "se_fc_1")(reshape)
        fc_2 = Dense(filters, activation = "sigmoid", kernel_initializer = "he_normal", use_bias = False, name = "se_fc_2")(fc_1)

        if K.image_data_format() == "channels_first":
            fc_2 = Permute((3, 1, 2))(fc_2)

        return multiply([initial_input, fc_2])




# class ResNetTrial(BaselineCNN):


