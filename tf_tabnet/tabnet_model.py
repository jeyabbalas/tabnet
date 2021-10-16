import tensorflow as tf
import tensorflow_addons as tfa


class GLULayer(tf.keras.layers.Layer):
    def __init__(self, units=None, momentum=0.99, epsilon=1e-5, 
                 virtual_batch_size=None, instance_norm=False, 
                 **kwargs):
        """
        Creates a layer with a fully-connected linear layer, followed by batch 
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer.
        momentum: float
            Momentum for exponential moving average in batch normalization. 
            Lower values correspond to larger impact of batch statistics on the 
            rolling statistics computed in each batch. Valid values range from 
            0.0 to 1.0. Default (0.99).
        epsilon: float
            Small value added to the running variance calculaiton to prevent a 
            value of zero. Default (1e-5).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be smaller 
            than overall batch size. Default (None) runs regular batch normalization. 
            If an integer value is specified, GBN is run.
        instance_norm: boolean
            If True, it runs Instance Normalization. Default (None) runs regular batch 
            normalization when virtual_batch_size is  None, else runs Ghost Batch
            Normalization.
        """
        super(GLULayer, self).__init__(**kwargs)
        self.units = units
        self.momentum = momentum
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.instance_norm = instance_norm
    
    def build(self, input_shape):
        if not self.units:
            self.units = input_shape[-1]
        self.fc = tf.keras.layers.Dense(self.units*2, use_bias=False)

        if not self.instance_norm: # Ghost Batch Normalization (default)
            self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, 
                                                         epsilon=self.epsilon, 
                                                         virtual_batch_size=self.virtual_batch_size)
        else: # Instance Normalization
            self.bn = tfa.layers.GroupNormalization(groups=-1, 
                                                    epsilon=self.epsilon)
    
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, :self.units], tf.nn.sigmoid(x[:, self.units:]))
        return x
