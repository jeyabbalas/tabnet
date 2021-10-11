import tensorflow as tf
import tensorflow_addons as tfa


def glu(x):
    n = tf.shape(x)[-1] // 2
    return x[:, :n] * tf.nn.sigmoid(x[:, n:])


class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, units=None, momentum=0.99, epsilon=1e-3, 
                 virtual_batch_size=None, instance_norm=False, 
                 **kwargs):
        super(GLUBlock, self).__init__(**kwargs)
        self.units = units
        self.momentum = momentum
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.instance_norm = instance_norm
    
    def build(self, input_shape):
        if not self.units:
            self.units = input_shape[-1]
        self.fc = tf.keras.layers.Dense(self.units*2, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, 
                                                     epsilon=self.epsilon, 
                                                     virtual_batch_size=self.virtual_batch_size)
        if self.instance_norm:
            self.bn = tfa.layers.GroupNormalization(groups=-1, 
                                                    epsilon=self.epsilon)
    
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = glu(x)
        return x
