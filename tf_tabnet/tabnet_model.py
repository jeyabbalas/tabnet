from typing import Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import entmax15


class GLULayer(tf.keras.layers.Layer):
    def __init__(
            self, 
            units: int = 16, 
            instance_norm: bool = False, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
        ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch 
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        instance_norm: bool
            If True, it runs Instance Normalization. Default (False) runs regular batch 
            normalization when virtual_batch_size is None, else runs Ghost Batch
            Normalization.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be smaller 
            than overall batch size. Default (None) runs regular batch normalization. 
            If an integer value is specified, GBN is run.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(GLULayer, self).__init__()
        self.units = units

        # Building layer components
        self.fc = tf.keras.layers.Dense(self.units*2, use_bias=False)

        if not instance_norm: # Ghost Batch Normalization (default)
            self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=virtual_batch_size, 
                                                         momentum=momentum)
        else: # Instance Normalization
            self.bn = tfa.layers.GroupNormalization(groups=-1)
    
    def call(self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, :self.units], tf.nn.sigmoid(x[:, self.units:]))
        return x


class GLUBlock(tf.keras.layers.Layer):
    def __init__(
            self, 
            n_glu_layers: int = 2, 
            skip_first_residual: bool = True, 
            units: int = 16, 
            instance_norm: bool = False, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
        ):
        """
        Creates a sequence of n_glu_layers GLU layers with residual connections.

        Parameters:
        -----------
        n_glu_layers: int
            Number of GLU (FC-BN-GLU) layers. Default (2).
        skip_first_residual: boolean
            Skip the residual connection from input to the first GLU layer's output? 
            In the TabNet paper, this is True for the shared GLU block but False for 
            step-dependent GLU block. Default (True).
        units: int
            Number of units in layer. Default (16).
        instance_norm: boolean
            If True, it runs Instance Normalization. Default (False) runs regular batch 
            normalization when virtual_batch_size is None, else runs Ghost Batch
            Normalization.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be smaller 
            than overall batch size. Default (None) runs regular batch normalization. 
            If an integer value is specified, GBN is run.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(GLUBlock, self).__init__()
        self.n_glu_layers = n_glu_layers
        self.skip_first_residual = skip_first_residual
        self.norm_factor = tf.math.sqrt(0.5)

        # Building block components
        self.glu_layers = list()
        for _ in range(self.n_glu_layers):
            self.glu_layers.append(GLULayer(units=units, 
                                            instance_norm=instance_norm, 
                                            virtual_batch_size=virtual_batch_size, 
                                            momentum=momentum))
    
    def call(self, inputs: Union[tf.Tensor, np.ndarray], training: Optional[bool] = None):
        for _ in range(5):
            x = 6
        return x


class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(
            self, 
            units: int = 16, 
            instance_norm: bool = False, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
        ):
        """
        Creates a Feature Transformer capable of non-linear processing of features.

        Parameters:
        -----------
        """
        super(FeatureTransformer, self).__init__()


class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(
        self, 
        units: int, 
        mask_type: str = "sparsemax", 
        relaxation_factor: float = 1.3, 
        instance_norm: bool = False, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        ):
        """
        Creates an Attentive Transformer that learns to select features and output a 
        mask to select features to pay attention to for further analysis.
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = tf.keras.layers.Dense(units, use_bias=False)

        if not instance_norm: # Ghost Batch Normalization (default)
            self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=virtual_batch_size, 
                                                         momentum=momentum)
        else: # Instance Normalization
            self.bn = tfa.layers.GroupNormalization(groups=-1)

        if mask_type == "sparsemax": # Sparsemax
            self.sparse_activation = tfa.activations.sparsemax(axis=-1)
        elif mask_type == "entmax": # Entmax 1.5
            self.sparse_activation = entmax15(axis=-1)
        elif mask_type == "softmax": # Softmax
            self.sparse_activation = tf.nn.softmax(axis=-1)
        else:
            raise NotImplementedError(
                "Available options for mask_type: {'sparsemax', 'entmax', 'softmax'}"
            )
        
        def call(self, inputs: Union[tf.Tensor, np.ndarray], 
                 priors: Optional[Union[tf.Tensor, np.ndarray]], 
                 training: Optional[bool] = None):
            x = self.fc(inputs)
            x = self.bn(x)
            x = self.sparse_activation(x)
            return x


class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        output_dim: int, 
        attention_dim: int = 8, 
        decision_dim: int = 8, 
        n_steps: int = 3, 
        n_shared_glus: int = 2, 
        n_dependent_glus: int = 2, 
        relaxation_factor: float = 1.3, 
        epsilon: float = 1e-15, 
        instance_norm: bool = False, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        mask_type="sparsemax", 
        ):
        """
        Creates the TabNet Encoder network.

        Parameters:
        -----------
        n_steps: int
            Number of sequential attention steps. Default (3).
        instance_norm: boolean
            If True, it runs Instance Normalization. Default (False) runs regular batch 
            normalization when virtual_batch_size is None, else runs Ghost Batch
            Normalization.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be smaller 
            than overall batch size. Default (None) runs regular batch normalization. 
            If an integer value is specified, GBN is run.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(TabNetEncoder, self).__init__()


class TabNet(tf.keras.layers.Layer):
    def __init__(
        self, 
        output_dim, 
        categorical_idxs=[], 
        categorical_dim=[], 
        categorical_embedding_dim=1, 
        attention_dim: int = 8, 
        decision_dim: int = 8, 
        n_steps: int = 3, 
        n_shared_glus: int = 2, 
        n_dependent_glus: int = 2, 
        relaxation_factor: float = 1.3, 
        epsilon: float = 1e-15, 
        instance_norm: bool = False, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        mask_type="sparsemax", 
        ):
        """
        Creates a TabNet network.

        Parameters:
        -----------
        n_steps: int
            Number of sequential attention steps. Default (3).
        instance_norm: boolean
            If True, it runs Instance Normalization. Default (False) runs regular batch 
            normalization when virtual_batch_size is None, else runs Ghost Batch
            Normalization.
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be smaller 
            than overall batch size. Default (None) runs regular batch normalization. 
            If an integer value is specified, GBN is run.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.99).
        """
        super(TabNetEncoder, self).__init__()
        self.n_steps = n_steps

        self._verify_parameters()

    def _verify_parameters(self):
        if self.n_steps <= 0:
            raise ValueError("Invalid argument: n_steps should be a positive integer.")