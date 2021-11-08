from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import entmax15


class GLULayer(tf.keras.layers.Layer):
    def __init__(
            self, 
            units: int = 16, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
            groups: Optional[int] = None, 
            **kwargs
    ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch 
        normalization, and a gated linear unit (GLU) as the activation function.

        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        groups: int
            Number of groups for Group Normalization. Default (None) runs either regular 
            batch normalization or Ghost Batch Normalization depending upon the value of 
            virtual_batch_size. If an integer value is specified, Group Normalization is 
            run with that number of groups. For Layer Normalization, set group=1. For 
            Instance Normalization, set group=-1.
        """
        super(GLULayer, self).__init__(**kwargs)
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.groups = groups

        self.fc = tf.keras.layers.Dense(self.units*2, use_bias=False)

    def build(
        self, 
        input_shape: tf.TensorShape
    ):
        if self.groups: # Group Normalization
            if self.groups == -1:
                self.groups = input_shape[-1]
            self.bn = tfa.layers.GroupNormalization(groups=self.groups)
        else: # Batch Normalization
            self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                         momentum=self.momentum)
    
    def call(
        self, 
        inputs: Union[tf.Tensor, np.ndarray], 
        training: Optional[bool] = None, 
    ) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, :self.units], tf.nn.sigmoid(x[:, self.units:]))
        return x


class GLUBlock(tf.keras.layers.Layer):
    def __init__(
            self, 
            n_glu_layers: int = 2, 
            units: int = 16, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
            groups: Optional[int] = None, 
            **kwargs
    ):
        """
        Creates a sequence of n_glu_layers GLU layers with residual connections.

        Parameters:
        -----------
        n_glu_layers: int
            Number of GLU (FC-BN-GLU) layers. Should be greater than 0. Default (2).
        units: int
            Number of units in each GLU layer. Default (16).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        groups: int
            Number of groups for Group Normalization. Default (None) runs either regular 
            batch normalization or Ghost Batch Normalization depending upon the value of 
            virtual_batch_size. If an integer value is specified, Group Normalization is 
            run with that number of groups. For Layer Normalization, set group=1. For 
            Instance Normalization, set group=-1.
        """
        super(GLUBlock, self).__init__(**kwargs)
        if n_glu_layers <= 0:
            raise ValueError("Invalid Argument: Number of GLU layers should be greater than 0.")

        self.units = units
        self.norm_factor = tf.math.sqrt(tf.constant(0.5))

        # Building block components
        self.glu_layers = list()
        for _ in range(n_glu_layers):
            glu_layer = GLULayer(
                units=self.units, 
                virtual_batch_size=virtual_batch_size, 
                momentum=momentum, 
                groups=groups, 
            )
            self.glu_layers.append(glu_layer)
    
    def build(
        self, 
        input_shape: tf.TensorShape, 
    ):
        if input_shape[-1] != self.units:
            self.omit_first_residual = True
        else: 
            self.omit_first_residual = False
    
    def call(
        self, 
        inputs: Union[tf.Tensor, np.ndarray], 
        training: Optional[bool] = None, 
    ) -> tf.Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            x = glu_layer(inputs, training=training)
            if self.omit_first_residual and (i==0):
                inputs = x
            else:
                x = tf.math.multiply(self.norm_factor, tf.math.add(inputs, x))
                inputs = x
        
        return x


class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(
            self, 
            n_dependent_glus: int = 2, 
            shared_layers: Optional[GLUBlock] = None, 
            units: int = 16, 
            virtual_batch_size: Optional[int] = None, 
            momentum: float = 0.98, 
            groups: Optional[int] = None, 
            **kwargs
    ):
        """
        Creates a Feature Transformer for non-linear processing of features.

        Parameters:
        -----------
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        shared_layers: GLUBlock
            GLU block that is shared among all feature transformers. Default (None).
        units: int
            Number of units in each GLU layer. Default (16).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        groups: int
            Number of groups for Group Normalization. Default (None) runs either regular 
            batch normalization or Ghost Batch Normalization depending upon the value of 
            virtual_batch_size. If an integer value is specified, Group Normalization is 
            run with that number of groups. For Layer Normalization, set group=1. For 
            Instance Normalization, set group=-1.
        """
        super(FeatureTransformer, self).__init__(**kwargs)
        self.shared_layers = shared_layers
        self.dependent_layers = None
        if n_dependent_glus > 0:
            self.dependent_layers = GLUBlock(
                n_glu_layers=n_dependent_glus, 
                units=units, 
                virtual_batch_size=virtual_batch_size, 
                momentum=momentum, 
                groups=groups, 
            )
    
    def call(
        self, 
        inputs: Union[tf.Tensor, np.ndarray], 
        training: Optional[bool] = None, 
    ) -> tf.Tensor:
        x = inputs
        if not self.shared_layers:
            x = self.shared_layers(x, training=training)
        if not self.dependent_layers:
            x = self.dependent_layers(x, training=training)
        return x


class Split(tf.keras.layers.Layer):
    def __init__(
        self, 
        split_dim: int = 8, 
        **kwargs
    ):
        """
        Splits the input tensor into two at a specified column dimension.

        Parameters:
        -----------
        split_dim: int
            Column dimension where the input tensor should be split into two. Default (8).
        """
        super(Split, self).__init__(**kwargs)
        self.split_dim = split_dim
    
    def call(
        self, 
        inputs: Union[tf.Tensor, np.ndarray], 
    ) -> Tuple[tf.Tensor]:
        return inputs[:, :self.split_dim], inputs[:, self.split_dim:]


class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(
        self, 
        units: int, 
        mask_type: str = "sparsemax", 
        relaxation_factor: float = 1.3, 
        instance_norm: bool = False, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        **kwargs
    ):
        """
        Creates an Attentive Transformer that learns to select features and output a 
        mask to select features to pay attention to for further analysis.
        """
        super(AttentiveTransformer, self).__init__(**kwargs)
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
        decision_dim: int = 8, 
        attention_dim: int = 8, 
        n_steps: int = 3, 
        n_shared_glus: int = 2, 
        n_dependent_glus: int = 2, 
        relaxation_factor: float = 1.3, 
        epsilon: float = 1e-15, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        groups: Optional[int] = None, 
        mask_type: str = "sparsemax", 
        **kwargs
    ):
        """
        Creates a TabNet Encoder network.

        Parameters:
        -----------
        output_dim: int
            Dimension of the network's output layer.
        decision_dim: int
            Dimension of the decision layer. Typically ranges from 8 to 128. Assuming 
            decision_dim to be equal to attention_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        attention_dim: int
            Dimension of the attention layer. Typically ranges from 8 to 128. Assuming 
            attention_dim to be equal to decision_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the 
            data has more informative features, the number of steps is higher. Large 
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the 
            number of shared layers is an effective strategy to improve predictive performance 
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        relaxation_factor: float
            Relaxation parameter used to compute the prior in the Attentive Transformer 
            layers. Typically ranges from 1.0 to 2.0. This is an important hyperparameter 
            to tune in TabNets. Default (1.3).
        epsilon: float
            Tiny number to prevent computing log(0).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        groups: int
            Number of groups for Group Normalization. Default (None) runs either regular 
            batch normalization or Ghost Batch Normalization depending upon the value of 
            virtual_batch_size. If an integer value is specified, Group Normalization is 
            run with that number of groups. For Layer Normalization, set group=1. For 
            Instance Normalization, set group=-1.
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Sparsemax generates 
            a highly sparse mask. To learn more, refer: https://arxiv.org/abs/1905.05702.
        """
        super(TabNetEncoder, self).__init__(**kwargs)
        self.momentum = momentum

        # plain batch normalization
        self.initial_bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)

        # shared glu block
        self.shared_glu_block = None
        if n_shared_glus > 0:
            self.shared_glu_block = GLUBlock(
                n_glu_layers=n_shared_glus, 
                units=decision_dim+attention_dim, 
                virtual_batch_size=virtual_batch_size, 
                momentum=momentum, 
                groups=groups, 
            )
        
        # initial feature transformer
        self.initial_feature_transformer = FeatureTransformer(
            n_shared_glus=n_shared_glus, 
            n_dependent_glus=n_dependent_glus, 
            shared_layers=self.shared_glu_block, 
            units=decision_dim+attention_dim, 
            virtual_batch_size=virtual_batch_size, 
            momentum=momentum, 
            groups=groups, 
        )

    def call(
        self, 
        inputs: Union[tf.Tensor, np.ndarray], 
    ):
        x = self.initial_bn(inputs)



class TabNet(tf.keras.layers.Layer):
    def __init__(
        self, 
        output_dim: int, 
        categorical_idxs=[], 
        categorical_dim=[], 
        categorical_embedding_dim=1, 
        decision_dim: int = 8, 
        attention_dim: int = 8, 
        n_steps: int = 3, 
        n_shared_glus: int = 2, 
        n_dependent_glus: int = 2, 
        relaxation_factor: float = 1.3, 
        epsilon: float = 1e-15, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        groups: Optional[int] = None, 
        mask_type: str = "sparsemax", 
        **kwargs
    ):
        """
        Creates a TabNet network.

        Parameters:
        -----------
        output_dim: int
            Dimension of the network's output layer.
        decision_dim: int
            Dimension of the decision layer. Typically ranges from 8 to 128. Assuming 
            decision_dim to be equal to attention_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        attention_dim: int
            Dimension of the attention layer. Typically ranges from 8 to 128. Assuming 
            attention_dim to be equal to decision_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the 
            data has more informative features, the number of steps is higher. Large 
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the 
            number of shared layers is an effective strategy to improve predictive performance 
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        relaxation_factor: float
            Relaxation parameter used to compute the prior in the Attentive Transformer 
            layers. Typically ranges from 1.0 to 2.0. This is an important hyperparameter 
            to tune in TabNets. Default (1.3).
        epsilon: float
            Tiny number to prevent computing log(0).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        groups: int
            Number of groups for Group Normalization. Default (None) runs either regular 
            batch normalization or Ghost Batch Normalization depending upon the value of 
            virtual_batch_size. If an integer value is specified, Group Normalization is 
            run with that number of groups. For Layer Normalization, set group=1. For 
            Instance Normalization, set group=-1.
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Sparsemax generates 
            a highly sparse mask. To learn more, refer: https://arxiv.org/abs/1905.05702.
        """
        super(TabNetEncoder, self).__init__(**kwargs)
        self.n_steps = n_steps

        self._verify_parameters()

    def _verify_parameters(self):
        if self.n_steps <= 0:
            raise ValueError("Invalid argument: n_steps should be a positive integer.")