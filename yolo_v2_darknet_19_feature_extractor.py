#  models/research/object_detection/models/yolo_v2_darknet_19_feature_extractor.py
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""YOLOFeatureExtractor for Darknet19 features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from nets import darknet

slim = tf.contrib.slim


class YOLOv2Darknet19FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """YOLO Feature Extractor using Darknet19 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None):
    """Darknet19 Feature Extractor for SSD Models.
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: Not used in YOLO
      min_depth:        Not used in YOLO
      pad_to_multiple:  Not used in YOLO
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(YOLOv2Darknet19FeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights)

  def preprocess(self, resized_inputs):
    """Darknet19 preprocessing.
    Maps pixel values to the range [-1, 1].
    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.
    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(darknet.darknet_arg_scope(is_training = self._is_training)):
        with tf.variable_scope('darknet_19',
                               reuse=self._reuse_weights) as scope:
          net, end_points = darknet.darknet_19_base(preprocessed_inputs, 
                                                        scope='base')
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_19')
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_20')
          scope_name = end_points['scope_name']
          conv_13 = end_points[scope_name+'/Conv2D_13']          
          conv_21 = slim.conv2d(conv_13, 64, [1, 1], scope='Conv2D_21')
          conv_21 = tf.space_to_depth(conv_21, block_size=2)
          net = tf.concat([conv_21, net], axis=-1)
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_22')
          feature_map = net
          
    return [feature_map]
