#  models/research/object_detection/anchor_generators/yolo_grid_anchor_generator.py
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

"""Generates grid anchors on the fly corresponding to multiple CNN layers.

Generates grid anchors on the fly corresponding to multiple CNN layers as
described in:
"SSD: Single Shot MultiBox Detector"
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
(see Section 2.2: Choosing scales and aspect ratios for default boxes)
"""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops
from object_detection.core import box_list
from object_detection.utils import ops

class YoloGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               anchors,
               base_anchor_size=None,
               anchor_stride=None,
               anchor_offset=None):
    """Constructs a YoloGridAnchorGenerator.

    To construct anchors, at multiple grid resolutions, one must provide a
    feature_map_shape_list (e.g., [(13, 13)]) a corresponding list of anchor box specifications.

    For example:
    anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 
              5.47434, 7.88282, 3.52778, 9.77052, 9.16828]  # for grid

    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while anchor box specifications are fixed at construction
    time.

    Args:
      anchors: list of anchors of pairs
      base_anchor_size: base anchor size as [height, width]
                        (length-2 float tensor, default=[1.0, 1.0]).
                        The height and width values are normalized to the
                        minimum dimension of the input height and width, so that
                        when the base anchor height equals the base anchor
                        width, the resulting anchor is square even if the input
                        image is not square.
      anchor_stride: pairs of strides in pixels (in y and x directions
        respectively). For example, setting anchor_strides=(25, 25)
        means that we want the anchors corresponding to the first layer to be
        strided by 25 pixels in both y and x directions. If anchor_strides=None, they are set
        to be the reciprocal of the corresponding feature map shapes.
      anchor_offset: pairs of offsets in pixels (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offset=(10, 10) means that we want the
        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space. 
        If anchor_offsets=None, then they are set to be half of the corresponding anchor stride.

    Raises:
      ValueError: if anchors is not a list of floats of pairs
    """
    
    if isinstance(anchors, list) and \
       all([isinstance(list_item, float) for list_item in anchors]) and \
       len(anchors)%2 == 0:
      self._anchors = zip(*[iter(anchors)]*2)
    else:
      raise ValueError('anchors is expected to be a '
                       'list of floats of pairs')

    if base_anchor_size is None:
      base_anchor_size = tf.constant([256, 256], dtype=tf.float32)
    self._base_anchor_size = base_anchor_size
    
    if (anchor_stride and len(anchor_stride) != 2):
      raise ValueError('anchor_stride must be a pair.')

    if (anchor_offset and len(anchor_offset) != 2):
      raise ValueError('anchor_offset must be a pair.')
      
    self._anchor_stride = anchor_stride
    self._anchor_offset = anchor_offset

  def name_scope(self):
    return 'YoloGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a integer, one for expected feature map to be passed to
      the Generate function.
    """
    return [len(self._anchors)]

  def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
    """Generates a collection of bounding boxes to be used as anchors.

    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids.

    Args:
      feature_map_shape_list: list of a pair of convnet layer resolutions in the
        format [(height, width)]. For example,
        setting feature_map_shape=[(8, 8)] asks for anchors that
        correspond to an 8x8 layer.
      im_height: the height of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        normalized coordinates, otherwise absolute coordinates are used for the
        grid.
      im_width: the width of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        normalized coordinates, otherwise absolute coordinates are used for the
        grid.

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if len(feature_map_shape_list) != 1 or len(feature_map_shape_list[0]) != 2:
      raise ValueError('feature_map_shape_list must be a list of a pair')
    
    # Yolo has only one feature_map. so [0] mean the only map which is first.
    feature_map_shape = feature_map_shape_list[0] 
    im_height = tf.to_float(im_height)
    im_width = tf.to_float(im_width)

    if not self._anchor_stride:
      anchor_stride = (1.0 / tf.to_float(feature_map_shape[0]), 
                       1.0 / tf.to_float(feature_map_shape[1]))
    else:
      anchor_stride = (tf.to_float(self._anchor_stride[0]) / im_height,
                       tf.to_float(self._anchor_stride[1]) / im_width)
                         
    if not self._anchor_offset:
      anchor_offset = (0.5 * anchor_stride[0], 
                       0.5 * anchor_stride[1])
    else:
      anchor_offset = (tf.to_float(self._anchor_offset[0]) / im_height,
                       tf.to_float(self._anchor_offset[1]) / im_width)

    if (anchor_stride and len(anchor_stride) != 2):
      raise ValueError('anchor_stride must be a pair.')

    if (anchor_offset and len(anchor_offset) != 2):
      raise ValueError('anchor_offset must be a pair.')
    
    # Anchors are devided into size of feature map to make the size of anchors within (0 ~ 1)  
    anchor_widths  = [anchor[0]/feature_map_shape[0] for anchor in self._anchors]
    anchor_heights = [anchor[1]/feature_map_shape[1] for anchor in self._anchors]
    heights = anchor_heights * self._base_anchor_size[0]  
    widths  = anchor_widths  * self._base_anchor_size[1]

    x_centers = tf.to_float(tf.range(feature_map_shape[0]))
    x_centers = x_centers * anchor_stride[0] + anchor_offset[0]
    y_centers = tf.to_float(tf.range(feature_map_shape[1]))
    y_centers = y_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = ops.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = grid_anchor_generator._center_size_bbox_to_corners_bbox(
                                                   bbox_centers, bbox_sizes)
    anchors = box_list.BoxList(bbox_corners)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is None:
      num_anchors = anchors.num_boxes()
    stddevs_tensor = 0.01 * tf.ones(
        [num_anchors, 4], dtype=tf.float32, name='stddevs')
    anchors.add_field('stddev', stddevs_tensor)

    return [anchors]


def create_yolo_anchors(anchors=None,
                       base_anchor_size=None,
                       anchor_stride=None,
                       anchor_offset=None):
  """Creates YoloGridAnchorGenerator for YOLO anchors.

  This function instantiates a YoloGridAnchorGenerator that reproduces
  ``default box`` construction proposed by YOLO 9000 paper.
  
  Anchors that are returned by calling the `generate` method on the returned
  YoloGridAnchorGenerator object are always in normalized coordinates

  Args:
    anchors: As list of anchors to use. When not None and not emtpy,
    base_anchor_size: base anchor size as [height, width].
      The height and width values are normalized to the minimum dimension of the
      input height and width, so that when the base anchor height equals the
      base anchor width, the resulting anchor is square even if the input image
      is not square.
      anchor_stride: pairs of strides in pixels (in y and x directions
        respectively). For example, setting anchor_strides=(25, 25)
        means that we want the anchors corresponding to the first layer to be
        strided by 25 pixels in both y and x directions. If anchor_strides=None, they are set
        to be the reciprocal of the corresponding feature map shapes.
      anchor_offset: pairs of offsets in pixels (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offset=(10, 10) means that we want the
        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space. 
        If anchor_offsets=None, then they are set to be half of the corresponding anchor stride.

  Returns:
    a YoloGridAnchorGenerator
  """  
  if base_anchor_size is None:
    base_anchor_size = [1.0, 1.0]
  base_anchor_size = tf.constant(base_anchor_size, dtype=tf.float32)
  
  return YoloGridAnchorGenerator(anchors , base_anchor_size,
                                anchor_stride, anchor_offset)

