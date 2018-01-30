#  models/research/object_detection/anchor_generators/yolo_grid_anchor_generator_test.py
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

"""Tests for anchor_generators.yolo_grid_anchor_generator_test.py."""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import yolo_grid_anchor_generator as ag


class YoloGridAnchorGeneratorTest(tf.test.TestCase):

  def test_construct_single_anchor_grid(self):
    """Builds a 1x1 anchor grid to test the size of the output boxes."""
    exp_anchor_corners = [[0.2, 0.25, 0.8, 0.75]]
    ## h_min, w_min, h_max, w_max

    box_specs = [0.5, 0.6] # w, h 
    anchor_generator = ag.YoloGridAnchorGenerator(
        box_specs,
        base_anchor_size=tf.constant([1, 1], dtype=tf.float32))
    anchors = anchor_generator.generate(feature_map_shape_list=[(1, 1)])
    anchor_corners = anchors.get()
    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)


  def test_construct_anchor_grid(self):
    box_specs_list = [0.5, 0.6, 1., 1.]

    exp_anchor_corners = [[0.09999999, 0.125, 0.40000001, 0.375],
                          [0.        , 0.   , 0.5       , 0.5  ],
                          [0.09999999, 0.625, 0.40000001, 0.875],
                          [0.        , 0.5  , 0.5       , 1.   ],
                          [0.60000002, 0.125, 0.89999998, 0.375],
                          [0.5       , 0.   , 1.        , 0.5  ],
                          [0.60000002, 0.625, 0.89999998, 0.875],
                          [0.5       , 0.5  , 1.        , 1.   ]]

    anchor_generator = ag.YoloGridAnchorGenerator(
        box_specs_list,
        base_anchor_size=tf.constant([1, 1], dtype=tf.float32))
    anchors = anchor_generator.generate(feature_map_shape_list=[(2, 2)])
    anchor_corners = anchors.get()
    with self.test_session():
      anchor_corners_out = anchor_corners.eval()
      self.assertAllClose(anchor_corners_out, exp_anchor_corners)


if __name__ == '__main__':
  tf.test.main()

