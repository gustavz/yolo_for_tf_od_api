#  models/research/slim/nets/darknet_test.py
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.darknet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import darknet

scope_prefix = 'darknet_19/base/'

class DarknetTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    predictions = end_points['Predictions']
    self.assertTrue(logits.op.name.startswith('darknet_19/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    self.assertTrue(predictions.op.name.startswith(
        'darknet_19/Predictions'))
    self.assertListEqual(predictions.get_shape().as_list(),
                         [batch_size, num_classes])

  def testAllEndPointsShapes(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    _, end_points = darknet.darknet_19(inputs, num_classes)
    endpoints_shapes = {scope_prefix+'Conv2D_1': [batch_size, 224, 224, 32],
                        scope_prefix+'MaxPool_1': [batch_size,112, 112, 32],
                        #
                        scope_prefix+'Conv2D_2': [batch_size, 112, 112, 64],
                        scope_prefix+'MaxPool_2': [batch_size,56, 56, 64],
                        #
                        scope_prefix+'Conv2D_3': [batch_size, 56, 56, 128],
                        scope_prefix+'Conv2D_4': [batch_size, 56, 56, 64],
                        scope_prefix+'Conv2D_5': [batch_size, 56, 56, 128],
                        scope_prefix+'MaxPool_3': [batch_size,28, 28, 128],
                        #
                        scope_prefix+'Conv2D_6': [batch_size, 28, 28, 256],
                        scope_prefix+'Conv2D_7': [batch_size, 28, 28, 128],
                        scope_prefix+'Conv2D_8': [batch_size, 28, 28, 256],
                        scope_prefix+'MaxPool_4': [batch_size,14, 14, 256],
                        #
                        scope_prefix+'Conv2D_9': [batch_size,  14, 14, 512],
                        scope_prefix+'Conv2D_10': [batch_size, 14, 14, 256],
                        scope_prefix+'Conv2D_11': [batch_size, 14, 14, 512],
                        scope_prefix+'Conv2D_12': [batch_size, 14, 14, 256],
                        scope_prefix+'Conv2D_13': [batch_size, 14, 14, 512],
                        scope_prefix+'MaxPool_5': [batch_size,  7,  7, 512],
                        #
                        scope_prefix+'Conv2D_14': [batch_size, 7, 7, 1024],
                        scope_prefix+'Conv2D_15': [batch_size, 7, 7, 512],
                        scope_prefix+'Conv2D_16': [batch_size, 7, 7, 1024],
                        scope_prefix+'Conv2D_17': [batch_size, 7, 7, 512],
                        scope_prefix+'Conv2D_18': [batch_size, 7, 7, 1024],
                        #
                        'Conv2D_19': [batch_size, 7, 7, num_classes],
                        # Logits and predictions
                        'global_pool': [batch_size, 1, 1, num_classes],
                        'Logits': [batch_size, num_classes],
                        'Predictions': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)

  def testBuildBaseNetwork(self):
    batch_size = 5
    height, width = 224, 224
    inputs = tf.random_uniform((batch_size, height, width, 3))
    scope_name = 'darknet_19_base/'
    net, end_points = darknet.darknet_19_base(inputs, scope=scope_name)
    self.assertTrue(net.op.name.startswith(scope_name+'Conv2D_18'))
    self.assertListEqual(net.get_shape().as_list(), [batch_size, 7, 7, 1024])
    all_endpoints = [scope_name+'Conv2D_' + str(i) for i in range(1, 19)]+ \
                    [scope_name+'MaxPool_'+ str(i) for i in range(1, 6)]
    self.assertItemsEqual(end_points.keys(), all_endpoints)
    for name, op in end_points.iteritems():
      self.assertTrue(op.name.startswith(name))

  def testVariablesSetDevice(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    # Force all Variables to reside on the device.
    with tf.variable_scope('on_cpu'), tf.device('/cpu:0'):
      darknet.darknet_19(inputs, num_classes)
    with tf.variable_scope('on_gpu'), tf.device('/gpu:0'):
      darknet.darknet_19(inputs, num_classes)
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_cpu'):
      self.assertDeviceEqual(v.device, '/cpu:0')
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='on_gpu'):
      self.assertDeviceEqual(v.device, '/gpu:0')

  def testHalfSizeImages(self):
    batch_size = 5
    height, width = 112, 112
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('darknet_19/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points[scope_prefix+'Conv2D_18']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 3, 3, 1024])

  def testGlobalPool(self):
    batch_size = 2
    height, width = 448, 448
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    logits, end_points = darknet.darknet_19(inputs, num_classes)
    self.assertTrue(logits.op.name.startswith('darknet_19/Logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [batch_size, num_classes])
    pre_pool = end_points[scope_prefix+'Conv2D_18']
    self.assertListEqual(pre_pool.get_shape().as_list(),
                         [batch_size, 14, 14, 1024])

  def testGlobalPoolUnknownImageShape(self):
    batch_size = 2
    height, width = 448, 448
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (batch_size, None, None, 3))
      logits, end_points = darknet.darknet_19(
          inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('darknet_19/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      pre_pool = end_points[scope_prefix+'Conv2D_18']
      images = tf.random_uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      logits_out, pre_pool_out = sess.run([logits, pre_pool],
                                          {inputs: images.eval()})
      self.assertTupleEqual(logits_out.shape, (batch_size, num_classes))
      self.assertTupleEqual(pre_pool_out.shape, (batch_size, 14, 14, 1024))

  def testUnknownBatchSize(self):
    batch_size = 1
    height, width = 224, 224
    num_classes = 1000
    with self.test_session() as sess:
      inputs = tf.placeholder(tf.float32, (None, height, width, 3))
      logits, _ = darknet.darknet_19(inputs, num_classes)
      self.assertTrue(logits.op.name.startswith('darknet_19/Logits'))
      self.assertListEqual(logits.get_shape().as_list(),
                           [None, num_classes])
      images = tf.random_uniform((batch_size, height, width, 3))
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEquals(output.shape, (batch_size, num_classes))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session() as sess:
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = darknet.darknet_19(eval_inputs,
                                         num_classes,
                                         is_training=False)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (batch_size,))

  def testTrainEvalWithReuse(self):
    train_batch_size = 5
    eval_batch_size = 2
    height, width = 150, 150
    num_classes = 1000
    with self.test_session() as sess:
      train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
      darknet.darknet_19(train_inputs, num_classes)
      eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
      logits, _ = darknet.darknet_19(eval_inputs,
                                         num_classes,
                                         is_training=False,
                                         reuse=True)
      predictions = tf.argmax(logits, 1)
      sess.run(tf.global_variables_initializer())
      output = sess.run(predictions)
      self.assertEquals(output.shape, (eval_batch_size,))


if __name__ == '__main__':
  tf.test.main()

