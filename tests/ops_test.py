import unittest
import tensorflow as tf
import numpy as np

from ops import *

class TestOps(tf.test.TestCase):
    def test_skew(self):
        batch, height, width, channel = 2,2,2,3 # 24
        a = np.arange(batch*height*width*channel)
        a = np.reshape(a, (batch, height, width, channel))

        expected_outputs = np.array([[[[0,1,2], [3,4,5], [0,0,0]],
          [[0,0,0], [6,7,8], [9,10,11]]],
         [[[12,13,14], [15,16,17], [0,0,0]],
          [[0,0,0], [18,19,20], [21,22,23]]]])

        inputs = tf.constant(a, tf.int32)

        outputs = skew(inputs)
        self.assertEqual(get_shape(outputs), [batch, height, height+width-1, channel])
        with self.test_session():
            self.assertAllEqual(outputs.eval(), expected_outputs)

    def test_unskew(self):
        batch, height, width, channel = 20,5,7,3
        a = np.arange(batch*height*width*channel)
        a = np.reshape(a, (batch, height, width, channel))

        inputs = tf.constant(a, tf.int32)

        skew_outputs = skew(inputs)
        unskew_outputs = unskew(skew_outputs)
        self.assertEqual(get_shape(unskew_outputs), [batch, height, width, channel])
        with self.test_session():
            self.assertAllEqual(unskew_outputs.eval(), inputs.eval())
