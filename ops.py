import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

def get_shape(layer):
    return layer.get_shape().as_list()

def skew(inputs, scope="skew"):
    with tf.name_scope(scope):
        batch, height, width, channel = get_shape(inputs)
        rows = tf.split(inputs, height, 1)

        new_width = width + height - 1
        new_rows = []

        for idx, row in enumerate(rows):
            transposed_row = tf.transpose(tf.squeeze(row, [1]), [0, 2, 1]) # [batch, channel, width]
            squeezed_row = tf.reshape(transposed_row, [-1, width]) # [batch*channel, width]
            padded_row = tf.pad(squeezed_row, [[0, 0], [idx, height - 1 - idx]]) # [batch*channel, width]

            unsqueezed_row = tf.reshape(padded_row, [-1, channel, new_width]) # [batch, channel, new_width]
            untransposed_row = tf.transpose(unsqueezed_row, [0, 2, 1]) # [bacth, new_width, channel]

            assert get_shape(untransposed_row) == [batch, new_width, channel], "wrong shape of skewed row"
            new_rows.append(untransposed_row)
        
        outputs = tf.stack(new_rows, axis=1, name="output")
        assert get_shape(outputs) == [batch, height, new_width, channel], "wrong shape of outputs"

    logger.debug('[skew] %s : %s %s -> %s %s' \
        % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
    return outputs

def unskew(inputs, width=None, scope="unskew"):
    with tf.name_scope(scope):
        batch, height, skewed_width, channel = get_shape(inputs)
        width = width if width else height

        new_rows = []
        rows = tf.split(inputs, height, 1)

        for idx, row in enumerate(rows):
            new_rows.append(tf.slice(row, [0, 0, idx, 0], [-1, -1, skewed_width - width + 1,-1]))
        # outputs = tf.stack(new_rows, axis=1, name="output")
        outputs = tf.concat(new_rows, axis=1, name="output")
    logger.debug('[unskew] %s : %s %s -> %s %s'\
        % (scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))
    return outputs

WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

def conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initilizer=tf.zeros_initializer(),
    biases_regularizer=None,
    scope="conv2d"):
    with tf.variable_scope(scope):
        mask_type = mask_type.lower()
        batch, height, width, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
        "kernel height and width should be odd number"

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        weights_shape = [kernel_h, kernel_w, channel, num_outputs]
        weights = tf.get_variable("weights", weights_shape,
        tf.float32, weights_initializer, weights_regularizer)

        if mask_type is not None:
            mask = np.ones((kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)
            mask[center_h, center_w+1:,:, :] = 0.
            mask[center_h+1, :, :, :] = 0.

            if mask_type == 'a':
                mask[center_h, center_w, :, :] = 0.

            weights *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

        outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
        tf.add_to_collection('conv2d_outputs', outputs)

        if biases_initilizer != None:
            biases = tf.get_variable("biases", [num_outputs,],
                tf.float32, biases_initilizer, biases_regularizer)
            outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')
        
        if activation_fn:
            outputs = activation_fn(outputs, name='outputs_with_fn')

        logger.debug('[conv2d_%s] %s : %s %s -> %s %s' \
            % (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

        return outputs
