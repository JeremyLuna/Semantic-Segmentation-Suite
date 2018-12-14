# this network will use simple convolution only
import tensorflow as tf
import tensorflow.contrib.slim as slim

def base_conv2d(inputs, shape, non_linearity):
    layer = inputs
    filter = tf.Variable(tf.random.uniform(shape, minval=-1, maxval=1))
    layer = tf.nn.conv2d(layer, filter, [1, 1, 1, 1], 'SAME')
    if non_linearity:
        layer = tf.nn.bias_add(layer, tf.Variable(tf.random.uniform([shape[-1]], minval=0, maxval=1)))
        layer = tf.nn.relu(layer)
    return layer

def lrn_conv2d(inputs, shape, non_linearity):
    layer = inputs
    layer = tf.nn.lrn(layer, int(shape[0]/2))
    filter = tf.Variable(tf.random.uniform(shape, minval=-1, maxval=1))
    layer = tf.nn.conv2d(layer, filter, [1, 1, 1, 1], 'SAME')
    if non_linearity:
        layer = tf.nn.bias_add(layer, tf.Variable(tf.random.uniform([shape[-1]], minval=0, maxval=1)))
        layer = tf.nn.relu(layer)
    return layer

# This is the experimental layer
# local response feature (as opposed to local response norm)
# this is a mathematically accurate way of generalizing
# l2 norm spatially
def lrf_conv2d(inputs, shape, non_linearity):
    layer = inputs
    filter = tf.Variable(tf.random.uniform(shape, minval=-1, maxval=1))
    conv = tf.nn.conv2d(layer, filter, [1, 1, 1, 1], 'SAME')
    norm_f = tf.norm(tf.reshape(filter, [-1, shape[-1]]), axis=0)
    local_norm_im = tf.sqrt(tf.nn.conv2d(tf.square(inputs),
                                         tf.ones([shape[0], shape[1], shape[2], 1]),
                                         [1, 1, 1, 1],
                                         'SAME'))
    divisor = local_norm_im * norm_f
    layer = conv/divisor
    layer = tf.tanh(layer)
    if non_linearity:
        layer = tf.nn.bias_add(layer, tf.Variable(tf.random.uniform([shape[-1]], minval=0, maxval=1)))
        layer = tf.nn.relu(layer)
    return layer

def build_test_net(inputs, num_classes, layer_type, non_linearity):
    net = inputs

    if layer_type == "base_conv2d":
        layer_to_test = base_conv2d
    elif layer_type == "lrf_conv2d":
        layer_to_test = lrf_conv2d
    elif layer_type == "lrn_conv2d":
        layer_to_test = lrn_conv2d
    else:
        print('invalid layer type')

    net = layer_to_test(net, [8, 8, 3, 32], non_linearity)
    # put net here

    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net
