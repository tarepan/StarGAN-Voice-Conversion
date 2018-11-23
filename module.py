import tensorflow as tf


def gated_linear_layer(inputs, gates, name=None):

    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


# Why not BN but IN? original article use BN (IN is for CycleGAN-VC)
def instance_norm_layer(inputs, epsilon=1e-05, activation_fn=None, name=None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs, center=True, scale=True, epsilon=epsilon, activation_fn=activation_fn, scope=name)

    return instance_norm_layer


def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None, kernel_initializer=None, name=None):

    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p, name=name + 'conv2d_pad')

    conv_layer = tf.layers.conv2d(
        inputs=out,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid',
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


# GLU + Conv2d + IN
def downsample2d_block(inputs, filters, kernel_size, strides, padding: list = None, name_prefix='downsample2d_block_'):

    h1 = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    return h1_glu


# Used only for unsampling in Generator
def upsample2d_block(inputs, filters, kernel_size, strides, name_prefix='upsample2d_block_'):
    t1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    t2 = tf.contrib.layers.instance_norm(t1, scope=name_prefix + 'instance1')

    x1_gates = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    x1_norm_gates = tf.contrib.layers.instance_norm(x1_gates, scope=name_prefix + 'instance2')

    x1_glu = gated_linear_layer(t2, x1_norm_gates)
    return x1_glu


# Generator
def generator_gatedcnn(inputs, speaker_id=None, reuse=False, scope_name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #downsample
        d1 = downsample2d_block(inputs, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='down_1')
        d2 = downsample2d_block(d1, filters=64, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_2')
        d3 = downsample2d_block(d2, filters=128, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_3')
        d4 = downsample2d_block(d3, filters=64, kernel_size=[3, 5], strides=[1, 1], padding=[1, 2], name_prefix='down_4')
        d5 = downsample2d_block(d4, filters=5, kernel_size=[9, 5], strides=[9, 1], padding=[1, 2], name_prefix='down_5')

        #upsample
        speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        print(c.shape.as_list())
        concated = tf.concat([d5, c], axis=-1)
        # print(concated.shape.as_list())

        u1 = upsample2d_block(concated, 64, kernel_size=[9, 5], strides=[9, 1], name_prefix='gen_up_u1')
        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        u1_concat = tf.concat([u1, c1], axis=-1)
        u2 = upsample2d_block(u1_concat, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        u2_concat = tf.concat([u2, c2], axis=-1)
        u3 = upsample2d_block(u2_concat, 64, [4, 8], [2, 2], name_prefix='gen_up_u3')
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        u3_concat = tf.concat([u3, c3], axis=-1)
        u4 = upsample2d_block(u3_concat, 32, [4, 8], [2, 2], name_prefix='gen_up_u4')
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        u4_concat = tf.concat([u4, c4], axis=-1)
        u5 = tf.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same', name='generator_last_deconv')(u4_concat)

        return u5


def discriminator(inputs, speaker_id, reuse=False, scope_name='discriminator'):

    # inputs has shape [batch_size, height,width, channels]

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        #convert data type to float32
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inputs.shape[1], inputs.shape[2], 1])

        concated = tf.concat([inputs, c], axis=-1)

        # Downsample
        d1 = downsample2d_block(
            inputs=concated, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='downsample2d_dis_block1_')
        c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = downsample2d_block(
            inputs=d1_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block2_')
        c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)

        d3 = downsample2d_block(
            inputs=d2_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block3_')
        c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = downsample2d_block(
            inputs=d3_concat, filters=32, kernel_size=[3, 6], strides=[1, 2], padding=[1, 2], name_prefix='downsample2d_diss_block4_')
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1], name='discriminator-last-conv')

        c1_red = tf.reduce_mean(c1, keepdims=True)

        return c1_red

# Why not strided-Conv but max-pooling? In G & D, you use strided-Conv
def domain_classifier(inputs, reuse=False, scope_name='classifier'):

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #   add slice input shape [batchsize, 8, 512, 1]
        #get one slice
        one_slice = inputs[:, 0:8, :, :]

        d1 = tf.layers.conv2d(one_slice, 8, kernel_size=[4, 4], padding='same', name=scope_name + '_conv2d01')
        d1_p = tf.layers.max_pooling2d(d1, [2, 2], strides=[2, 2], name=scope_name + 'p1')

        d2 = tf.layers.conv2d(d1_p, 16, [4, 4], padding='same', name=scope_name + '_conv2d02')
        d2_p = tf.layers.max_pooling2d(d2, [2, 2], strides=[2, 2], name=scope_name + 'p2')

        d3 = tf.layers.conv2d(d2_p, 32, [4, 4], padding='same', name=scope_name + '_conv2d03')
        d3_p = tf.layers.max_pooling2d(d3, [2, 2], strides=[2, 2], name=scope_name + 'p3')

        d4 = tf.layers.conv2d(d3_p, 16, [3, 4], padding='same', name=scope_name + '_conv2d04')
        d4_p = tf.layers.max_pooling2d(d4, [1, 2], strides=[1, 2], name=scope_name + 'p4')

        d5 = tf.layers.conv2d(d4_p, 4, [1, 4], padding='same', name=scope_name + '_conv2d05')
        d5_p = tf.layers.max_pooling2d(d5, [1, 2], strides=[1, 2], name=scope_name + 'p5')

        p = tf.keras.layers.GlobalAveragePooling2D()(d5_p)

        o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])

        return o_r
