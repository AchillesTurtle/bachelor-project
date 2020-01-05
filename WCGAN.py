import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import cond_batchnorm
import numpy as np
from os import sep, getcwd
from time import time
from os.path import join
from tensorflow.python import debug as tf_debug


def batcher(n_data, batch_size):
    for ind in range(0, n_data - batch_size + 1, batch_size):
        # batch starting index, actual batch size
        yield ind, np.minimum(batch_size, n_data - ind)


def get_one_hot(targets, depth):
    if targets.ndim>1 and targets.shape[-1]==depth:
        return targets
    return np.eye(depth)[np.array(targets).reshape(-1)]


def gaussian_noise(input_layer, std=0.15):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def init_normal(std=0.02):
    return tf.random_normal_initializer(0, std, None, tf.float32)


class WCGAN(object):
    name = 'WCGAN'

    def __init__(self, name=None, log_and_save=True, debug=False):
        self.conditional_bn=True
        self.n_noise = 100
        self.n_pixel = 32
        self.n_channel = 1
        self.n_class = 10
        self.batch_norm_G = True
        self.batch_norm_D = False  # gp must be false
        self.log_and_save = log_and_save
        self.debug = debug
        self.filename = self.name
        if name is not None:
            self.name += '_' + name
        if self.debug:
            self.name += '_debug'
        self.path = getcwd() + sep + 'output' + sep + self.filename + sep

        # network variables
        self.training = tf.placeholder(tf.bool, None, 'training')
        self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        self.input_y = tf.placeholder(tf.float32, (None, self.n_class), 'labeled_class')
        self.input_z = tf.placeholder(tf.float32, (None, self.n_noise), 'input_z')
        self.input_z_y = tf.placeholder(tf.float32, (None, self.n_class), 'self_chosen_class')

    def _start_logging_and_saving(self, sess, log=True, save=True):
        if self.log_and_save and (log or save):
            # saver to save model
            if save:
                self.saver = tf.train.Saver()
            # summary writer
            if log:
                self.writer_train = tf.summary.FileWriter(join(self.path, self.name, 'train'), sess.graph)
                self.writer_test = tf.summary.FileWriter(join(self.path, self.name, 'test'), sess.graph)

            print('Saving to ' + self.path)

    def _log(self, summary, counter=None, test=False):
        if self.log_and_save:
            if test:
                self.writer_test.add_summary(summary, counter)
            else:
                self.writer_train.add_summary(summary, counter)

    def _save(self, session, counter=None):
        if self.log_and_save:
            self.saver.save(session, join(self.path, self.name, self.name + '.ckpt'), counter)

    def _restore(self, session):
        if self.log_and_save:
            self.saver.restore(session, tf.train.latest_checkpoint(join(self.path, self.name)))

    def load(self, path=None):
        self._build_loss()
        self.session = tf.Session()
        self._start_logging_and_saving(None, log=False)
        if path is None:
            path = tf.train.latest_checkpoint(join(self.path, self.name))
        self.saver.restore(self.session, path)

    def _build_generator_contrib(self, tensor=None,labels=None, training=False):
        if tensor is None:
            print('NONE VALUE INPUT TO G.')
            raise
        if self.batch_norm_G:
            def bn(x,axes=[0,1,2], name=None):
                if self.conditional_bn is True:
                    return cond_batchnorm.Batchnorm(axes=axes,inputs=x,labels=labels,n_labels=self.n_class)
                else:
                    return tf.layers.batch_normalization(x, fused=False, training=training)
        else:
            bn = tf.identity
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
            n_filt = 512//2
            use_bias = True
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, n_filt)):
                tensor = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(tf.reshape(tensor, [-1, self.n_noise+self.n_class]), units=4 * 4 * n_filt,
                                                       kernel_initializer=init_normal(0.02), use_bias=use_bias,
                                                       name='dense'),training=training))
                tensor = tf.reshape(tensor, shape=tf.stack([-1, 4, 4, n_filt]))

            # upscaling layers
            for layers in range(2):
                with tf.variable_scope('pyramid.{0}-{1}'.format(n_filt, n_filt // 2)):
                    tensor = tf.nn.relu(
                        bn(tfcl.conv2d_transpose(tensor, n_filt // 2, [3, 3], [2, 2], 'SAME', activation_fn=None,
                                                 weights_initializer=init_normal(0.02), biases_initializer=None),
                           name='bn'))
                n_filt //= 2
            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(n_filt, self.n_channel)):
                tensor = tfcl.conv2d_transpose(tensor, self.n_channel, [3, 3], [2, 2], 'SAME',
                                               activation_fn=tf.tanh,
                                               weights_initializer=init_normal(0.02), biases_initializer=None)
            return tensor

    def _build_generator(self, tensor=None,label=None, training=False):
        if tensor is None:
            print('NONE VALUE INPUT TO G.')
            raise
        if self.batch_norm_G:
            def bn(x, name=None):
                return tf.layers.batch_normalization(x, fused=False, training=training)
        else:
            bn = tf.identity
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
            n_filt = 1024
            use_bias = False
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, n_filt)):
                tensor = bn(tf.layers.dense(tf.reshape(tensor, [-1, (self.n_noise+self.n_class)]), units=4 * 4 * n_filt,
                                            kernel_initializer=init_normal(0.01), use_bias=False,
                                            name='dense'), name='bn')
                tensor = tf.reshape(tf.nn.relu(tensor), shape=tf.stack([-1, 4, 4, n_filt]))

            # upscaling layers
            for layers in range(2):
                with tf.variable_scope('pyramid.{0}-{1}'.format(n_filt, n_filt // 2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, n_filt // 2, 2, 2, 'same',
                                                                      use_bias=use_bias,
                                                                      kernel_initializer=init_normal(0.01),
                                                                      name='conv_t'), name='bn'))
                n_filt //= 2
            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(n_filt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 3, 2, 'same', use_bias=use_bias,
                                                    activation=tf.tanh,
                                                    kernel_initializer=init_normal(0.01),
                                                    name='conv_t')
            return tensor

    def _build_discriminator(self, tensor=None, training=False):
        if tensor is None:
            print('NONE VALUE INPUT TO D.')
            raise
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
            n_filt = 64
            use_bias = True
            # if self.batch_norm_D:
            #    def bn(tensor, name=None):
            #        return tf.contrib.layers.batch_norm(tensor, is_training=training)
            # else:
            #    bn = tf.identity
            # initial layer
            with tf.variable_scope('pyramid.{0}-{1}'.format(self.n_channel + self.n_class, n_filt)):
                tensor = tf.nn.leaky_relu(
                    tf.layers.conv2d(tensor, n_filt, 5, 2, 'same', use_bias=use_bias, kernel_initializer=init_normal(),
                                     name='conv'))
                # tensor=tf.contrib.layers.layer_norm(tensor)
            n_filt *= 2
            # downscaling layers
            for layers in range(2):
                with tf.variable_scope('pyramid.{0}-{1}'.format(n_filt // 2, n_filt)):
                    tensor = tf.nn.leaky_relu(tf.layers.conv2d(tensor, n_filt, 5, 2, 'same', use_bias=use_bias,
                                                               kernel_initializer=init_normal(), name='conv'))
                    # tensor=tf.contrib.layers.layer_norm(tensor)
                n_filt *= 2
            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], n_filt)):
                # tensor = tf.nn.leaky_relu(tf.layers.dense(tf.contrib.layers.flatten(tensor), units=1024*3, use_bias=True, name='dense_1024'))
                # ---
                tensor = tf.nn.leaky_relu(
                    tf.layers.conv2d(tensor, n_filt, 5, 2, 'same', use_bias=use_bias, kernel_initializer=init_normal(),
                                     name='conv'))
                #tensor=tf.contrib.layers.layer_norm(tensor)
            with tf.variable_scope('extra_final.{0}-{1}'.format(tensor.shape[-1], 1)):
                tensor = tf.nn.leaky_relu(tf.layers.conv2d(tensor, 1, 4, 1, 'same', use_bias=use_bias, kernel_initializer=init_normal(),
                                     name='conv'))
                tensor = tf.layers.dense(tf.contrib.layers.flatten(tensor), units=1, use_bias=False, name='dense')
            return tensor

    def merge_image(self,tensor):
        for i in range(self.n_class):
            ver=tensor[i*5,0:self.n_pixel,0:self.n_pixel,0:self.n_channel]
            for j in range(1,5):
                ver=tf.concat([ver,tensor[i*5+j,0:self.n_pixel,0:self.n_pixel,0:self.n_channel]],0)
            if i is 0:
                output=ver
            else:
                output=tf.concat([output,ver],1)
        return tf.expand_dims(output,0)

    def _build_loss(self, training=False, penalty_strength=10):
        # networks
        x_shape = self.input_x.get_shape()
        merge_input_x = tf.concat([self.input_x,
                                   tf.tile(tf.reshape(self.input_y, [-1, 1, 1, self.n_class]),
                                           [1, x_shape[1], x_shape[2], 1])], axis=3)
        merge_input_z = tf.concat([self.input_z, self.input_z_y], 1)

        fake = self._build_generator_contrib(merge_input_z,tf.argmax(self.input_z_y,axis=1,output_type=tf.int64), training=training)
        merge_fake=tf.concat([fake,
                                   tf.tile(tf.reshape(self.input_z_y, [-1, 1, 1, self.n_class]),
                                           [1, x_shape[1], x_shape[2], 1])], axis=3)
        #epsilon = tf.reshape(tf.tile(tf.random_uniform((tf.shape(merge_input_z)[0], 1), 0.0, 1.0),
        #                             [1, self.n_pixel * self.n_pixel * (self.n_channel + self.n_class)]),
        #                     [-1, self.n_pixel, self.n_pixel, (self.n_channel + self.n_class)])
        epsilon = tf.random_uniform((tf.shape(fake)[0], 1, 1, 1), 0.0, 1.0)
        semifake = epsilon * merge_input_x + (1 - epsilon) * merge_fake
        fake_logits = self._build_discriminator(merge_fake)
        semifake_logits = self._build_discriminator(semifake)
        real_logits = self._build_discriminator(merge_input_x)

        # generator loss
        # lossG =-tf.reduce_mean(fake_logits)
        lossG = -tf.reduce_mean(fake_logits)*0.1
        # discriminator loss
        with tf.name_scope('define_lossD') as scope:
            lossD_d = -tf.reduce_mean(real_logits)
            lossD_g = tf.reduce_mean(fake_logits)
            norm_gradD = tf.sqrt(
                tf.reduce_sum(tf.square(tf.gradients(semifake_logits, semifake)[0]), reduction_indices=[-1]))
            # norm_gradD = tf.norm(tf.reshape(tf.gradients(semifake_logits, semifake)[0], [tf.shape(semifake)[0], -1]), axis=1)
            lossD_p = penalty_strength * tf.reduce_mean(tf.square(norm_gradD - 1.))
            lossD = lossD_d + lossD_g + lossD_p
        # summaries
        # lossD=tf.Print(lossD,[fake,self.input_x])
        summary_input=tf.concat([tf.random_normal([5*self.n_class, self.n_noise]),
                                tf.convert_to_tensor(np.array(get_one_hot(np.repeat(np.arange(self.n_class),5),self.n_class),dtype=float),dtype=np.float32)],1)
        summary_fake=self._build_generator_contrib(summary_input,tf.convert_to_tensor(np.array(np.repeat(np.arange(self.n_class),5),dtype=int),dtype=tf.int32),training=False)
        if training:
            #tf.summary.image('fake', fake,max_outputs=1)
            tf.summary.image('sum_fake', self.merge_image(summary_fake))
            tf.summary.image('single_fake',summary_fake,max_outputs=1)
            # tf.summary.image('semifake', semifake)
            # tf.summary.image('real', self.input_x,max_outputs=5)
            tf.summary.histogram('real_logits', real_logits)
            tf.summary.histogram('distance', real_logits - fake_logits)
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD_p', lossD_p)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)
        else:
            # should not be here
            raise
        return lossG, lossD

    def train(self, trainx, trainy, testx, testy, n_epochs=25, n_batch=64, learning_rate=1e-4, penalty_strength=10,
              n_updated_d=5):

        # handle data
        n_train = trainx.shape[0]
        n_test = testx.shape[0]

        # setup learning
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(penalty_strength=penalty_strength, training=True)
        # evalG, evalD = self._build_loss(penalty_strength=penalty_stength)
        # tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        # tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        tvarsG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        tvarsD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        print('Generator VARS')
        for var in tvarsG:
            print(var)
        print('\nDiscriminator VARS')
        for var in tvarsD:
            print(var)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print('\nUPDATE OPS')
        for ops in update_ops:
            print(ops)

        print('LEARNING RATE: {0}'.format(learning_rate))
        print('BATCH SIZE: {0}'.format(n_batch))
        decay_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=30000, decay_rate=0.9,staircase=True)
        with tf.control_dependencies(update_ops):
            # adamD = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
            #    .minimize(lossD, var_list=tvarsD)
            # adamG = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
            #    .minimize(lossG, var_list=tvarsG)
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=decay_learning_rate,
                                                    # optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5, beta2=0.9),
                                                    # clip_gradients=20.0,
                                                    update_ops=[],
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=decay_learning_rate,
                                                    # optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5, beta2=0.9),
                                                    # clip_gradients=1e-7,
                                                    update_ops=[],
                                                    name='optD',
                                                    variables=tvarsD)
        # clipD = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in tvarsD]
        # summary
        merged_summary = tf.summary.merge_all()

        check_op = tf.add_check_numerics_ops()
        # start session
        with tf.Session() as sess:
            # initialize variables
            if self.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.run(tf.global_variables_initializer())

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                rand_index = np.random.rand(trainx.shape[0]).argsort()
                np.take(trainx, rand_index, axis=0, out=trainx)
                np.take(trainy, rand_index, axis=0, out=trainy)
                n, lg, ld = 0, 0, 0
                t_index = 1
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    n += n_batch_actual
                    t_index += 1
                    # discriminator
                    # sess.run(clipD)
                    _, temp = sess.run([adamD, lossD],
                                       {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                        self.input_y: get_one_hot(trainy[
                                                                  batch_index:batch_index + n_batch_actual],
                                                                  self.n_class),
                                        self.input_z:  np.random.uniform(-1,1,[n_batch_actual, self.n_noise]).astype(np.float32),
                                        self.input_z_y: get_one_hot(
                                            np.random.randint(self.n_class, size=n_batch_actual),self.n_class)})
                    ld += temp * n_batch_actual
                    if t_index % n_updated_d == 0:
                        # generator
                        _, summary, step, temp = sess.run([adamG, merged_summary, global_step, lossG],
                                                          {self.input_x: trainx[
                                                                         batch_index:batch_index + n_batch_actual],
                                                           self.input_y: get_one_hot(trainy[
                                                                                     batch_index:batch_index + n_batch_actual],
                                                                                     self.n_class),
                                                           self.input_z:  np.random.uniform(-1,1,[n_batch_actual, self.n_noise]).astype(np.float32),
                                                           self.input_z_y: get_one_hot(
                                                               np.random.randint(self.n_class, size=n_batch_actual),self.n_class)})
                        _ = sess.run([check_op,decay_learning_rate], {self.input_x: trainx[
                                                                batch_index:batch_index + n_batch_actual],
                                                  self.input_y: get_one_hot(
                                                      trainy[batch_index:batch_index + n_batch_actual], self.n_class),
                                                  self.input_z: np.random.uniform(-1,1,[n_batch_actual, self.n_noise]).astype(np.float32),
                                                  self.input_z_y: get_one_hot(
                                                      np.random.randint(self.n_class, size=n_batch_actual),self.n_class)})
                        lg += temp * n_batch_actual
                        self._log(summary, step)
                        print('epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                              .format(epoch + 1, n_epochs, n,(lg*n_updated_d + ld) / n, lg / n*n_updated_d, ld / n, int(time() - start)))
                # save after each epoch
                self._save(sess, step)


if __name__ == '__main__':
    from data_process import load_cifar10_all, load_cifar10_rgb, load_celebA,load_mnist,load_fashion

    train_data, train_label, test_data, test_label = load_mnist()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_boolean('debug', False, 'activate tfdb')
    tf.app.flags.DEFINE_integer('ep', 25, 'training epochs')
    debug = FLAGS.debug
    epochs = FLAGS.ep
    wc_gan = WCGAN(debug=debug)
    start = time()
    wc_gan.train(train_data, train_label, test_data, test_label, n_epochs=epochs, n_updated_d=5, learning_rate=1e-4)
