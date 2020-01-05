from tfGAN_indvBN import batcher, lrelu, safe_log, init_normal
import numpy as np
import tensorflow as tf
from os import sep, getcwd
from time import time
from os.path import join, exists

BATCH_NORM_DECAY = 0.999
BATCH_RENORM = False

class GANBase (object):

    name = 'GANBase'

    def __init__(self, n_extra_generator_layers=0, n_extra_discriminator_layers=0, mask=None, use_batch_norm_G=True,
                 use_batch_norm_D=False, name=None, log_and_save=True, seed=np.random.randint(int(1e8)), debug=False):
        # parameters
        self.n_noise = 100
        self.n_pixel = 32
        self.n_channel = 3
        if mask is None or mask.dtype is not bool:
            self.mask = np.ones(self.n_pixel * self.n_pixel, dtype=bool)
            self.mask[mask] = False
        else:
            self.mask = tf.constant(mask.reshape(), dtype=tf.float32, name='mask')
        self.mask = tf.constant(self.mask.reshape(1, self.n_pixel, self.n_pixel, 1), dtype=tf.float32, name='mask')
        self.batch_norm_G = use_batch_norm_G
        self.batch_norm_D = use_batch_norm_D
        self.seed = seed
        self.n_extra_generator_layers = n_extra_generator_layers
        self.n_extra_discriminator_layers = n_extra_discriminator_layers
        self.log_and_save = log_and_save
        self.debug = debug
        self.filename = self.name
        if name is not None:
            self.name += '_' + name
        if self.debug:
            self.name += '_debug'
        self.path = getcwd() + sep + 'output' + sep + self.filename + sep

        # network variables
        self.batch_ind = tf.placeholder(tf.int32, 0, 'batch_ind')
        self.batch_size = tf.placeholder(tf.int32, 0, 'batch_size')
        self.training = tf.placeholder(tf.bool,None, 'training')
        self.input_x = tf.placeholder(tf.float32, (None, self.n_pixel, self.n_pixel, self.n_channel), 'image')
        self.input_n = tf.placeholder(tf.float32, (None, self.n_noise), 'noise')
        # self.input_x = tf.Variable(self.input_x_ph, trainable=False, collections=[])
        # self.input_n = tf.Variable(self.input_n_ph, trainable=False, collections=[])

        # logging'
        self.saver = None
        self.writer_train = None
        self.writer_test = None

        # etc
        self.session = None

    def _build_generator(self, tensor=None, training=False, batch_norm=None):
        assert self.n_pixel % 16 == 0, "isize has to be a multiple of 16"
        nfilt = 2000
        csize = 4
        if tensor is None:
            tensor = self.input_n
        if batch_norm is None:
            batch_norm = self.batch_norm_G
        if batch_norm:
            def bn(x, name=None):
                return tf.contrib.layers.batch_norm(x, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        with tf.variable_scope('generator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # initial layer
            with tf.variable_scope('initial.{0}-{1}'.format(self.n_noise, nfilt)):
                tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tf.reshape(tensor, [-1, 1, 1, self.n_noise]),
                                                                  nfilt, 4, 2, 'valid', use_bias=not batch_norm,
                                                                  kernel_initializer=init_normal(),
                                                                  name='conv'), name='bn'))

            # upscaling layers
            while csize < self.n_pixel//2:
                with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt/2)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt//2, 4, 2, 'same',
                                                                      use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))
                csize *= 2
                nfilt //= 2

            # extra layers
            for it in range(self.n_extra_generator_layers):
                with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                    tensor = tf.nn.relu(bn(tf.layers.conv2d_transpose(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                                      kernel_initializer=init_normal(),
                                                                      name='conv'), name='bn'))

            # final layer
            with tf.variable_scope('final.{0}-{1}'.format(nfilt, self.n_channel)):
                tensor = tf.layers.conv2d_transpose(tensor, self.n_channel, 4, 2, 'same', activation=tf.tanh,
                                                    kernel_initializer=init_normal(),
                                                    name='conv')

            # mask layer
            return tensor

    def _build_discriminator_base(self, tensor=None, training=False, batch_norm=None):
        nfilt = 500
        if tensor is None:
            tensor = self.input_x
        if batch_norm is None:
            batch_norm = self.batch_norm_D
        if batch_norm:
            def bn(tensor, name=None):
                return tf.contrib.layers.batch_norm(tensor, is_training=training,
                                                    renorm=BATCH_RENORM, decay=BATCH_NORM_DECAY)
        else:
            bn = tf.identity

        # initial layer
        with tf.variable_scope('initial.{0}-{1}'.format(self.n_channel, nfilt)):
            tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 4, 2, 'same', use_bias=not batch_norm,
                                               kernel_initializer=init_normal(),
                                               name='conv'), name='bn'))
        nfilt /= 2
        csize = self.n_pixel / 2

        # extra layers
        for it in range(self.n_extra_discriminator_layers):
            with tf.variable_scope('extra-{0}.{1}'.format(it, nfilt)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt, 3, 1, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))

        # downscaling layers
        while csize > 4:
            with tf.variable_scope('pyramid.{0}-{1}'.format(nfilt, nfilt * 2)):
                tensor = lrelu(bn(tf.layers.conv2d(tensor, nfilt*2, 4, 2, 'same', use_bias=not batch_norm,
                                                   kernel_initializer=init_normal(),
                                                   name='conv'), name='bn'))
            nfilt *= 2
            csize /= 2

        return tensor

    def _build_loss(self, label_strength=1.):
        raise NotImplementedError

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


class DCGAN (GANBase):

    name = 'DCGAN'

    def _build_discriminator(self, tensor=None, training=False):

        with tf.variable_scope('discriminator') as scope:
            # set reuse if necessary
            if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name):
                scope.reuse_variables()

            # discriminator base
            tensor = self._build_discriminator_base(tensor, training)

            # final layer
            d_out = 2
            with tf.variable_scope('final.{0}-{1}'.format(tensor.shape[-1], d_out)):
                out_logits = tf.reshape(tf.layers.conv2d(tensor, d_out, 4, 2, 'valid', kernel_initializer=init_normal(),
                                                         name='conv'), [-1, d_out])

        return tf.nn.softmax(out_logits), out_logits

    def _build_loss(self, label_strength=1., training=False):
        fake = self._build_generator(training=training) #tf.random_normal((self.batch_size, self.n_noise)))

        fake_label, fake_logits = self._build_discriminator(fake, training=training)
        real_label, real_logits = self._build_discriminator(training=training)
        label_goal = tf.concat((tf.ones((tf.shape(fake_logits)[0], 1)), tf.zeros((tf.shape(fake_logits)[0], 1))), 1)
        label_smooth = tf.concat((label_strength * tf.ones((tf.shape(fake_logits)[0], 1)),
                                  (1 - label_strength) * tf.ones((tf.shape(fake_logits)[0], 1))), 1)

        # generator
        # self.lossG =
        # -safe_log(1 - fake_label[:, -1]) or -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        # -safe_log(fake_label[:, 0]) (better) or tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal) (best)
        lossG = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=label_goal))

        # discriminator
        lossD_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=real_logits, labels=label_smooth))
        lossD_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fake_logits, labels=1-label_goal))
        lossD = lossD_d + lossD_g

        # summaries
        if training:
            tf.summary.image('fake', fake)
            tf.summary.image('real', self.input_x)
            tf.summary.histogram('D_fake', fake_label[:, -1])
            tf.summary.histogram('D_real', real_label[:, -1])
            tf.summary.scalar('lossG', lossG)
            tf.summary.scalar('lossD_d', lossD_d)
            tf.summary.scalar('lossD_g', lossD_g)
            tf.summary.scalar('lossD', lossD)
            tf.summary.scalar('loss', lossG + lossD)

        return lossG, lossD

    def train(self, trainx, testx, n_epochs=2, n_batch=128, learning_rate=2e-4, label_strength=1.):

        # handle data
        n_train = trainx.shape[0]
        n_test = testx.shape[0]
        # train = tf.constant(trainx, name='train')
        # test = tf.constant(testx, name='test')
        # dataset = tf.contrib.data.Dataset.from_tensor_slices(self.input_x)
        # iterator = dataset.make_initializable_iterator()
        # train = tf.contrib.data.Dataset.from_tensor_slices(trainx)
        # train = tf.contrib.data.Dataset.from_tensor_slices(testx)
        # iterator_train = train.make_initializable_iterator()

        # setup learning
        # train_batch = tf.train.shuffle_batch([train], n_batch, 50000, 10000, 2,
        #                                      enqueue_many=True, allow_smaller_final_batch=True, name='batch')
        global_step = tf.train.get_or_create_global_step(graph=None)
        lossG, lossD = self._build_loss(label_strength=label_strength, training=True)
        evalG, evalD = self._build_loss(label_strength=label_strength)
        tvarsG = [var for var in tf.trainable_variables() if 'generator' in var.name]
        tvarsD = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adamG = tf.contrib.layers.optimize_loss(loss=lossG,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer='SGD',
                                                    clip_gradients=20.0,
                                                    name='optG',
                                                    variables=tvarsG)
            adamD = tf.contrib.layers.optimize_loss(loss=lossD,
                                                    global_step=global_step,
                                                    learning_rate=learning_rate,
                                                    optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                    clip_gradients=20.0,
                                                    name='optD',
                                                    variables=tvarsD)

        # summary
        merged_summary = tf.summary.merge_all()

        # start session
        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})
            # sess.run(self.input_x.initializer, {self.input_x_ph: trainx})

            # train
            self._start_logging_and_saving(sess)
            for epoch in range(n_epochs):
                # train on epoch
                start = time()
                n, lg, ld = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_train, n_batch):
                    n += n_batch_actual
                    # discriminator
                    temp = sess.run(adamD,
                                    {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                     self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    ld += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}D):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    # generator
                    temp, summary, step = sess.run([adamG,merged_summary, global_step],
                                                   {self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lg += temp * n_batch_actual
                    # self._log(summary, step)
                    # print 'epoch {:d}/{:d} (part {:d}G):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds'\
                    #     .format(epoch, n_epochs, n, (lg + ld)/n, lg/n, ld/n, int(time() - start))
                    if n % (10 * n_batch) == 0:
                        self._log(summary, step)
                        print('epoch {:d}/{:d} (part {:d}):  training loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
                            .format(epoch + 1, n_epochs, n, (lg + ld) / n, lg / n, ld / n, int(time() - start)))
                # save after each epoch
                self._save(sess, step)

                # evaluate
                n, lge, lde = 0, 0, 0
                for batch_index, n_batch_actual in batcher(n_test, n_batch):
                    n += n_batch_actual
                    out = sess.run([evalG, evalD],
                                   {self.training: [False],
                                    self.input_x: trainx[batch_index:batch_index + n_batch_actual],
                                    self.input_n: np.random.randn(n_batch_actual, self.n_noise).astype(np.float32)})
                    lge += out[0] * n_batch_actual
                    lde += out[1] * n_batch_actual
              #  print 'epoch {:d}/{:d}:  evaluation loss: {:f} (G: {:f}  D: {:f})  time: {:d} seconds' \
              #      .format(epoch + 1, n_epochs, (lge + lde) / n, lge / n, lde / n, int(time() - start))

if __name__ == '__main__':
    from data_process import load_cifar10

    labeled_data, labeled_label, unlabeled_data, test_data, test_label = load_cifar10()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_boolean('debug', False, 'activate tfdb')
    tf.app.flags.DEFINE_integer('ep', 2, 'training epochs')
    debug=FLAGS.debug
    epochs=FLAGS.ep

    gan_test = DCGAN()
    gan_test.train(unlabeled_data, labeled_label,
                     n_batch=100, label_strength=0.9)