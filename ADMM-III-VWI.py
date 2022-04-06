import tensorflow as tf
import numpy as np
import h5py
import os
from skimage import io
import time
import scipy.io as sio
from numpy.fft import fft2, ifft2, fftshift


def apply_conv(x, n_out):
    n_in = x.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[3, 3, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        biases = tf.Variable(bias_init_var, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
    return z


def generate_data(x, csm, shuffle=False):
    """Generate a set of random data."""
    n = len(x)
    ind = np.arange(n)
    if shuffle:
        ind = np.random.permutation(ind)
        x = x[ind]
        csm = csm[ind]
        # mask = mask[ind]

    for j in range(0, n, BATCH_SIZE):
        yield x[j:j + BATCH_SIZE], csm[j:j + BATCH_SIZE]


def get_data(x, csm):
    train = np.ndarray([BATCH_SIZE, n_coil, n_FE, n_PE], dtype=np.complex64)
    label = np.ndarray([BATCH_SIZE, n_FE, n_PE], dtype=np.complex64)
    mask = np.ndarray([BATCH_SIZE, n_coil, n_FE, n_PE], dtype=np.complex64)
    # csm_train = np.ndarray([BATCH_SIZE, n_coil, n_FE, n_PE], dtype=np.complex64)
    scale = np.empty((BATCH_SIZE, 1), dtype='float32')

    mask_coil = np.tile(mk_trn, (n_coil, 1, 1))

    for i in range(BATCH_SIZE):
        mask[i] = mask_coil
        k_label = x[i]
        csm_label = csm[i]

        image = fftshift(ifft2(fftshift(k_label, axes=(-2, -1))), axes=(-2, -1))
        im_label = np.sum(image * np.conjugate(csm_label), 0)

        k_und = fft2(image) * mask_coil
        img_dc = ifft2(k_und)

        scale[i] = np.abs(img_dc).max()
        label[i, :, :] = im_label / scale[i]
        # csm_train[i, :, :, :] = csm_label
        train[i, :, :, :] = k_und / scale[i]

    return train, label, mask, csm


if __name__ == "__main__":
    lr_base = 1e-05
    BATCH_SIZE = 2
    lr_decay_rate = 0.98
    # EPOCHS = 200
    num_epoch = 200
    n_iter = 10

    base_dir = '.'
    # name = 'ADMM_ultimate'
    name = os.path.splitext(os.path.basename(__file__))[0]

    model_save_path = os.path.join(base_dir, 'models')
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    checkpoint_path = os.path.join(model_save_path, '{}.ckpt'.format(name))

    # new_name = 'ADMM-VWI-UIH'
    # new_model_save_path = os.path.join(base_dir, 'models/%s' % new_name)
    # if not os.path.isdir(new_model_save_path):
    #     os.makedirs(new_model_save_path)
    new_ckpt_path = os.path.join(model_save_path, '{}-UIH.ckpt'.format(name))

    # data for train
    # data_dir = './data_DL-VWI'
    # data_dir = '/media/chengjing/Elements/Dynamic data from ziwen'
    # with h5py.File(os.path.join(data_dir, './train_real.h5')) as f:
    #     data_real = f['train_real'][:]
    # data_real = np.transpose(data_real, (0, 1, 3, 2))
    # with h5py.File(os.path.join(data_dir, './train_imag.h5')) as f:
    #     data_imag = f['train_imag'][:]

    # data_dir = '/opt/tf_data/DL_TF/ADMM_1_MRI/new_trndata/'
    # with h5py.File(os.path.join(data_dir, './trnData.h5')) as f:
    #     data_real = f['trn_real'][:]
    #     data_img = f['trn_img'][:]
    # with h5py.File(os.path.join(data_dir, './trnCsm.h5')) as f:
    #     csm_real = f['csm_real'][:]
    #     csm_img = f['csm_img'][:]
    # train_csm = csm_real + 1j * csm_img
    # train_data = data_real + 1j * data_img
    # num_train, n_coil, n_FE, n_PE = train_data.shape
    data_dir = '/media/chengjing/Elements/SIAT/HEAD'
    with h5py.File(os.path.join(data_dir, './UIH_Brain.h5')) as f:
        data_real = f['trnData_real'][:]
        data_img = f['trnData_img'][:]
        mk_trn = f['Mask_2D_VWI'][:]
    with h5py.File(os.path.join(data_dir, './UIH_Brain_csm.h5')) as f:
        csm_real = f['trnCsm_real'][:]
        csm_img = f['trnCsm_img'][:]

    train_csm = csm_real + 1j * csm_img
    train_data = data_real + 1j * data_img
    num, n_coil, n_FE, n_PE = train_data.shape

    mk_trn = np.fft.fftshift(mk_trn, axes=(-1, -2))

    num_train = 800
    # num_validate = 184
    train_data = train_data[0:num_train]
    # validate_data = data[num_train:num_train + num_validate]
    train_csm = train_csm[0:num_train]
    # # validate_csm = csm[num_train:num_train + num_validate]
    # # train_data = np.random.permutation(train_data)
    del data_real, data_img, csm_real, csm_img

    # mask_t = io.imread('PD_256_256_012_5_R6.png', as_gray=True)
    # mask_t = np.fft.fftshift(mask_t, axes=(-1, -2))
    # mk = sio.loadmat(os.path.join(data_dir, './cs_mask.mat'))
    # # mk = sio.loadmat(os.path.join(data_dir, './mask_r4.mat'))
    # mk_trn = np.fft.fftshift(mk['mask'], axes=(-1, -2))
    # mask_t = mk['mask']

    with tf.name_scope('placeholders'):
        x_true = tf.placeholder(tf.complex64, shape=[None, n_FE, n_PE], name="x_true")
        y_m = tf.placeholder(tf.complex64, shape=[None, n_coil, n_FE, n_PE], name="k_train")
        mask = tf.placeholder(tf.complex64, shape=[None, n_coil, n_FE, n_PE], name='mask')
        csm_t = tf.placeholder(tf.complex64, shape=[None, n_coil, n_FE, n_PE], name="csm")

    with tf.name_scope('reconstruction'):
        with tf.name_scope('initial_values'):
            kdata = tf.stack([tf.real(y_m), tf.imag(y_m)], axis=4)
            x = tf.zeros_like(x_true)
            beta = tf.stack([tf.real(x), tf.imag(x)], axis=3)
            z = tf.stack([tf.real(x), tf.imag(x)], axis=3)

        for iter in range(n_iter):
            with tf.variable_scope('DC_layer_{}'.format(iter)):

                x_mc = tf.stack([x for j in range(n_coil)], axis=1)
                Ax = tf.fft2d(x_mc * csm_t) * mask
                evalop_k = tf.stack([tf.real(Ax), tf.imag(Ax)], axis=4)

                update = tf.concat([evalop_k, kdata], axis=-1)
                update = tf.reshape(update, [-1, n_FE, n_PE, 4])

                update = tf.nn.relu(apply_conv(update, n_out=32), name='relu_1')
                # update = tf.nn.relu(apply_conv(update, n_out=32), name='relu_2')
                update = apply_conv(update, n_out=2)
                update = tf.reshape(update, [-1, n_coil, n_FE, n_PE, 2])

                update_cplx = tf.complex(update[..., 0], update[..., 1])
                im_cplx = tf.ifft2d(update_cplx * mask)
                evalop_cplx = tf.reduce_sum(im_cplx * tf.conj(csm_t), axis=1)
                im = tf.stack([tf.real(evalop_cplx), tf.imag(evalop_cplx)], axis=3)

            with tf.variable_scope('recon_layer_{}'.format(iter)):
                v = z - beta
                x_float = tf.stack([tf.real(x), tf.imag(x)], axis=3)
                update = tf.concat([v, x_float, im], axis=-1)

                update = tf.nn.relu(apply_conv(update, n_out=32), name='relu_1')
                update = tf.nn.relu(apply_conv(update, n_out=32), name='relu_2')
                update = apply_conv(update, n_out=2)

                x_float = x_float + update

            with tf.variable_scope('denoise_layer_{}'.format(iter)):
                update = tf.nn.relu(apply_conv(x_float + beta, n_out=16), name='relu_1')
                update = tf.nn.relu(apply_conv(update, n_out=16), name='relu_2')
                update = apply_conv(update, n_out=2)
                z = x_float + beta + update

            with tf.variable_scope('update_layer_{}'.format(iter)):
                eta = tf.Variable(tf.constant(1, dtype=tf.float32), name='eta')
                beta = beta + tf.multiply(eta, x_float - z)

                x = tf.complex(x_float[..., 0], x_float[..., 1])

        x_pred = x

    with tf.name_scope("loss"):
        residual_cplx = x_pred - x_true
        residual = tf.stack([tf.real(residual_cplx), tf.imag(residual_cplx)], axis=3)
        loss = tf.reduce_mean(residual ** 2)

    with tf.name_scope('optimizer'):
        # Learning rate
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = lr_base
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=num_train / BATCH_SIZE,
                                                   decay_rate=0.95,
                                                   name='learning_rate')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                              beta2=0.99)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
            optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                                 global_step=global_step)

    # sess = tf.InteractiveSession()
    with tf.Session() as sess:
        # Initialize all TF variables
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # train_plot = []
        # validate_plot = []

        # train the network
        
        for i in range(num_epoch):
            count_train = 0
            loss_sum_train = 0.0
            for ys, csm_train in generate_data(train_data, train_csm, shuffle=True):
                train, label, mask_d, csm_train = get_data(ys, csm_train)
                im_start = time.time()
                _, loss_value, step, pred = sess.run([optimizer, loss, global_step, x_pred],
                                                     feed_dict={x_true: label,
                                                                y_m: train,
                                                                mask: mask_d,
                                                                csm_t: csm_train})
                im_end = time.time()
                loss_sum_train += loss_value
                print("{}\{}\{} of training loss:\t\t{:.6f} \t using :{:.4f}s".
                      format(i + 1, count_train + 1, int(num_train / BATCH_SIZE),
                             loss_sum_train / (count_train + 1), im_end - im_start))
                count_train += 1

            # # validating and get train loss
            # count_train_per = 0
            # loss_sum_train = 0.0
            # for ys_train in generate_data(train_data, shuffle=True):
            #     y_arr_train, x_true_train, mask_train = get_data(ys_train)
            #     im_start = time.time()
            #     loss_value_train = sess.run(loss, feed_dict={y_m: y_arr_train,
            #                                                  mask: mask_train,
            #                                                  x_true: x_true_train})
            #     im_end = time.time()
            #     loss_sum_train += loss_value_train
            #     count_train_per += 1
            #     print("{}\{}\{} of train loss (just get loss):\t\t{:.6f} \t using :{:.4f}s"
            #           .format(i + 1, count_train_per, int(num_train / BATCH_SIZE),
            #                   loss_sum_train / count_train_per, im_end - im_start))
            #
            # # get validation loss
            # count_validate = 0
            # loss_sum_validate = 0.0
            # for ys_validate in generate_data(validate_data, shuffle=True):
            #     y_rt_validate, x_true_validate, mask_validate = get_data(ys_validate)
            #     im_start = time.time()
            #     loss_value_validate = sess.run(loss, feed_dict={y_m: y_rt_validate,
            #                                                     mask: mask_validate,
            #                                                     x_true: x_true_validate})
            #     im_end = time.time()
            #     loss_sum_validate += loss_value_validate
            #     count_validate += 1
            #     print("{}\{}\{} of validation loss:\t\t{:.6f} \t using :{:.4f}s".
            #           format(i + 1, count_validate, int(num_validate / BATCH_SIZE),
            #                  loss_sum_validate / count_validate, im_end - im_start))
            #
            # train_plot.append(loss_sum_train / count_train_per)
            # validate_plot.append(loss_sum_validate / count_validate)

            # if i > 0 and (i + 1) % 10 == 0:
            #     saver.save(sess, os.path.join(model_save_path, '%s_epoch%d.ckpt' % (name, i+1)))
            saver.save(sess, new_ckpt_path)
        # train_plot_name = 'train_plot.npy'
        # np.save(os.path.join(checkpoint_dir, train_plot_name), train_plot)
        # validate_plot_name = 'validate_plot.npy'
        # np.save(os.path.join(checkpoint_dir, validate_plot_name), validate_plot)
