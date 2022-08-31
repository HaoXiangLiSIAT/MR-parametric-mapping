
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import h5py
import os
import datetime
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
        # bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        # biases = tf.Variable(bias_init_var, trainable=True, name='b')
        # z = tf.nn.bias_add(conv, biases)
    return conv


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
    scale = np.empty((BATCH_SIZE, 1), dtype='float32')

    mask_coil = np.tile(mk_trn, (n_coil, 1, 1))

    for i in range(BATCH_SIZE):
        mask[i] = mask_coil
        k_label = x[i]
        csm_label = csm[i]
        image = fftshift(ifft2(fftshift(k_label, axes=(-2, -1))), axes=(-2, -1))
        im_label = np.sum(image * np.conjugate(csm_label), 0)
        k_und = fft2(image) * mask_coil  # e-1
        img_dc = ifft2(k_und)
        scale[i] = np.abs(img_dc).max()   ### 5e-5
        label[i, :, :] = im_label
        train[i, :, :, :] = k_und
    return train, label, mask, csm


if __name__ == "__main__":
    # lr_base = 1e-03
    lr_base = 3e-04
    BATCH_SIZE = 5
    lr_decay_rate = 0.98
    # EPOCHS = 200
    num_epoch = 70
    n_iter = 12
    startnum = 0
    #4800
    #slices = 100
    slices = 4800
    num_validate = slices
    train_plot=[]
    validate_plot=[]
    ##########################
    #######Noise Robust
    ##########################
    # for i in range(train_data.shape[0]):
    #     train_data_slice = train_data[i][:][:][:]
    #     image = fftshift(ifft2(fftshift(train_data_slice, axes=(-2, -1))), axes=(-2, -1))
    #     sigma = 300
    #     # print(np.mean(np.abs(image)))
    #     img_real = np.real(image)
    #     img_imag = np.imag(image)
    #     noise = sigma * np.random.standard_normal(img_real.shape)
    #     noise = noise - np.mean(noise)
    #     img_real = noise + img_real
    #     noise = sigma * np.random.standard_normal(img_imag.shape)
    #     noise = noise - np.mean(noise)
    #     img_imag = noise + img_imag
    #     img_noise = img_real + 1j * img_imag
    #     # image = np.abs(image)
    #     # img_noise = np.abs(img_noise)
    #     k_noise = fftshift(fft2(fftshift(img_noise, axes=(-2, -1))), axes=(-2, -1))
    #     train_data[i][:][:][:] = k_noise

    #train_data = train_data/1000
### for brain:
    with h5py.File('/brain/train_kspace.h5') as f:
        data_real = f['kspace_real']
        print('data_real:',data_real.shape)
        data_real = f['kspace_real'][startnum:slices]
        data_img = f['kspace_imag'][startnum:slices]
        # mk_trn = f['Mask_2D_VWI'][:]
    with h5py.File('/brain/train_csm.h5') as f:
        csm_real = f['csm_real'][startnum:slices]
        csm_img = f['csm_imag'][startnum:slices]

    data_real = data_real / 2000
    data_img = data_img / 2000
    data_real = data_real * 10000
    data_img = data_img * 10000

    ### for new brain data:
    data_real = data_real / 1000
    data_img = data_img / 1000

    train_csm = csm_real + 1j * csm_img
    train_data = data_real + 1j * data_img
    num, n_coil, n_FE, n_PE = train_data.shape

    # brain mask
    mask = h5py.File('./mask/mask7_6_brain.mat', 'r')['mask'][1]


    mk_trn = np.transpose(mask)
    mk_trn = np.fft.fftshift(mk_trn, axes=(-1, -2))

    num_train = slices - startnum
    # num_validate = 100
    train_data = train_data[0:num_train]
    train_csm = train_csm[0:num_train]




    name = 'ADMM-VWI2'
    base_dir = '.'
    model_save_path = os.path.join(base_dir, 'models/%s' % name)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    time_now = datetime.datetime.now()
    time_now = str(time_now)
    date = time_now[0:13] + time_now[14:16]
    checkpoint_dir = os.path.join(model_save_path, 'checkpoints/data_%s' % num_train)
    checkpoint_dir = checkpoint_dir + ' ' + date
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.ckpt'.format(name))


    new_ckpt_path = os.path.join(model_save_path, '{}-UIH.ckpt'.format(name))
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
    with tf.Session() as sess:
        # Initialize all TF variables
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # saver.save(sess, checkpoint_path)
        # saver.restore(sess, checkpoint_path)
        # train_plot = []
        # validate_plot = []

        # train the network

        for i in range(num_epoch):
            print('****************epoch{:02d}************'.format(i))
            count_train = 0
            loss_sum_train = 0.0
            index = 0
            for ys, csm_train in generate_data(train_data, train_csm, shuffle=False):
                train, label, mask_d, csm_train = get_data(ys, csm_train)
                im_start = time.time()
                _, loss_value, step, pred = sess.run([optimizer, loss, global_step, x_pred],
                                                     feed_dict={x_true: label,
                                                                y_m: train,
                                                                mask: mask_d,
                                                                csm_t: csm_train})

                index += 1
                if i == 10:
                    if index == 1:
                        pred_m = np.transpose(pred)
                        label_m = np.transpose(label)
                        # mask_d_m = np.transpose(mask_t)
                    else:
                        pred_m = np.concatenate((pred_m,np.transpose(pred)),axis=-1)
                        label_m = np.concatenate((label_m,np.transpose(label)),axis=-1)
                    # train = np.transpose(train)
                    #if i % 10 ==0:
                    if index == (slices - startnum) / BATCH_SIZE:
                        sio.savemat('./outputs/output.mat', {'output': pred_m})
                        sio.savemat('./outputs/label.mat',{'label': label_m})
                        print('result saved')


                im_end = time.time()
                loss_sum_train += loss_value
                # print("{}\{}\{} of training loss:\t\t{:.10f} \t using :{:.4f}s".
                #       format(i + 1, count_train + 1, int(num_train / BATCH_SIZE),
                #              loss_sum_train / (count_train + 1), im_end - im_start))
                print("{}\{}\{} of training loss:\t\t{:.10f} \t using :{:.4f}s".
                      format(i + 1, count_train + 1, int(num_train / BATCH_SIZE),
                             loss_value, im_end - im_start))
                count_train += 1
            #saver.save(sess,'./ADMM_model/',global_step = i)
            print('*****************************Save Model *************************')
            saver.save(sess, checkpoint_path)
            #################################
            # # validating and get train loss
            ################################
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
            # get validation loss
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
            # checkpoint_path = os.path.join(checkpoint_dir, 'epoch%d' % (i + 1))
            # if not os.path.exists(checkpoint_path):
            #     os.makedirs(checkpoint_path)
            # checkpoint_path = checkpoint_path + '/{}.ckpt'.format(name)
            #saver.save(sess, checkpoint_path,step = epoch)
        # train_plot_name = 'train_plot.npy'
        # np.save(os.path.join(checkpoint_dir, train_plot_name), train_plot)
        # validate_plot_name = 'validate_plot.npy'
        # np.save(os.path.join(checkpoint_dir, validate_plot_name), validate_plot)
