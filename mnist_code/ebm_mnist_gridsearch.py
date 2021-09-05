#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import gc
import os
import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Add, Conv2D, Dense, Layer, Conv2DTranspose, LeakyReLU, Reshape, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean

from helper_functions2 import make_gallery, make_gallery2, plot_loss3, plot_loss5
from sklearn.model_selection import ParameterGrid

directory = "ebmout_1"
parent_dir = "/users/apokkunu/trial/mnist/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
save = True


# In[ ]:

image_shape = (28, 28, 1)
latent_dim = 32

max_epoch = 50
ebm_units = 128
inf_iter_train = 20
inf_iter_val = 100
inf_iter_test = 100

# In[ ]:

####################################################################################

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal([batch, dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def conv_block(x, channels, kernel_size = 3, padding = 'same'):
    x = Conv2D(channels, kernel_size, padding=padding, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)
    return x

def res_block(x, channels, kernel_size = 3):
    input_x = x
    x = conv_block(x, channels, kernel_size = kernel_size)
    x = Conv2D(channels, kernel_size, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = Add()([input_x, x])
    return x

def downsampling_conv_block(x, channels, kernel_size = 4):
    x = ZeroPadding2D()(x)
    x = Conv2D(channels, kernel_size, strides=(2, 2), use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)
    return x

def upsampling_conv_block(x, channels, kernel_size = 3):
    x = Conv2DTranspose(channels, kernel_size, strides=2, padding="same")(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = LeakyReLU()(x)
    return x

def create_encoder(latent_dim, num_layer):
    encoder_iput = Input(shape=image_shape, name='image')
    channels = 32
    x = conv_block(encoder_iput, channels, kernel_size = 4)
    
    print(K.int_shape(x))
    for i in range(num_layer):
        channels *= 2
        x = downsampling_conv_block(x, channels)
        print(K.int_shape(x))
        x = res_block(x, channels)
        
    last_conv_shape = K.int_shape(x)
    x = Flatten()(x)
    
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    model = Model(encoder_iput, [z_mean, z_log_var, z], name='encoder', trainable=False)
    model.trainable = False
    model.load_weights('/users/apokkunu/trial/mnist/output1/128_1_0.0003/en_im_100.h5')
    model.summary()
    return model, last_conv_shape

num_layer = 2
model_encoder, last_conv_shape = create_encoder(latent_dim, num_layer)

# In[]:

def create_decoder(latent_dim, first_conv_shape, num_layer):
    decoder_input = Input(shape=(latent_dim,), name='latent_z')
    x = Dense(first_conv_shape[1] * first_conv_shape[2] * first_conv_shape[3])(decoder_input)
    x = Reshape((first_conv_shape[1], first_conv_shape[2], first_conv_shape[3]))(x)
    
    print(K.int_shape(x))
    channels = first_conv_shape[3]
    
    for i in range(num_layer):
        x = res_block(x, channels)
        channels //= 2
        x = upsampling_conv_block(x, channels)
        print(K.int_shape(x))
    
    x = Conv2D(1, 3, padding='same', use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(x)
    x = Activation('sigmoid', name='rec_image')(x)
    print(K.int_shape(x))
    model = Model(decoder_input, x, name='decoder', trainable=False)
    model.trainable = False
    model.load_weights('/users/apokkunu/trial/mnist/output1/128_1_0.0003/de_im_100.h5')
    model.summary()
    return model

model_decoder = create_decoder(latent_dim, last_conv_shape, num_layer)

####################################################################################


# In[ ]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print("Number of original training examples:", len(x_train), x_train.shape, flush=True)
print("Number of original test examples:", len(x_test), x_test.shape, flush=True)


# In[ ]:


def norm(z):
    return 0.01 * tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(z, 2), axis=1)), axis=1)

def compute_energy(z, orig_z):
    norm_val = norm(z - orig_z)
    energy = ebm(z) + norm_val
    return energy, norm_val

def compute_gradient(z, orig_img):
    with tf.GradientTape() as tape:
        tape.watch(z)
        energy, norm_val = compute_energy(z, orig_img)
    return tape.gradient(energy, z), energy, norm_val

def langevin_inf(z, orig_img, inf_iter, inf_rate, eta, mode, step):
    current_z = z
    
    if mode == 'val':
        z_arr = [current_z]
        
    for i in range(inf_iter):
        gradients, energy, norm_val = compute_gradient(current_z, orig_img)

        # Langevin dynamics
        term1 = 0.5 * inf_rate * gradients
        term2 = eta * tf.random.normal(current_z.get_shape().as_list())
        next_z = current_z - term1 + term2
        current_z = next_z
        
        if step % 100 == 0 and mode == 'train':
            template = "LD Step: {}, Avg. MSE: {}, Avg. energy: {}, Avg. Norm Z: {}"
            mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(model_decoder(orig_img), model_decoder(current_z))).numpy()
            print(template.format(i, mse, tf.reduce_mean(energy).numpy(), tf.reduce_mean(norm_val).numpy()), flush=True)
            
        if mode == 'val':
            z_arr.append(current_z)
    
    if mode == 'train':
        return current_z
    else:
        return current_z, np.array(z_arr)


# In[ ]:


latent_in = Input(shape=(latent_dim,))
x = Dense(ebm_units, activation=tf.nn.softplus)(latent_in)
energy_vals = Dense(1, use_bias=False)(x)
ebm = Model(latent_in, energy_vals, name="EBM")
ebm.summary()


# In[ ]:


class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = np.concatenate((np.random.uniform(0, 1, [size, latent_dim]), np.expand_dims(np.arange(size), axis=1)), axis=1)
        self.maxsize = size
    
    def length(self):
        return len(self.storage)
    
    def return_storage(self):
        return self.storage
    
    def add(self, ims):
        ims = ims.numpy()
        for ind, i in enumerate(ims[:, latent_dim]):
            n = np.equal(self.return_storage()[:, latent_dim], i)
            condition = np.where(n)[0]
            if condition.shape[0] != 0:
                condition = np.repeat(np.transpose(np.expand_dims(n, axis=0)), latent_dim+1, axis=1)
                self.storage = np.where(condition, ims[ind], self.return_storage())
            else:
                self.storage = np.vstack([self.storage[np.random.choice(self.length(), (self.length() - 1), replace=False)], ims[ind]])
    
    def sample_byindex(self, inds):
        temp = []
        for i in inds.numpy():
            try:
                temp.append(self.return_storage()[np.where(np.equal(self.return_storage()[:, latent_dim], i))[0][0], :latent_dim])
            except IndexError:
                temp.append(np.random.uniform(0, 1, [latent_dim]))
        return tf.convert_to_tensor(temp, dtype=tf.float32)

    
# In[ ]:


class EBM(Model):
    def __init__(self, ebm, optimizer, encoder_image, decoder_image):
        super(EBM, self).__init__()
        self.ebm = ebm
        
        self.encoder_image = encoder_image
        self.decoder_image = decoder_image
        
        self.optimizer = optimizer

        self.train_loss_track = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.val_loss_track = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        
        self.re_train = tf.keras.metrics.Mean('re_train', dtype=tf.float32)
        self.lde_train = tf.keras.metrics.Mean('lde_train', dtype=tf.float32)
        
        self.re_val = tf.keras.metrics.Mean('re_val', dtype=tf.float32)
        self.lde_val = tf.keras.metrics.Mean('lde_val', dtype=tf.float32)
        
        self.mse_train = tf.keras.metrics.Mean('mse_train', dtype=tf.float32)
        self.mse_val = tf.keras.metrics.Mean('mse_val', dtype=tf.float32)
        self.mse_batch = tf.keras.metrics.Mean('mse_batch', dtype=tf.float32)
        
        self.train_loss_batch = tf.keras.metrics.Mean('train_loss_batch', dtype=tf.float32)
        self.re_batch = tf.keras.metrics.Mean('re_batch', dtype=tf.float32)
        self.lde_batch = tf.keras.metrics.Mean('lde_batch', dtype=tf.float32)
    
    def logmeanexp(self, inputs):
        return tf.reduce_max(inputs) + tf.math.log( tf.reduce_mean(tf.math.exp(inputs - tf.reduce_max(inputs))) )
    
    def softmax(self, inputs):
        exp_inputs = tf.math.exp(inputs - tf.reduce_max(inputs))
        return exp_inputs / tf.reduce_sum(exp_inputs)
    
    def decision(self, probability):
        return tf.math.less(tf.random.uniform([1]), probability)
    
    def find_orig(self, orig_data, z_data, orig_inds):
        return tf.convert_to_tensor([orig_data[tf.where(tf.equal(i, orig_inds)).numpy()[0][0]] for i in z_data[:, 256]], dtype=tf.float32)
    
    def compute_loss(self, data, inf_iter, inf_rate, eta, gamma, l2_reg, image_buffer, batch_size, step):
        
        z_mean, z_log_var, z_image = self.encoder_image(data[0])
        indexs = tf.expand_dims(data[1], axis=1)
        combined = tf.concat([z_image, indexs], axis=1)
        
        # LD Image sampling
        if self.decision(0.95) == True:
            z = image_buffer.sample_byindex(combined[:, latent_dim])
        else:
            z = tf.random.uniform(combined[:, :latent_dim].get_shape().as_list())
        z = langevin_inf(z, combined[:, :latent_dim], inf_iter_train, inf_rate, eta, 'train', step)
        
        with tf.GradientTape() as tape:
            # training objective
            x_pos, norm_val = compute_energy(combined[:, :latent_dim], combined[:, :latent_dim])
            x_neg, norm_val = compute_energy(z, combined[:, :latent_dim])
            
            importance_weight = self.softmax(-gamma * x_neg)
            part1 = - 1 / gamma * self.logmeanexp(-gamma * x_pos) - tf.reduce_sum(x_neg * tf.stop_gradient(importance_weight))
            part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
            ebm_loss = part1 + part2
        grads = tape.gradient(ebm_loss, self.ebm.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ebm.trainable_weights))
        
        image_buffer.add(tf.concat([z, tf.expand_dims(combined[:, latent_dim], axis=1)], axis=1))
        
        ans = tf.reduce_mean(x_pos)
        self.re_batch.update_state(ans);          self.re_train.update_state(ans)
        
        ans = tf.reduce_mean(x_neg)
        self.lde_train.update_state(ans);         self.lde_batch.update_state(ans)
        
        self.train_loss_track.update_state(ebm_loss);        self.train_loss_batch.update_state(ebm_loss)
        
        ans = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.decoder_image(combined[:, :latent_dim]), self.decoder_image(z)))
        self.mse_train.update_state(ans);        self.mse_batch.update_state(ans)
        
        
    def eval_loss(self, data, inf_iter, inf_rate, eta, gamma, l2_reg, batch_size, step):
        
        z_mean, z_log_var, z_im = self.encoder_image(data[0])
        
        # LD Image sampling
        z = tf.random.uniform(z_im.get_shape().as_list())
        z, z_arr = langevin_inf(z, z_im, inf_iter_train, inf_rate, eta, 'val', step)
        del z_arr
        
        # training objective
        x_pos, norm_val = compute_energy(z_im, z_im)
        x_neg, norm_val = compute_energy(z, z_im)
        
        importance_weight = self.softmax(-gamma * x_neg)
        part1 = - 1 / gamma * self.logmeanexp(-gamma * x_pos) - tf.reduce_sum(x_neg * tf.stop_gradient(importance_weight))
        part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
        ebm_loss = part1 + part2
        
        self.re_val.update_state(tf.reduce_mean(x_pos))
        self.lde_val.update_state(tf.reduce_mean(x_neg))
        self.val_loss_track.update_state(ebm_loss)
        self.mse_val.update_state(tf.reduce_mean(tf.keras.losses.mean_squared_error(self.decoder_image(z_im), self.decoder_image(z))))
    
    def test_ebm(self, data, epoch, inf_rate_test, eta):
        z_mean, z_log_var, z_image = self.encoder_image(data[0])
        
        name = 'image'
        z = tf.random.uniform(z_image.get_shape().as_list())
        z_ld, z_rr = langevin_inf(z, z_image, inf_iter_test, inf_rate_test, eta, 'val', epoch)
        
        # show results
        decoded_image_orig = self.decoder_image.predict_on_batch(z_image)
        decoded_image_ld = self.decoder_image.predict_on_batch(z_ld)
        draw3(data[0][0], decoded_image_orig[0], decoded_image_ld[0], save_path, save, epoch, name)

        # ld image all
        img_arr = np.array([self.decoder_image(z_rr[i, :, :latent_dim]).numpy() for i in range(len(z_rr))])
        make_img(img_arr, 10, epoch, save_path, save, name)

        mse_loss_ld = tf.reduce_mean(tf.keras.losses.mean_squared_error(data[0], decoded_image_ld))
        mse_loss_vae = tf.reduce_mean(tf.keras.losses.mean_squared_error(data[0], decoded_image_orig))
        print('MSE VAE: {}, MSE LD: {}'.format(mse_loss_ld, mse_loss_vae), flush=True)

        


# In[ ]:


def create_training_loop(ebm_model, train_combo, val_combo, image_buffer, lr, batch_size, eta, l2_reg, gamma, inf_rate_train, inf_rate_val, inf_rate_test):
    print('MAX_EPOCHS: {}, LR: {}, BS: {}, Z_DIM: {}, IMG_S: {}'.format(max_epoch, lr, batch_size, latent_dim, image_shape[0]), flush=True)
    print('Inf Rate Train: {}, Inf Iter Train: {}'.format(inf_rate_train, inf_iter_train), flush=True)
    print('Inf Rate Val: {}, Inf Iter Val: {}'.format(inf_rate_val, inf_iter_val), flush=True)
    print('Eta: {}, L2 Reg: {}, Gamma: {}'.format(eta, l2_reg, gamma), flush=True)
    
    train_loss = []; val_loss = []
    re_energy_tr = []; unre_energy_tr = []; ld_energy_tr = []
    re_energy_val = []; unre_energy_val = []; ld_energy_val = []
    mse_tr = []; mse_tst = []
    epoch = 0

    while epoch < max_epoch:
        print("\nEpoch:", epoch, flush=True)
        start = timer()

        print('Training EBM', flush=True)
        step_count = 0
        for re_image, indx in train_combo:
            step_count += 1
            ebm_model.compute_loss([re_image, indx], inf_iter_train, inf_rate_train, eta, gamma, l2_reg, image_buffer, batch_size, step_count)

            with train_summary_writer.as_default():
                tf.summary.scalar('Loss Batch', ebm_model.train_loss_batch.result(), step=epoch)
                tf.summary.scalar('Re-Energy Batch', ebm_model.re_batch.result(), step=epoch)
                tf.summary.scalar('LD-Energy Batch', ebm_model.lde_batch.result(), step=epoch)
                tf.summary.scalar('MSE Batch', ebm_model.mse_batch.result(), step=epoch)

            if epoch % 4 == 0:
                print('Loss Batch: {}, MSE Batch: {}'.format(ebm_model.train_loss_batch.result(), ebm_model.mse_batch.result()), flush=True)

                print("Re-Energy Batch: {}, Unre-Energy Batch: {}, LD-Energy Batch: {}".format(ebm_model.re_batch.result(), 
                                                                                               ebm_model.lde_batch.result()), flush=True)
            ebm_model.re_batch.reset_states()
            ebm_model.lde_batch.reset_states()
            ebm_model.train_loss_batch.reset_states()
            ebm_model.mse_batch.reset_states()
            
            tf.keras.backend.clear_session()
            gc.collect()

        print('Validating EBM', flush=True)
        for re_image, indx in val_combo:
            ebm_model.eval_loss([re_image, indx], inf_iter_val, inf_rate_val, eta, gamma, l2_reg, batch_size, epoch)
            tf.keras.backend.clear_session()
            gc.collect()

        train_loss.append(ebm_model.train_loss_track.result());    val_loss.append(ebm_model.val_loss_track.result())
        re_energy_tr.append(ebm_model.re_train.result());          ld_energy_tr.append(ebm_model.lde_train.result())
        re_energy_val.append(ebm_model.re_val.result());           ld_energy_val.append(ebm_model.lde_val.result())
        mse_tr.append(ebm_model.mse_train.result());    mse_tst.append(ebm_model.mse_val.result())

        if epoch % 4 == 0:
            if epoch > 0:
                plot_loss3(epoch, train_loss, val_loss, 'EBM', save_path, save)
                plot_loss5(epoch, re_energy_tr, unre_energy_tr, ld_energy_tr, re_energy_val, unre_energy_val, ld_energy_val, 
                           'Energy', save_path, save, 're_tr', 'unre_tr', 'ld_tr', 're_val', 'unre_val', 'ld_val')
                ebm.save_weights(save_path + 'ebm_' + str(epoch) + '.h5')

            for re_image, indx in val_combo.take(1):
                print('Example Image generation', flush=True)
                ebm_model.test_ebm(re_image, epoch, inf_rate_test, eta)
                tf.keras.backend.clear_session()
                gc.collect()

        with train_summary_writer.as_default():
            tf.summary.scalar('Train Loss', ebm_model.train_loss_track.result(), step=epoch)
            tf.summary.scalar('Re-Energy Train', ebm_model.re_train.result(), step=epoch)
            tf.summary.scalar('LD-Energy Train', ebm_model.lde_train.result(), step=epoch)
            tf.summary.scalar('MSE Train', ebm_model.mse_train.result(), step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('Val Loss', ebm_model.val_loss_track.result(), step=epoch)
            tf.summary.scalar('Re-Energy Train', ebm_model.re_val.result(), step=epoch)
            tf.summary.scalar('LD-Energy Train', ebm_model.lde_val.result(), step=epoch)
            tf.summary.scalar('MSE Val', ebm_model.mse_val.result(), step=epoch)

        print("Train Loss: {}, Val Loss: {}".format(ebm_model.train_loss_track.result(), ebm_model.val_loss_track.result()), flush=True)

        print("Re-Energy Train: {}, Unre-Energy Train: {}, LD-Energy Train: {}".format(ebm_model.re_train.result(), 
                                                                                       ebm_model.lde_train.result()), flush=True)

        print("Re-Energy Val: {}, Unre-Energy Val: {}, LD-Energy Val: {}".format(ebm_model.re_val.result(), 
                                                                                 ebm_model.lde_val.result()), flush=True)

        print("MSE Train: {}, MSE Val: {}".format(ebm_model.mse_train.result(), ebm_model.mse_val.result()), flush=True)
        
        ebm_model.train_loss_track.reset_states()
        ebm_model.val_loss_track.reset_states()

        ebm_model.mse_train.reset_states()
        ebm_model.mse_val.reset_states()

        ebm_model.re_train.reset_states()
        ebm_model.lde_train.reset_states()

        ebm_model.re_val.reset_states()
        ebm_model.lde_val.reset_states()
        
        epoch += 1

        tf.keras.backend.clear_session()
        gc.collect()

        end = timer()
        print('Loop Time: ', end - start, 'secs', flush=True)
        del start, end
        
        
# In[ ]:


# grid search
param_grid = {'batch_size': [64, 128, 256], 'lr' : [0.0001, 0.0003],
              'gamma' : [1, 0.1, 0.5], 'eta': [0.01, 0.001, 0.005], 'l2_reg': [0.1, 0.01, 0.001],
              'inf_rate_train' : [0.1, 0.01, 0.001, 0.005], 'inf_rate_val' : [0.1, 0.01, 0.001, 0.005], 'inf_rate_test' : [0.1, 0.01, 0.001, 0.005]}
grid = ParameterGrid(param_grid)

for ind, params in enumerate(grid):
    print('\nTrial: ', ind, flush=True)
    # save path
    name = 'trial_' + str(ind)
    save_path = './' + directory + '/' + name + '/'
    new_path = os.path.join(parent_dir, directory, name)
    if os.path.isdir(directory) == True:
        os.mkdir(new_path)
    
    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = save_path + 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = save_path + 'logs/gradient_tape/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # create datasets
    indexs = np.arange(len(x_train)).astype('float32').tolist()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, indexs)).shuffle(x_train.shape[0]).batch(params['batch_size']).prefetch(buffer_size=10000)
    
    testindexs = np.arange(len(x_test)).astype('float32').tolist()
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, testindexs)).shuffle(x_test.shape[0]).batch(params['batch_size']).prefetch(buffer_size=10000)
    
    # create VAE
    image_buffer = ReplayBuffer(10000)
    opt_ebm = tf.keras.optimizers.Adam(params['lr'])
    ebm_model = EBM(ebm, opt_ebm, model_encoder, model_decoder)
    
    # train
    create_training_loop(ebm_model, train_dataset, test_dataset, image_buffer, params['lr'], params['batch_size'], 
                         params['eta'], params['l2_reg'], params['gamma'], 
                         params['inf_rate_train'], params['inf_rate_val'], params['inf_rate_test'])
    
    print('***end loop***', flush=True)
    