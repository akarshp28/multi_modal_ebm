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

from helper_functions2 import make_img, plot_loss3, plot_loss7, draw3
from sklearn.model_selection import train_test_split

directory = "ebmout_18"
parent_dir = "/users/apokkunu/trial/mnist/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
save = True
save_path = path + '/'

# In[ ]:

image_shape = (28, 28, 1)
latent_dim = 32
max_epoch = 1000

batch_size = 256
lr = 0.0001

ebm_units = 512

inf_iter_val = 100
inf_rate_val = 0.001
eta = 0.001

l2_reg = 0.1


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

# create datasets
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(batch_size).prefetch(buffer_size=10000)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(x_test.shape[0]).batch(batch_size).prefetch(buffer_size=10000)


# In[ ]:


def compute_gradient(z):
    with tf.GradientTape() as tape:
        tape.watch(z)
        energy = ebm(z) + 0.5 * tf.reduce_sum(tf.pow(z,2), axis=1)
    return tape.gradient(energy, z), energy

def langevin_inf(z, orig_z, inf_iter, inf_rate, eta, mode, step, epoch):
    current_z = z
    
    if mode == 'val':
        z_arr = [current_z]
        
    for i in range(inf_iter):
        gradients, energy = compute_gradient(current_z)
        
        # Langevin dynamics
        term1 = 0.5 * inf_rate * gradients
        term2 = eta * tf.random.normal(current_z.get_shape().as_list())
        next_z = current_z - term1 + term2
        current_z = tf.clip_by_value(next_z, 0.0, 1.0)
        
        if step % 50 == 0 and epoch % 10 == 0 and mode == 'train':
            template = "LD Step: {}, Avg. MSE: {}, Avg. energy: {}"
            mse = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(model_decoder(orig_z), model_decoder(current_z)), axis=(1, 2))).numpy()
            print(template.format(i, mse, tf.reduce_mean(energy).numpy()), flush=True)
        
        if mode == 'val':
            z_arr.append(current_z)
    
    if mode == 'train':
        return current_z
    else:
        return current_z, np.array(z_arr)
    

# In[ ]:


latent_in = Input(shape=(latent_dim,))
x = Dense(ebm_units, activation=tf.nn.leaky_relu)(latent_in)
x = Dense(ebm_units//2, activation=tf.nn.leaky_relu)(x)
energy_vals = Dense(1)(x)
ebm = Model(latent_in, energy_vals, name="EBM")
ebm.summary()


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
        
        self.train_loss_batch = tf.keras.metrics.Mean('train_loss_batch', dtype=tf.float32)
        self.re_batch = tf.keras.metrics.Mean('re_batch', dtype=tf.float32)
        self.lde_batch = tf.keras.metrics.Mean('lde_batch', dtype=tf.float32)
    
    def compute_loss(self, data, l2_reg, mode):
        z_mean, z_log_var, z_image = self.encoder_image(data)
        z_s = tf.random.normal(z_image.get_shape().as_list())
        
        if mode == 'train':
            with tf.GradientTape() as tape:
                x_pos = self.ebm(z_image)
                x_neg = self.ebm(z_s)
                part1 = -tf.math.log(tf.nn.sigmoid(-x_pos)) + tf.math.log(tf.nn.sigmoid(-x_neg))
                part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
                ebm_loss = part1 + part2
            grads = tape.gradient(ebm_loss, self.ebm.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.ebm.trainable_weights))
            
            ans = tf.reduce_mean(x_pos);        self.re_batch.update_state(ans);          self.re_train.update_state(ans)
            ans = tf.reduce_mean(x_neg);        self.lde_train.update_state(ans);         self.lde_batch.update_state(ans)
            self.train_loss_track.update_state(ebm_loss);        self.train_loss_batch.update_state(ebm_loss)
        else:
            x_pos = self.ebm(z_image)
            x_neg = self.ebm(z_s)
            part1 = -tf.math.log(tf.nn.sigmoid(-x_pos)) + tf.math.log(tf.nn.sigmoid(-x_neg))
            part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
            ebm_loss = part1 + part2
            
            self.re_val.update_state(tf.reduce_mean(x_pos))
            self.lde_val.update_state(tf.reduce_mean(x_neg))
            self.val_loss_track.update_state(ebm_loss)
        
    def test_ebm(self, data, epoch):
        z_mean, z_log_var, z_image = self.encoder_image(data)
        z = tf.random.normal(z_image.get_shape().as_list())
        z_ld, z_rr = langevin_inf(z, z_image, inf_iter_val, inf_rate_val, eta, 'val', epoch, epoch)
        
        color_mode = False
        name = 'testing_images'
        
        # show results
        decoded_image_orig = self.decoder_image(z_image)
        decoded_image_ld = self.decoder_image(z_ld)
        
        draw3(data[0], decoded_image_orig[0], decoded_image_ld[0], color_mode, save_path, save, epoch, name)
        
        # ld step wise image all
        img_arr = np.array([self.decoder_image(z_rr[i, :, :latent_dim]).numpy() for i in range(len(z_rr))])
        make_img(img_arr, 10, epoch, color_mode, save_path, save, name)
        
        mse_loss_ld = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, decoded_image_ld), axis=(1, 2)))
        mse_loss_vae = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, decoded_image_orig), axis=(1, 2)))
        print('MSE VAE: {}, MSE LD: {}'.format(mse_loss_ld, mse_loss_vae), flush=True)


# In[ ]:

# tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = path + '/logs/gradient_tape/' + current_time + '/train'
val_log_dir = path + '/logs/gradient_tape/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)


# In[ ]:


# create VAE
opt_ebm = tf.keras.optimizers.Adam(lr)
ebm_model = EBM(ebm, opt_ebm, model_encoder, model_decoder)


# In[ ]:


print('MAX_EPOCHS: {}, LR: {}, BS: {}, Z_DIM: {}, IMG_S: {}'.format(max_epoch, lr, batch_size, latent_dim, image_shape[0]), flush=True)
print('Inf Rate Test: {}, Inf Iter Test: {}'.format(inf_rate_val, inf_iter_val), flush=True)
print('Eta: {}, L2 Reg: {}'.format(eta, l2_reg), flush=True)


# In[ ]:


train_loss = []; val_loss = []
re_energy_tr = []; ld_energy_tr = []
re_energy_val = []; ld_energy_val = []
epoch = 0

while epoch < max_epoch:
    print("\nEpoch:", epoch, flush=True)
    start = timer()

    print('Training EBM', flush=True)
    for re_image in train_dataset:
        ebm_model.compute_loss(re_image, l2_reg, 'train')
        
        if epoch % 20 == 0:
            template = "Loss Batch: {}, E_POS Batch: {}, E_NEG Batch: {}"
            print(template.format(ebm_model.train_loss_batch.result(), ebm_model.re_batch.result(), ebm_model.lde_batch.result()), flush=True)
        
        ebm_model.re_batch.reset_states()
        ebm_model.lde_batch.reset_states()
        ebm_model.train_loss_batch.reset_states()
        tf.keras.backend.clear_session()
        gc.collect()
    
    print('Validating EBM', flush=True)
    for re_image in test_dataset:
        ebm_model.compute_loss(re_image, l2_reg, 'val')
        tf.keras.backend.clear_session()
        gc.collect()
    
    train_loss.append(ebm_model.train_loss_track.result());    val_loss.append(ebm_model.val_loss_track.result())
    re_energy_tr.append(ebm_model.re_train.result());          ld_energy_tr.append(ebm_model.lde_train.result())
    re_energy_val.append(ebm_model.re_val.result());           ld_energy_val.append(ebm_model.lde_val.result())

    if epoch % 20 == 0:
        if epoch > 0:
            plot_loss3(epoch, train_loss, val_loss, 'Loss', save_path, save)
            plot_loss7(epoch, re_energy_tr, ld_energy_tr, re_energy_val, ld_energy_val, 'Energy', save_path, save, 'epos_tr', 'eneg_tr', 'epos_val', 'eneg_val')
            ebm.save_weights(save_path + 'ebm_' + str(epoch) + '.h5')

        for re_image in test_dataset.take(1):
            print('Example Image generation', flush=True)
            ebm_model.test_ebm(re_image, epoch)
            tf.keras.backend.clear_session()
            gc.collect()
    
    with train_summary_writer.as_default():
        tf.summary.scalar('Train Loss', ebm_model.train_loss_track.result(), step=epoch)
        tf.summary.scalar('E_POS Train', ebm_model.re_train.result(), step=epoch)
        tf.summary.scalar('E_NEG Train', ebm_model.lde_train.result(), step=epoch)
    
    with val_summary_writer.as_default():
        tf.summary.scalar('Val Loss', ebm_model.val_loss_track.result(), step=epoch)
        tf.summary.scalar('E_POS Train', ebm_model.re_val.result(), step=epoch)
        tf.summary.scalar('E_NEG Train', ebm_model.lde_val.result(), step=epoch)
    
    print("Train Loss: {}, Val Loss: {}".format(ebm_model.train_loss_track.result(), ebm_model.val_loss_track.result()), flush=True)
    print("E_POS Train: {}, E_NEG Train: {}".format(ebm_model.re_train.result(), ebm_model.lde_train.result()), flush=True)
    print("E_POS Val: {}, E_NEG Val: {}".format(ebm_model.re_val.result(), ebm_model.lde_val.result()), flush=True)
    
    ebm_model.train_loss_track.reset_states();    ebm_model.val_loss_track.reset_states()
    ebm_model.re_train.reset_states();            ebm_model.lde_train.reset_states()
    ebm_model.re_val.reset_states();              ebm_model.lde_val.reset_states()

    epoch += 1
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    end = timer()
    print('Loop Time: ', end - start, 'secs', flush=True)
    del start, end
        
        
# final epoch
plot_loss3(epoch-1, train_loss, val_loss, 'Loss_final', save_path, save)
plot_loss7(epoch-1, re_energy_tr, ld_energy_tr, re_energy_val, ld_energy_val, 'Energy_final', save_path, save, 'epos_tr', 'eneg_tr', 'epos_val', 'eneg_val')
ebm.save_weights(save_path + 'ebm_final_' + str(epoch) + '.h5')

for re_image in test_dataset.take(1):
    ebm_model.test_ebm(re_image, epoch)
    tf.keras.backend.clear_session()
    gc.collect()

