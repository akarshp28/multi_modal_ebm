#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Add, Conv2D, Dense, Layer, Dropout, Conv2DTranspose, LeakyReLU, Reshape, GlobalMaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
import tensorflow_addons as tfa

from helper_functions2 import get_imgs, load_caps_img, create_image_gen, make_gallery, make_gallery2, plot_loss3

directory = "output37"
parent_dir = "/users/apokkunu/trial/img/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
save_path = './' + directory + '/'
save = True


# In[ ]:

image_shape = (128, 128, 3)
use_all_data = True
max_epoch = 500
latent_dim = 128

num_layer = 5
channel_multiplier = 8

l2_reg = 0.001
droprate = 0.1
kl_weight = 1

batch_size = 64
lr = 0.0001

# In[ ]:


strategy = tf.distribute.MirroredStrategy()
print ('\nNumber of devices: {}'.format(strategy.num_replicas_in_sync), flush=True)

GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync


# In[5]:

dataset_name = 'train'
train_capspath, train_imgspath = get_imgs(dataset_name)
train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)

dataset_name = 'test'
test_capspath, test_imgspath = get_imgs(dataset_name)
test_imgs = load_caps_img(test_capspath, test_imgspath, dataset_name, use_all_data)

train_dataset = create_image_gen(train_imgs, batch_size, image_shape[0])
test_dataset = create_image_gen(test_imgs, batch_size, image_shape[0])

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# In[2]:


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal([batch, dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def conv_block(x, channels, kernel_size = 3, padding = 'same'):
    x = Conv2D(channels, kernel_size, padding=padding, 
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = LeakyReLU()(x)
    return x

def res_block(x, channels, kernel_size = 3):
    input_x = x
    x = Conv2D(channels, kernel_size, padding='same', 
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = LeakyReLU()(x)
    x = Add()([input_x, x])
    return x

def downsampling_conv_block(x, channels, kernel_size = 4):
    x = Conv2D(channels, kernel_size, strides=(2, 2), padding="same",
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = LeakyReLU()(x)
    return x

def upsampling_conv_block(x, channels, kernel_size = 3):
    x = Conv2DTranspose(channels, kernel_size, strides=2, padding="same", 
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = LeakyReLU()(x)
    return x

def create_encoder(latent_dim, num_layer, channel_multiplier):
    encoder_iput = Input(shape=image_shape, name='image')
    channels = channel_multiplier
    x = conv_block(encoder_iput, channels, kernel_size = 4)
    x = res_block(x, channels)
    x = Dropout(rate=droprate)(x)
    
    print(K.int_shape(x))
    for i in range(num_layer):
        channels *= 2
        x = downsampling_conv_block(x, channels)
        print(K.int_shape(x))
        x = res_block(x, channels)
        x = Dropout(rate=droprate)(x)
        
    last_conv_shape = K.int_shape(x)
    x = GlobalMaxPool2D()(x)

    z_mean = Dense(latent_dim, name='z_mean', 
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    z_log_var = Dense(latent_dim, name='z_log_var', 
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    z = Sampling()([z_mean, z_log_var])
    
    model = Model(encoder_iput, [z_mean, z_log_var, z], name='encoder')
    model.summary()
    return model, last_conv_shape

def create_decoder(latent_dim, first_conv_shape, num_layer):
    decoder_input = Input(shape=(latent_dim,), name='latent_z')
    x = Dense(first_conv_shape[1] * first_conv_shape[2] * first_conv_shape[3], 
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(decoder_input)
    x = Reshape((first_conv_shape[1], first_conv_shape[2], first_conv_shape[3]))(x)
    
    print(K.int_shape(x))
    channels = first_conv_shape[3]
    
    for i in range(num_layer):
        x = res_block(x, channels)
        channels //= 2
        x = upsampling_conv_block(x, channels)
        print(K.int_shape(x))
        x = Dropout(rate=droprate)(x)
        
    x = res_block(x, channels)
    x = Conv2D(3, 3, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(l2_reg), bias_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = Activation('sigmoid', name='rec_image')(x)
    print(K.int_shape(x))
    model = Model(decoder_input, x, name='decoder')
    model.summary()
    return model


# In[ ]:


class VAE(Model):
    def __init__(self, encoder, decoder, optimizer):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = optimizer

        self.elbo = Mean(name="elbo")
        self.bce = Mean(name="reconstruction_loss")
        self.kld = Mean(name="kl_loss")
        
        self.elbo_test = Mean(name="elbo_test")
        self.bce_test = Mean(name="reconstruction_loss_test")
        self.kld_test = Mean(name="kl_loss_test")
        
        self.mse_train = Mean(name="mse_train")
        self.mse_test = Mean(name="mse_test")
    
    @property
    def metrics(self):
        return [self.elbo, self.bce, self.kld,
               self.elbo_test, self.bce_test, self.kld_test]
    
    def loss_func(self, data, reconstruction, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        reconstruction_loss = tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        
        kl_loss = (-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))) * kl_weight
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.nn.compute_average_loss(kl_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss, kl_loss, total_loss = self.loss_func(data, reconstruction, z_mean, z_log_var)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.elbo.update_state(total_loss)
        self.bce.update_state(reconstruction_loss)
        self.kld.update_state(kl_loss)
        
        mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
        self.mse_train.update_state(tf.nn.compute_average_loss(mse, global_batch_size=GLOBAL_BATCH_SIZE))
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss, kl_loss, total_loss = self.loss_func(data, reconstruction, z_mean, z_log_var)

        self.elbo_test.update_state(total_loss)
        self.bce_test.update_state(reconstruction_loss)
        self.kld_test.update_state(kl_loss)

        mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
        self.mse_test.update_state(tf.nn.compute_average_loss(mse, global_batch_size=GLOBAL_BATCH_SIZE))
        
            
# In[9]:


with strategy.scope():
    model_encoder, last_conv_shape = create_encoder(latent_dim, num_layer, channel_multiplier)
    model_decoder = create_decoder(latent_dim, last_conv_shape, num_layer)
    
    optimizer = tf.keras.optimizers.Adam(lr)
    model = VAE(model_encoder, model_decoder, optimizer)
    
    @tf.function
    def distributed_train_step(dataset_inputs):
        return strategy.run(model.train_step, args=(dataset_inputs,))
    
    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(model.test_step, args=(dataset_inputs,))


    elbo_train = []; bce_train = []; kld_train = [];
    elbo_test = []; bce_test = []; kld_test = [];
    mse_tr = []; mse_tst = []
    
    epoch = 0
    best_mse = 1000000
    mode = 'direct' #'vae'
    color_mode = True
    best_epoch = 0
    
    print('\nModel Hyperparameters:', flush=True)
    print('MAX_EPOCHS: {}, LR: {}, BS: {}, Z_DIM: {}, KLW: {}'.format(max_epoch, lr, batch_size, latent_dim, kl_weight), flush=True)
    print('\n', flush=True)

    while epoch < max_epoch:

        print('\nEpoch: {}'.format(epoch), flush=True)
        print('Training VAE', flush=True)
        for x in train_dist_dataset:
            distributed_train_step(x)

        print('Testing VAE', flush=True)
        for x in test_dist_dataset:
            distributed_test_step(x)
        
        elbo_train.append(model.elbo.result());            bce_train.append(model.bce.result());            kld_train.append(model.kld.result())
        elbo_test.append(model.elbo_test.result());        bce_test.append(model.bce_test.result());        kld_test.append(model.kld_test.result())
        mse_tr.append(model.mse_train.result());           mse_tst.append(model.mse_test.result());

        if model.mse_test.result().numpy() < best_mse:
            best_mse = model.mse_test.result()
            best_epoch = epoch
            model.encoder.save_weights(save_path + 'en_im_' + str(epoch) + '_' + str(best_mse.numpy()) + '.h5')
            model.decoder.save_weights(save_path + 'de_im_' + str(epoch) + '_' + str(best_mse.numpy()) + '.h5')
        
        print("Train ELBO: {}, Train BCE: {}, Train KLD: {}".format(model.elbo.result(), model.bce.result(), model.kld.result()), flush=True)
        print("Test ELBO: {}, Test BCE: {}, Test KLD: {}".format(model.elbo_test.result(), model.bce_test.result(), model.kld_test.result()), flush=True)
        print("Train MSE: {}, Test MSE: {}, Best MSE: {}, Best Epoch:{}".format(model.mse_train.result(), model.mse_test.result(), best_mse, best_epoch), flush=True)
        
        if epoch % 10 == 0:
            if epoch > 0:
                plot_loss3(epoch, elbo_train, elbo_test, 'ELBO', save_path, save)
                plot_loss3(epoch, bce_train, bce_test, 'BCE', save_path, save)
                plot_loss3(epoch, kld_train, kld_test, 'KLD', save_path, save)
                plot_loss3(epoch, mse_tr, mse_tst, 'MSE', save_path, save)

            for test_batch in test_dataset.take(1):
                if batch_size > 64:
                    make_gallery(test_batch.numpy()[:100], 10, epoch, 'orig_', color_mode, save_path, save)
                    make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim)
                elif batch_size == 32:
                    ncols = 10; num_imgs = 30
                    make_gallery(test_batch.numpy()[:30], ncols, epoch, 'orig_', color_mode, save_path, save)
                    make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim, ncols, num_imgs)
                else:
                    ncols = 10; num_imgs = 50
                    make_gallery(test_batch.numpy()[:50], ncols, epoch, 'orig_', color_mode, save_path, save)
                    make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim, ncols, num_imgs)

        model.elbo.reset_states();         model.bce.reset_states();         model.kld.reset_states()
        model.elbo_test.reset_states();    model.bce_test.reset_states();    model.kld_test.reset_states()
        model.mse_train.reset_states();    model.mse_test.reset_states()
        epoch += 1
        
        if epoch - best_epoch >= 50:
            break

    # final epoch
    plot_loss3(epoch-1, elbo_train, elbo_test, 'ELBO', save_path, save)
    plot_loss3(epoch-1, bce_train, bce_test, 'BCE', save_path, save)
    plot_loss3(epoch-1, kld_train, kld_test, 'KLD', save_path, save)
    plot_loss3(epoch-1, mse_tr, mse_tst, 'MSE', save_path, save)
    
    for test_batch in test_dataset.take(1):
        if batch_size > 64:
            make_gallery(test_batch.numpy()[:100], 10, epoch, 'orig_', color_mode, save_path, save)
            make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim)
        elif batch_size == 32:
            ncols = 10; num_imgs = 30
            make_gallery(test_batch.numpy()[:30], ncols, epoch, 'orig_', color_mode, save_path, save)
            make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim, ncols, num_imgs)
        else:
            ncols = 10; num_imgs = 50
            make_gallery(test_batch.numpy()[:50], ncols, epoch, 'orig_', color_mode, save_path, save)
            make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'pred_', color_mode, save_path, save, latent_dim, ncols, num_imgs)