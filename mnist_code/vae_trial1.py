#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, ZeroPadding2D, Activation, Add, Conv2D, Dense, Layer, Conv2DTranspose, LeakyReLU, Reshape, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
import tensorflow_addons as tfa

from helper_functions2 import make_gallery, make_gallery2, plot_loss3
from sklearn.model_selection import ParameterGrid

directory = "output1"
parent_dir = "/users/apokkunu/trial/mnist/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
save = True


# In[ ]:


max_epoch = 100
latent_dim = 32
image_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print("Number of original training examples:", len(x_train), x_train.shape, flush=True)
print("Number of original test examples:", len(x_test), x_test.shape, flush=True)


# In[2]:

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
    
    model = Model(encoder_iput, [z_mean, z_log_var, z], name='encoder')
    model.summary()
    return model, last_conv_shape

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
    
    @tf.function
    def train_step(self, data, kl_weight):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), 
                                                               axis=(1, 2)))
            
            kl_loss = (-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))) * kl_weight
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.elbo.update_state(total_loss)
        self.bce.update_state(reconstruction_loss)
        self.kld.update_state(kl_loss)
        
        self.mse_train.update_state(tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))))
    
    def test_step(self, data, kl_weight):
        z_mean, z_log_var, z = self.encoder.predict(data)
        reconstruction = self.decoder.predict(z)
        reconstruction = tf.convert_to_tensor(reconstruction, dtype=tf.float32)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), 
                                                           axis=(1, 2)))
        
        kl_loss = (-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))) * kl_weight
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        
        self.elbo_test.update_state(total_loss)
        self.bce_test.update_state(reconstruction_loss)
        self.kld_test.update_state(kl_loss)
        
        self.mse_test.update_state(tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))))
        
            
# In[9]:


def create_training_loop(model, train_dataset, test_dataset, max_epoch, lr, batch_size, latent_dim, kl_weight, save_path):
    elbo_train = []; bce_train = []; kld_train = [];
    elbo_test = []; bce_test = []; kld_test = [];
    mse_tr = []; mse_tst = []
    epoch = 0
    
    mode = 'direct' #'vae'
    color_mode = False
    
    print('\nModel Hyperparameters:', flush=True)
    print('MAX_EPOCHS: {}, LR: {}, BS: {}, Z_DIM: {}, KLW: {}'.format(max_epoch, lr, batch_size, latent_dim, kl_weight), flush=True)
    print('\n', flush=True)
    
    while epoch < max_epoch:

        for batch in train_dataset:
            model.train_step(batch, kl_weight)
        
        for test_batch in test_dataset:
            model.test_step(test_batch, kl_weight)
        
        elbo_train.append(model.elbo.result());        bce_train.append(model.bce.result());        kld_train.append(model.kld.result())
        elbo_test.append(model.elbo_test.result());        bce_test.append(model.bce_test.result());        kld_test.append(model.kld_test.result())
        mse_tr.append(model.mse_train.result());        mse_tst.append(model.mse_test.result());
        
        if epoch % 10 == 0:
            print('\nEpoch: {}'.format(epoch), flush=True)
            print("Train ELBO: {}, Train BCE: {}, Train KLD: {}".format(model.elbo.result(), model.bce.result(), model.kld.result()), flush=True)
            print("Test ELBO: {}, Test BCE: {}, Test KLD: {}".format(model.elbo_test.result(), model.bce_test.result(), model.kld_test.result()), flush=True)
            print("Test MSE: {}, Test MSE: {}".format(model.mse_train.result(), model.mse_test.result()), flush=True)
            
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

            model.encoder.save_weights(save_path + 'en_im_' + str(epoch) + '.h5')
            model.decoder.save_weights(save_path + 'de_im_' + str(epoch) + '.h5')

        model.elbo.reset_states();    model.bce.reset_states();    model.kld.reset_states()
        model.elbo_test.reset_states();    model.bce_test.reset_states();    model.kld_test.reset_states()
        model.mse_train.reset_states();    model.mse_test.reset_states()
        epoch += 1
    
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
    
    model.encoder.save_weights(save_path + 'en_im_' + str(epoch) + '.h5')
    model.decoder.save_weights(save_path + 'de_im_' + str(epoch) + '.h5')
    
# In[]:

# grid search
param_grid = {'batch_size': [128, 64], 'kl_weight' : [1, 0.5, 0.1, 0.01], 'lr' : [0.0001, 0.0003]}
grid = ParameterGrid(param_grid)

for ind, params in enumerate(grid):
    
    # save path
    name = ''.join(map(str, [params['batch_size'], '_', params['kl_weight'], '_', params['lr']]))
    save_path = './' + directory + '/' + name + '/'
    new_path = os.path.join(parent_dir, directory, name)
    if os.path.isdir(directory) == True:
        os.mkdir(new_path)
    
    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(params['batch_size']).prefetch(buffer_size=10000)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(x_test.shape[0]).batch(params['batch_size']).prefetch(buffer_size=10000)

    # create en/dc
    num_layer = 2
    model_encoder, last_conv_shape = create_encoder(latent_dim, num_layer)
    model_decoder = create_decoder(latent_dim, last_conv_shape, num_layer)
    
    # create VAE
    optimizer = tf.keras.optimizers.Adam(params['lr'])
    model = VAE(model_encoder, model_decoder, optimizer)
    
    # train
    create_training_loop(model, train_dataset, test_dataset, max_epoch, params['lr'], params['batch_size'], latent_dim, params['kl_weight'], save_path)
    
    print('***end loop***')