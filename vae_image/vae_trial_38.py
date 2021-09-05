#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
from enum import Enum, auto

import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras import activations, Sequential, layers
from tensorflow_addons.layers import SpectralNormalization

from helper_functions2 import get_imgs, load_caps_img, create_image_gen, make_gallery, make_gallery2, plot_loss2

directory = "output38"
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

n_encoder_channels = 16
n_decoder_channels = 16

mult = 1
scale_factor = 2

latent_dim = 128
res_cells_per_group = 1
n_groups_per_scale = [2] * 6
n_latent_scales = len(n_groups_per_scale)

batch_size = 19
lr = 0.01 # initial
sr_lambda = 0.01
step_based_warmup = False
drop_remain = True


# In[ ]:


strategy = tf.distribute.MirroredStrategy()
print ('\nNumber of devices: {}'.format(strategy.num_replicas_in_sync), flush=True)

GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync


# In[5]:

dataset_name = 'train'
train_capspath, train_imgspath = get_imgs(dataset_name)
train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)

dataset_name = 'val'
val_capspath, val_imgspath = get_imgs(dataset_name)
val_caps, val_imgs = load_caps_img(val_capspath, val_imgspath, dataset_name, use_all_data)

dataset_name = 'test'
test_capspath, test_imgspath = get_imgs(dataset_name)
test_imgs = load_caps_img(test_capspath, test_imgspath, dataset_name, use_all_data)

train_dataset = create_image_gen(set(train_imgs), batch_size, image_shape[0], drop_remain)
val_dataset = create_image_gen(set(val_imgs), batch_size, image_shape[0], drop_remain)
test_dataset = create_image_gen(set(test_imgs), batch_size, image_shape[0], drop_remain)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# In[2]:


def squeeze_excite_block(input_x, ratio=16):
    batch_size, h, w, c = input_x.shape
    num_hidden = max(c / ratio, 4)
    x = layers.GlobalAveragePooling2D()(input_x)
    x = layers.Dense(units=num_hidden, activation='relu')(x)
    x = layers.Dense(units=c, activation='sigmoid')(x)
    x = layers.Reshape((1, 1, c))(x)
    x = layers.Multiply()([input_x, x])
    return x

def encoding_res_block(input_x, output_channels):
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(input_x)
    x = layers.Activation(activations.swish)(x)
    x = layers.Conv2D(output_channels, (3, 3), padding="same")(x)
    
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    x = layers.Conv2D(output_channels, (3, 3), padding="same")(x)
    x = squeeze_excite_block(x)
    
    x = layers.Add()([0.1 * input_x, x])
    return x

def generative_res_block(input_x, output_channels, expansion_ratio=6):
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(input_x)
    x = layers.Conv2D(expansion_ratio * output_channels, (1, 1), padding="same")(x)
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    x = layers.DepthwiseConv2D((5, 5), padding="same")(x)
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = layers.Activation(activations.swish)(x)
    x = layers.Conv2D(output_channels, (1, 1), padding="same")(x)
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(x)
    x = squeeze_excite_block(x)
    
    x = layers.Add()([0.1 * input_x, x])
    return x
    
def rescaler(input_x, n_channels, scale_factor, rescale_type):
    x = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)(input_x)
    x = layers.Activation(activations.swish)(x)
    
    if rescale_type == 'up':
        _, height, width, _ = x.shape
        x = tf.image.resize(x, size=(scale_factor * height, scale_factor * width), method="nearest")
        x = layers.Conv2D(n_channels, (3, 3), strides=(1, 1), padding="same")(x)
    else:
        x = layers.Conv2D(n_channels, (3, 3), strides=(scale_factor, scale_factor), padding="same")(x)
    return x


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal([batch, dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#########################################################

def create_encoder(mult=1):
    x_in = layers.Input(shape=image_shape, name='encoder_input')
    x = layers.Conv2D(n_encoder_channels, (3, 3), padding="same")(x_in)
    
    for scale in range(n_latent_scales):
        n_groups = n_groups_per_scale[scale]
        print('\nGroup: ', scale)

        for group_idx in range(n_groups):
            output_channels = n_encoder_channels * mult
            print('Output_channels: ', output_channels)
            
            for rb in range(res_cells_per_group):
                x = encoding_res_block(x, output_channels)
                print('res block')
        
        # We downsample in the end of each scale except last
        if scale < n_latent_scales - 1:
            output_channels = n_encoder_channels * mult * scale_factor
            x = rescaler(x, output_channels, scale_factor, 'down')
            print('Rescaler')
            print('New output_channels: ', output_channels)
            mult *= scale_factor
    
    x = layers.ELU()(x)
    x = layers.Conv2D(n_encoder_channels * mult, (1, 1), padding="same")(x)
    x = layers.ELU()(x)
    
    before_shape = x.shape[1:]
    x = layers.GlobalAveragePooling2D()(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    model_encoder = tf.keras.models.Model(x_in, [z_mean, z_log_var, z], name='encoder')
    model_encoder.summary()
    return model_encoder, before_shape, mult

#########################################################

def create_decoder(en_mult, before_shape):
    decoder_input = layers.Input(shape=(latent_dim), name='decoder_input')
    x = layers.Dense(before_shape[0] * before_shape[1] * before_shape[2])(decoder_input)
    x = layers.Reshape(before_shape)(x)
    
    for scale in range(n_latent_scales):
        print('\nGroup: ', scale)
        n_groups = n_groups_per_scale[scale]

        for group in range(n_groups):
            output_channels = int(n_decoder_channels * en_mult)
            print('Output channels', output_channels)
            
            for res in range(res_cells_per_group):
                x = generative_res_block(x, output_channels, expansion_ratio=6)
                print('Gen Res block', flush=True)

        if scale < n_latent_scales - 1:
            output_channels = int(n_decoder_channels * en_mult / scale_factor)
            x = rescaler(x, output_channels, scale_factor, 'up')
            print('Rescaler', flush=True)
            en_mult /= scale_factor
    
    x = layers.Conv2D(3, kernel_size=(3, 3), padding="same")(x)
    x = layers.Activation(activations.sigmoid)(x)
    
    model_decoder = tf.keras.models.Model(decoder_input, x, name='decoder')
    model_decoder.summary()
    return model_decoder

#########################################################



# In[ ]:


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, optimizer):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = optimizer

        self.elbo = Mean(name="elbo")
        self.bce = Mean(name="reconstruction_loss")
        self.kld = Mean(name="kl_loss")
        self.bnloss = Mean(name="bn_loss")
        
        self.elbo_val = Mean(name="elbo_test")
        self.bce_val = Mean(name="reconstruction_loss_val")
        self.kld_val = Mean(name="kl_loss_val")
        self.bnloss_val = Mean(name="bn_loss_val")
        
        self.elbo_test = Mean(name="elbo_test")
        self.bce_test = Mean(name="reconstruction_loss_test")
        self.kld_test = Mean(name="kl_loss_test")
        self.bnloss_test = Mean(name="bn_loss_test")
        
        self.mse_train = Mean(name="mse_train")
        self.mse_val = Mean(name="mse_val")
        self.mse_test = Mean(name="mse_test")
    
    def loss_func(self, data, reconstruction, z_mean, z_log_var, sr_lambda, step_based_warmup, steps, epoch):
        reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        reconstruction_loss = tf.nn.compute_average_loss(reconstruction_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        
        warmup_metric = steps if step_based_warmup else epoch
        beta = tf.math.minimum((warmup_metric / (0.3 * max_epoch)), 1)
        
        # activate_balancing
        kl_loss = (-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))) * beta
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = tf.nn.compute_average_loss(kl_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss, kl_loss, total_loss
    
    def calc_bn_loss(self, sr_lambda):
        def update_loss(layer):
            nonlocal bn_loss
            if isinstance(layer, layers.BatchNormalization):
                bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
        
        bn_loss = 0.0
        for model in [self.encoder, self.decoder]:
            for layer in model.layers:
                update_loss(layer)
        return sr_lambda * bn_loss
    
    def train_step(self, data, steps, epoch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss, kl_loss, total_loss = self.loss_func(data, reconstruction, 
                                                                      z_mean, z_log_var,
                                                                      sr_lambda, step_based_warmup, 
                                                                      steps, epoch)
            bn_loss = self.calc_bn_loss(sr_lambda)
            total_loss = total_loss + bn_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.elbo.update_state(total_loss)
        self.bce.update_state(reconstruction_loss)
        self.kld.update_state(kl_loss)
        self.bnloss.update_state(bn_loss)
        mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
        self.mse_train.update_state(tf.nn.compute_average_loss(mse, global_batch_size=GLOBAL_BATCH_SIZE))
    
    def test_step(self, data, steps, epoch, mode):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss, kl_loss, total_loss = self.loss_func(data, reconstruction, 
                                                                  z_mean, z_log_var, 
                                                                  sr_lambda, step_based_warmup, 
                                                                  steps, epoch)
        bn_loss = self.calc_bn_loss(sr_lambda)
        total_loss = total_loss + bn_loss
        if mode == 'test':
            self.elbo_test.update_state(total_loss)
            self.bce_test.update_state(reconstruction_loss)
            self.kld_test.update_state(kl_loss)
            self.bnloss_test.update_state(bn_loss)
            mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
            self.mse_test.update_state(tf.nn.compute_average_loss(mse, global_batch_size=GLOBAL_BATCH_SIZE))
        else:
            self.elbo_val.update_state(total_loss)
            self.bce_val.update_state(reconstruction_loss)
            self.kld_val.update_state(kl_loss)
            self.bnloss_val.update_state(bn_loss)
            mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
            self.mse_val.update_state(tf.nn.compute_average_loss(mse, global_batch_size=self.GLOBAL_BATCH_SIZE))
        
            
# In[9]:


with strategy.scope():
    model_encoder, before_shape, en_mult = create_encoder()
    model_decoder = create_decoder(en_mult, before_shape)
    
    train_imgs_len = len(set(train_imgs))
    batches_per_epoch = (train_imgs_len + batch_size - 1) // batch_size
    lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps=max_epoch * batches_per_epoch)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    model = VAE(model_encoder, model_decoder, optimizer)
    
#     @tf.function(experimental_relax_shapes=True, input_signature=[tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32), 
#                                                                    tf.TensorSpec(shape=[None], dtype=tf.float32),
#                                                                    tf.TensorSpec(shape=[None], dtype=tf.float32)])
    @tf.function
    def distributed_train_step(dataset_inputs, steps, epoch):
        return strategy.run(model.train_step, args=(dataset_inputs, steps, epoch))
    
    
#     @tf.function(experimental_relax_shapes=True, input_signature=[tf.TensorSpec(shape=[None, 128, 128, 3], dtype=tf.float32), 
#                                                                    tf.TensorSpec(shape=[None], dtype=tf.float32),
#                                                                    tf.TensorSpec(shape=[None], dtype=tf.float32),
#                                                                    tf.TensorSpec(shape=[None], dtype=tf.string)])

    @tf.function
    def distributed_test_step(dataset_inputs, steps, epoch, mode):
        return strategy.run(model.test_step, args=(dataset_inputs, steps, epoch, mode))


    elbo_train = []; bce_train = []; kld_train = []; bn_train = []
    elbo_val = []; bce_val = []; kld_val = []; bn_val = []
    elbo_test = []; bce_test = []; kld_test = []; bn_test = []
    mse_tr = []; mse_vl=[]; mse_tst = []

    epoch = 0
    best_mse = 1000000
    mode = 'direct' #'vae'
    color_mode = True
    best_epoch = 0

    print('\nHyperparameters:', flush=True)
    print("Use full dataset: {}, Drop stray batches: {}".format(use_all_data, drop_remain))
    print("Max Epochs: {}, BatchSize: {}, Batches per Epoch: {}, Initial LR: {}".format(max_epoch, batch_size, batches_per_epoch, lr))
    print("EN CH: {}, DEC CH: {}".format(n_encoder_channels, n_decoder_channels))
    print("Num Groups: {}, Res Blocks per Group: {}".format(n_groups_per_scale, res_cells_per_group))
    print("Spectral Reg: {}, IMG Scale: {}, Step based KL-Warmup: {}".format(sr_lambda, scale_factor, step_based_warmup))

    while epoch < max_epoch:
        print('\nEpoch: {}'.format(epoch), flush=True)

        print('Training VAE', flush=True)
        step = 0
        for x in train_dist_dataset:
            step += 1
            distributed_train_step(x, step, epoch)

        print('Validating VAE', flush=True)
        step = 0
        for x in val_dist_dataset:
            step += 1
            distributed_test_step(x, step, epoch, 'val')

        print('Testing VAE', flush=True)
        step = 0
        for x in test_dist_dataset:
            step += 1
            distributed_test_step(x, step, epoch, 'test')

        # Store current metric values
        elbo_tr_value = model.elbo.result();             bce_tr_value = model.bce.result();
        kld_tr_value = model.kld.result();               bnl_tr_value = model.bnloss.result()

        elbo_val_value = model.elbo_val.result();        bce_val_value = model.bce_val.result()
        kld_val_value = model.kld_val.result();          bnl_val_value = model.bnloss_val.result()

        elbo_test_value = model.elbo_test.result();      bce_test_value = model.bce_test.result()
        kld_test_value = model.kld_test.result();        bnl_test_value = model.bnloss_test.result()

        mse_tr_value = model.mse_train.result();         mse_val_value = model.mse_val.result();        mse_test_value = model.mse_test.result()

        # store all values
        elbo_train.append(elbo_tr_value);       bce_train.append(bce_tr_value);       kld_train.append(kld_tr_value);       bn_train.append(bnl_tr_value)
        elbo_val.append(elbo_val_value);        bce_val.append(bce_val_value);        kld_val.append(kld_val_value);        bn_val.append(bnl_val_value)
        elbo_test.append(elbo_test_value);      bce_test.append(bce_test_value);      kld_test.append(kld_test_value);      bn_test.append(bnl_test_value)
        mse_tr.append(mse_tr_value);            mse_vl.append(mse_val_value);         mse_tst.append(mse_test_value)

        if mse_val_value < best_mse:
            best_mse = mse_val_value
            best_epoch = epoch
            model.encoder.save_weights(save_path + 'en_im_' + str(epoch) + '.h5')
            model.decoder.save_weights(save_path + 'de_im_' + str(epoch) + '.h5')

        temp = "Train ELBO: {}, Train BCE: {}, Train KLD: {}, Train BN: {}"
        print(temp.format(elbo_tr_value, bce_tr_value, kld_tr_value, bnl_tr_value), flush=True)

        temp = "Val ELBO: {}, Val BCE: {}, Val KLD: {}, Val BN: {}"
        print(temp.format(elbo_val_value, bce_val_value, kld_val_value, bnl_val_value), flush=True)

        temp = "Test ELBO: {}, Test BCE: {}, Test KLD: {}, Test BN: {}"
        print(temp.format(elbo_test_value, bce_test_value, kld_test_value, bnl_test_value), flush=True)

        print("Train MSE: {}, Val MSE: {}, Test MSE: {}".format(mse_tr_value, mse_val_value, mse_test_value), flush=True)
        print("Best Val MSE: {}, Best Epoch: {}".format(best_mse, best_epoch), flush=True)

        if epoch % 10 == 0:
            if epoch > 0:
                plot_loss2(epoch, elbo_train, elbo_val, elbo_test, 'ELBO', save_path, save)
                plot_loss2(epoch, bce_train, bce_val, bce_test, 'BCE', save_path, save)
                plot_loss2(epoch, kld_train, kld_val, kld_test, 'KLD', save_path, save)
                plot_loss2(epoch, bn_train, bn_val, bn_test, 'BN', save_path, save)
                plot_loss2(epoch, mse_tr, mse_vl, mse_tst, 'MSE', save_path, save)

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

        model.elbo.reset_states();         model.bce.reset_states();         model.kld.reset_states();              model.bnloss.reset_states()
        model.elbo_val.reset_states();     model.bce_val.reset_states();     model.kld_val.reset_states();          model.bnloss_val.reset_states()
        model.elbo_test.reset_states();    model.bce_test.reset_states();    model.kld_test.reset_states();         model.bnloss_test.reset_states()
        model.mse_train.reset_states();    model.mse_val.reset_states();     model.mse_test.reset_states()
        epoch += 1

        # early stopping
        if epoch - best_epoch >= 50:
            break

        end = timer()
        print('Loop Time: ', end - start, 'secs', flush=True)

    # final epoch
    plot_loss2(epoch-1, elbo_train, elbo_val, elbo_test, 'ELBO', save_path, save)
    plot_loss2(epoch-1, bce_train, bce_val, bce_test, 'BCE', save_path, save)
    plot_loss2(epoch-1, kld_train, kld_val, kld_test, 'KLD', save_path, save)
    plot_loss2(epoch-1, bn_train, bn_val, bn_test, 'BN', save_path, save)
    plot_loss2(epoch-1, mse_tr, mse_vl, mse_tst, 'MSE', save_path, save)

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