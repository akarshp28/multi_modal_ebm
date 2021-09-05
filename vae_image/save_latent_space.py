#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from helper_functions2 import get_imgs, load_caps_img, clean_text_data, create_embd_dataset
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid


# In[ ]:

img_shape = 128
latent_dim = img_shape


####################################################################################

class Sampling(tfl.Layer):
    def __init__(self):
        super(Sampling, self).__init__()
        self.supports_masking = True

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal([batch, dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

input_dim = (img_shape, img_shape, 3)
encoder_inputs = tfl.Input(shape = input_dim, name = 'encoder_input')

use_batch_norm = False
use_dropout = True
droprate = 0.3

conv_filters = [64, 128, 256, 512, 1024]
conv_kernel_size = [4, 4, 4, 4, 4]
conv_strides = [2, 2, 2, 2, 2]
n_layers = len(conv_filters)

x = encoder_inputs

for i in range(n_layers):
    x = tfl.Conv2D(filters = conv_filters[i], kernel_size = conv_kernel_size[i], strides = conv_strides[i], 
                   padding = 'same')(x)

    if use_batch_norm:
        x = tfl.BatchNormalization()(x)

    x = tfl.LeakyReLU()(x)

    if use_dropout:
        x = tfl.Dropout(rate=droprate)(x)

x_before = x.shape[1:]
x = tfl.Flatten()(x)
x_after = x.shape[1]
x = tfl.Dense(768, activation=tf.nn.leaky_relu)(x)

z_mean_im = tfl.Dense(latent_dim, name='z_mean')(x)
z_log_var_im = tfl.Dense(latent_dim, name='z_log_var')(x)
z_img = Sampling()([z_mean_im, z_log_var_im])
encoder_image = Model(encoder_inputs, z_img, name='encoder_image', trainable=False)
encoder_image.trainable = False
encoder_image.load_weights('/users/apokkunu/trial/img/good_trials/output25/en_im_660.h5')


# In[]:


latent_inputs = tfl.Input(shape = (latent_dim,) , name = 'decoder_input')
x = tfl.Dense(768, activation=tf.nn.leaky_relu)(latent_inputs)
x = tfl.Dense(x_after, activation=tf.nn.leaky_relu)(x)
x = tfl.Reshape(x_before)(x)

conv_filters = [512, 256, 128, 64, 3]
n_layers = len(conv_filters)

for i in range(n_layers):
    x = tfl.Conv2DTranspose(filters=conv_filters[i], 
                        kernel_size=conv_kernel_size[i], strides = conv_strides[i], padding = 'same')(x)

    if use_batch_norm:
        x = tfl.BatchNormalization()(x)

    if i < n_layers - 1:
        x = tfl.LeakyReLU()(x)

        if use_dropout:
            x = tfl.Dropout(rate=droprate)(x)

    else:
        decoded_recon = sigmoid(x)

decoder_image = Model(latent_inputs, decoded_recon, name='decoder_image', trainable=False)
decoder_image.trainable = False
decoder_image.load_weights('/users/apokkunu/trial/img/good_trials/output25/de_im_660.h5')


# In[ ]:


####################################################################################

intr_dim = 256
droprate = 0.2
max_length = 50 #time steps
emb_dim = 300
vocab_size = 28444
kl_weight = 0.01
glove_embedding_matrix = np.load('/users/apokkunu/trial/text/annotations/train_copy.npy')

class custom_lstm(tfl.Layer):
    def __init__(self, intr_dim, droprate,  **kwargs):
        self.bi_lstm = tfl.Bidirectional(tfl.LSTM(intr_dim, recurrent_dropout=droprate, 
                                                                          return_sequences=False), merge_mode='concat')
        self.drop_layer = tfl.Dropout(droprate)
        super(custom_lstm, self).__init__(**kwargs)

    def call(self, inputs):
        h = self.bi_lstm(inputs)
        h = self.drop_layer(h)
        return h

    def compute_mask(self, inputs, mask=None):
        return mask

x = tfl.Input(shape=(max_length,))
embed_layer = tfl.Embedding(vocab_size, emb_dim, input_length=max_length, weights=[glove_embedding_matrix], 
                                        trainable=False, mask_zero=True)
encoder_layer = custom_lstm(intr_dim, droprate)

h = embed_layer(x)
h = encoder_layer(h)
z_mean = tfl.Dense(latent_dim, name='z_mean')(h)
z_log_var = tfl.Dense(latent_dim, name='z_log_var')(h)
z = Sampling()([z_mean, z_log_var])


class custom_decoder(tfl.Layer):
    def __init__(self, vocab_size, intr_dim, max_length, droprate, **kwargs):
        self.rpv = tfl.RepeatVector(max_length)
        self.lstm_layer_1 = tfl.LSTM(intr_dim, return_sequences=True, recurrent_dropout=droprate)
        self.droplayer_2 = tfl.Dropout(droprate)
        self.lstm_layer_2 = tfl.LSTM(intr_dim*2, return_sequences=True, recurrent_dropout=droprate)
        self.droplayer_3 = tfl.Dropout(droprate)
        self.decoded_logits = tfl.TimeDistributed(tfl.Dense(vocab_size, activation='linear'))
        super(custom_decoder, self).__init__(**kwargs)

    def call(self, inputs):
        h = self.rpv(inputs)
        h = self.lstm_layer_1(h)
        h = self.droplayer_2(h)
        h = self.lstm_layer_2(h)
        h = self.droplayer_3(h)
        decoded = self.decoded_logits(h)
        return decoded

    def compute_mask(self, inputs, mask=None):
        return mask

decoder_layer = custom_decoder(vocab_size, intr_dim, max_length, droprate)
decoded_logits = decoder_layer(z)

class ELBO_Layer(tfl.Layer):
    def __init__(self, **kwargs):
        super(ELBO_Layer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        broadcast_float_mask = tf.cast(mask, "float32")
        labels = tf.cast(x, tf.int32)
        reconstruction_loss = tf.reduce_sum(tfa.seq2seq.sequence_loss(inputs, labels, 
                                                                      weights=broadcast_float_mask,
                                                                      average_across_timesteps=False,
                                                                      average_across_batch=False), axis=1)

        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_weight * kl_loss)
        self.add_loss(total_loss, inputs=[x, inputs])
        return tf.ones_like(x)

    def compute_mask(self, inputs, mask=None):
        return mask

elbo_layer = ELBO_Layer()
fake_decoded_prob = elbo_layer(decoded_logits)

def zero_loss(y_true, y_pred):
    return tf.zeros_like(y_pred)

def kl_loss(x, fake_decoded_prob):
    kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=1)
    kl_loss = kl_weight * kl_loss
    return tf.reduce_mean(kl_loss)

vae = Model(x, fake_decoded_prob, name='VAE', trainable=False)
vae.trainable = False
PATH = '/users/apokkunu/trial/text/output5/' + 'weights.305-3.60.h5'
vae.load_weights(PATH)


encoder_text = Model(x, z, name='encoder', trainable=False)
encoder_text.trainable = False

ins = tfl.Input(shape=(latent_dim,))
x_logits = decoder_layer(ins)
decoder_text = Model(ins, x_logits, name='decoder', trainable=False)
decoder_text.trainable = False

####################################################################################


# In[ ]:

use_all_data = True
batch_size = 512


dataset_name = 'train'
train_capspath, train_imgspath = get_imgs(dataset_name)
train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)

dataset_name = 'val'
val_capspath, val_imgspath = get_imgs(dataset_name)
val_caps, val_imgs = load_caps_img(val_capspath, val_imgspath, dataset_name, use_all_data)

joint_caps = train_caps + val_caps
padded_seqs, vocab_size, word_index, index2word, tokenizer = clean_text_data(joint_caps)
train_padded_seqs, val_padded_seqs = train_test_split(padded_seqs, shuffle=False, test_size=len(val_caps)/len(joint_caps), random_state=28)

indexs = np.arange(len(train_imgs)).astype('float32').tolist()
train_combo = create_embd_dataset(train_imgs, train_padded_seqs, indexs, batch_size)

valindexs = np.arange(len(val_imgs)).astype('float32').tolist()
val_combo = create_embd_dataset(val_imgs, val_padded_seqs, valindexs, batch_size)


# In[ ]:


for (path, re_image), re_cap, indx in train_combo:
    z_image = encoder_image(re_image)
    z_text = encoder_text(re_cap)
    
    indexs = tf.expand_dims(indx, axis=1)
    combined = tf.concat([z_image, z_text, indexs], axis=1)
    
    for bf, p in zip(combined, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

print('Training set done!', flush=True)

# In[ ]:


for (path, re_image), re_cap, indx in val_combo:
    z_image = encoder_image(re_image)
    z_text = encoder_text(re_cap)
    
    indexs = tf.expand_dims(indx, axis=1)
    combined = tf.concat([z_image, z_text, indexs], axis=1)
    
    for bf, p in zip(combined, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())


print('Val set done!', flush=True)

