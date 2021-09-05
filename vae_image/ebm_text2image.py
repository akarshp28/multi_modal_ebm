#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
# tf.config.experimental_run_functions_eagerly(True)

from nltk.translate.bleu_score import sentence_bleu
from helper_functions2 import *
import datetime
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# In[ ]:


use_all_data = True
img_shape = 128
latent_dim = 128

# EBM parameters
max_epoch = 500

ebm_units = 128
batch_size = 64
lr = 0.0001

inf_iter_train = 20
inf_rate_train = 0.01

inf_iter_val = 20
inf_rate_val = 0.01

inf_iter_test = 100
inf_rate_test = 0.01

gamma = 1
eta = 0.001
l2_reg = 0.001

directory = "ebm_t2i_5"
parent_dir = "/users/apokkunu/trial/img/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)

save_path = './' + directory + '/'
save = True


# In[ ]:

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


dataset_name = 'train'
train_capspath, train_imgspath = get_imgs(dataset_name)
train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)

dataset_name = 'val'
val_capspath, val_imgspath = get_imgs(dataset_name)
val_caps, val_imgs = load_caps_img(val_capspath, val_imgspath, dataset_name, use_all_data)

joint_caps = train_caps + val_caps
padded_seqs, vocab_size, word_index, index2word, tokenizer = clean_text_data(joint_caps)
train_padded_seqs, val_padded_seqs = train_test_split(padded_seqs, shuffle=False, test_size=len(val_caps)/len(joint_caps), random_state=28)

# use objects from here on to use for training
buffer_size = 10000
indexs = np.arange(len(train_imgs)).astype('float32').tolist()
train_combo = create_dataset(train_imgs, train_padded_seqs, indexs, batch_size, buffer_size)

valindexs = np.arange(len(val_imgs)).astype('float32').tolist()
val_combo = create_dataset(val_imgs, val_padded_seqs, valindexs, batch_size, buffer_size)

del train_caps, val_caps, joint_caps, padded_seqs, indexs, valindexs


# In[ ]:


def recon_text(logits):
    recons = tf.nn.softmax(decoder_text.predict_on_batch(logits))
    reconstructed_indexes = clean_sent(np.apply_along_axis(np.argmax, 1, recons[0]), 4)
    return list(np.vectorize(index2word.get)(reconstructed_indexes))
    
            
# In[ ]:


def norm(z):
    return 0.01 * tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(z, 2), axis=1)), axis=1)

def compute_energy(z, orig_z):
    norm_val = norm(z[:, :128] - orig_z)
    energy = ebm(z) + norm_val
    return energy, norm_val

def compute_gradient(z, orig_img):
    with tf.GradientTape() as tape:
        tape.watch(z)
        energy, norm_val = compute_energy(z, orig_img)
    return tape.gradient(energy, z), energy, norm_val

def langevin_inf(z, orig_img, inf_iter, inf_rate, mode, epoch):
    current_z = z
    orig_z = current_z[:, 128:] # if ld on images then save text
    
    if mode == 'val':
        z_arr = [current_z]
        
    for i in range(inf_iter):
        gradients, energy, norm_val = compute_gradient(current_z, orig_img)

        # Langevin dynamics
        term1 = 0.5 * inf_rate * gradients
        term2 = eta * tf.random.normal(current_z.get_shape().as_list())
        next_img = current_z - term1 + term2

        # reset back to orig z
        current_z = tf.concat([next_img[:, :128], orig_z], axis=1)
        
        if epoch % 5 == 0 and mode == 'train':
            template = "LD Step: {}, Avg. MSE: {}, Avg. energy: {}, Avg. Norm Z: {}"
            mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(decoder_image(orig_img), decoder_image(current_z[:, :128]))).numpy()
            print(template.format(i, mse, tf.reduce_mean(energy).numpy(), tf.reduce_mean(norm_val).numpy()), flush=True)
            
        if mode == 'val':
            z_arr.append(current_z)
    
    if mode == 'train':
        return current_z
    else:
        return current_z, np.array(z_arr)


# In[ ]:


latent_in = tfl.Input(shape=(256,))
x = tfl.Dense(ebm_units, activation=tf.nn.softplus)(latent_in)
energy_vals = tfl.Dense(1, use_bias=False)(x)
ebm = Model(latent_in, energy_vals, name="EBM")
ebm.summary()


# In[ ]:


class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = np.concatenate((np.random.uniform(0, 1, [size, 128]), np.expand_dims(np.arange(size), axis=1)), axis=1)
        self.maxsize = size
    
    def length(self):
        return len(self.storage)
    
    def return_storage(self):
        return self.storage
    
    def add(self, ims):
        ims = ims.numpy()
        for ind, i in enumerate(ims[:, 128]):
            n = np.equal(self.return_storage()[:, 128], i)
            condition = np.where(n)[0]
            if condition.shape[0] != 0:
                condition = np.repeat(np.transpose(np.expand_dims(n, axis=0)), 129, axis=1)
                self.storage = np.where(condition, ims[ind], self.return_storage())
            else:
                self.storage = np.vstack([self.storage[np.random.choice(self.length(), (self.length() - 1), replace=False)], ims[ind]])
    
    def sample_byindex(self, inds):
        temp = []
        for i in inds.numpy():
            try:
                temp.append(self.return_storage()[np.where(np.equal(self.return_storage()[:, 128], i))[0][0], :128])
            except IndexError:
                temp.append(np.random.uniform(0, 1, [128]))
        return tf.convert_to_tensor(temp, dtype=tf.float32)

    
# In[ ]:


class EBM(Model):
    def __init__(self, ebm, optimizer, encoder_image, decoder_image, encoder_text, decoder_text):
        super(EBM, self).__init__()
        self.ebm = ebm
        
        self.encoder_image = encoder_image
        self.decoder_image = decoder_image
        
        self.encoder_text = encoder_text
        self.decoder_text = decoder_text

        self.optimizer = optimizer

        self.train_loss_track = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.val_loss_track = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        
        self.re_train = tf.keras.metrics.Mean('re_train', dtype=tf.float32)
        self.unre_train = tf.keras.metrics.Mean('unre_train', dtype=tf.float32)
        self.lde_train = tf.keras.metrics.Mean('lde_train', dtype=tf.float32)
        
        self.re_val = tf.keras.metrics.Mean('re_val', dtype=tf.float32)
        self.unre_val = tf.keras.metrics.Mean('unre_val', dtype=tf.float32)
        self.lde_val = tf.keras.metrics.Mean('lde_val', dtype=tf.float32)
        
        self.mse_train = tf.keras.metrics.Mean('mse_train', dtype=tf.float32)
        self.mse_val = tf.keras.metrics.Mean('mse_val', dtype=tf.float32)
        self.mse_batch = tf.keras.metrics.Mean('mse_batch', dtype=tf.float32)
        
        self.train_loss_batch = tf.keras.metrics.Mean('train_loss_batch', dtype=tf.float32)
        self.re_batch = tf.keras.metrics.Mean('re_batch', dtype=tf.float32)
        self.unre_batch = tf.keras.metrics.Mean('unre_batch', dtype=tf.float32)
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
    
    def compute_loss(self, data, inf_iter, inf_rate, image_buffer, epoch):
        
        z_image = self.encoder_image(data[0])
        z_text = self.encoder_text(data[1])
        indexs = tf.expand_dims(data[2], axis=1)
        
        combined = tf.concat([z_image, z_text, indexs], axis=1)
        combined = tf.random.shuffle(combined)
        
        if z_image.shape[0] == batch_size:
            major_split = int(batch_size * 0.8)
            s1, s2 = combined[:major_split, :], combined[major_split:, :]
        else:
            major_split = int(z_image.shape[0] * 0.8)
            s1, s2 = combined[:major_split, :], combined[major_split:, :]
        
        # negative sampling training
        z = tf.concat([tf.random.shuffle(s1[:, :128]), s1[:, 128:256]], axis=1)
        term1, norm_val = compute_energy(z, s1[:, :128])
        
        # LD Image sampling
        if self.decision(0.95) == True:
            z = tf.concat([image_buffer.sample_byindex(s2[:, 256]), s2[:, 128:256]], axis=1)
        else:
            z = tf.concat([tf.random.uniform(s2[:, :128].get_shape().as_list()), s2[:, 128:256]], axis=1)
        z = langevin_inf(z, s2[:, :128], inf_iter_train, inf_rate_train, 'train', epoch)
        term2, norm_val = compute_energy(z, s2[:, :128])
        
        with tf.GradientTape() as tape:
            # training objective
            x_pos, norm_val = compute_energy(combined[:, :256], combined[:, :128])
            x_neg = tf.concat([term1, term2], axis=0)
            
            importance_weight = self.softmax(-gamma * x_neg)
            part1 = - 1 / gamma * self.logmeanexp(-gamma * x_pos) - tf.reduce_sum(x_neg * tf.stop_gradient(importance_weight))
            part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
            ebm_loss = part1 + part2
        grads = tape.gradient(ebm_loss, self.ebm.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ebm.trainable_weights))
        
        image_buffer.add(tf.concat([z[:, :128], tf.expand_dims(s2[:, 256], axis=1)], axis=1))
        
        ans = tf.reduce_mean(x_pos)
        self.re_batch.update_state(ans);          self.re_train.update_state(ans)
        
        ans = tf.reduce_mean(term1)
        self.unre_train.update_state(ans);        self.unre_batch.update_state(ans)
        
        ans = tf.reduce_mean(term2)
        self.lde_train.update_state(ans);         self.lde_batch.update_state(ans)
        
        self.train_loss_track.update_state(ebm_loss);        self.train_loss_batch.update_state(ebm_loss)
        
        ans = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.decoder_image(s2[:, :128]), self.decoder_image(z[:, :128])))
        self.mse_train.update_state(ans);        self.mse_batch.update_state(ans)
        
        
    def eval_loss(self, data, inf_iter, inf_rate, epoch):
        
        z_image = self.encoder_image(data[0])
        z_text = self.encoder_text(data[1])
        
        combined = tf.concat([z_image, z_text], axis=1)
        combined = tf.random.shuffle(combined)
        
        if z_image.shape[0] == batch_size:
            major_split = int(batch_size * 0.8)
            s1, s2 = combined[:major_split, :], combined[major_split:, :]
        else:
            major_split = int(z_image.shape[0] * 0.8)
            s1, s2 = combined[:major_split, :], combined[major_split:, :]
        
        # negative sampling training
        z = tf.concat([tf.random.shuffle(s1[:, :128]), s1[:, 128:]], axis=1)
        term1, norm_val = compute_energy(z, s1[:, :128])
        
        # LD Image sampling
        z = tf.concat([tf.random.uniform(s2[:, :128].get_shape().as_list()), s2[:, 128:]], axis=1)
        z, z_arr = langevin_inf(z, s2[:, :128], inf_iter_train, inf_rate_train, 'val', epoch)
        del z_arr
        term2, norm_val = compute_energy(z, s2[:, :128])
        
        # training objective
        x_pos, norm_val = compute_energy(combined, combined[:, :128])
        x_neg = tf.concat([term1, term2], axis=0)
        
        importance_weight = self.softmax(-gamma * x_neg)
        part1 = - 1 / gamma * self.logmeanexp(-gamma * x_pos) - tf.reduce_sum(x_neg * tf.stop_gradient(importance_weight))
        part2 = l2_reg * (tf.reduce_mean(tf.math.pow(x_pos, 2)) + tf.reduce_mean(tf.math.pow(x_neg, 2)))
        ebm_loss = part1 + part2
        
        self.re_val.update_state(tf.reduce_mean(x_pos))
        self.unre_val.update_state(tf.reduce_mean(term1))
        self.lde_val.update_state(tf.reduce_mean(term2))
        self.val_loss_track.update_state(ebm_loss)
        self.mse_val.update_state(tf.reduce_mean(tf.keras.losses.mean_squared_error(self.decoder_image(s2[:, :128]), self.decoder_image(z[:, :128]))))
    
    def test_ebm(self, data, epoch):
        z_image = self.encoder_image(data[0])
        z_text = self.encoder_text(data[1])

        name = 'image'
        z_ld, z_rr = langevin_inf(tf.concat([tf.random.uniform(z_image.get_shape().as_list()), z_text], axis=1), z_image,
                                                     inf_iter_test, inf_rate_test, 'val', epoch)
        
        # show results
        # image part
        decoded_image_orig = self.decoder_image.predict_on_batch(z_image)
        decoded_image_ld = self.decoder_image.predict_on_batch(z_ld[:, :128])
        draw3(data[0][0], decoded_image_orig[0], decoded_image_ld[0], save_path, save, epoch, name)
        print('reconstructed: ', ' '.join(recon_text(z_text)), flush=True)

        # ld image all
        img_arr = np.array([self.decoder_image(z_rr[i, :, :128]).numpy() for i in range(len(z_rr))])
        make_img(img_arr, 10, epoch, save_path, save, name)

        mse_loss_ld = tf.reduce_mean(tf.keras.losses.mean_squared_error(data[0], decoded_image_ld))
        mse_loss_vae = tf.reduce_mean(tf.keras.losses.mean_squared_error(data[0], decoded_image_orig))
        print('MSE VAE: {}, MSE LD: {}'.format(mse_loss_ld, mse_loss_vae))


# In[ ]:


image_buffer = ReplayBuffer(10000)
opt_ebm = tf.keras.optimizers.Adam(lr)
ebm_model = EBM(ebm, opt_ebm, encoder_image, decoder_image, encoder_text, decoder_text)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = save_path + 'logs/gradient_tape/' + current_time + '/train'
val_log_dir = save_path + 'logs/gradient_tape/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)


# In[ ]:


print('\n', flush=True)
print('MAX_EPOCHS: {}, LR: {}, BS: {}, Z_DIM: {}, IMG_S: {}'.format(max_epoch, lr, batch_size, latent_dim, img_shape), flush=True)
print('Inf Rate Train: {}, Inf Iter Train: {}'.format(inf_rate_train, inf_iter_train), flush=True)
print('Inf Rate Val: {}, Inf Iter Val: {}'.format(inf_rate_val, inf_iter_val), flush=True)
print('Eta: {}, L2 Reg: {}, Gamma: {}'.format(eta, l2_reg, gamma), flush=True)
print('\n', flush=True)


# In[ ]:


train_loss = []; val_loss = []
re_energy_tr = []; unre_energy_tr = []; ld_energy_tr = []
re_energy_val = []; unre_energy_val = []; ld_energy_val = []
epoch = 0

while epoch < max_epoch:
    print("\nEpoch:", epoch, flush=True)
    start = timer()
    
    print('Training EBM', flush=True)
    for re_image, re_cap, indx in train_combo:
        ebm_model.compute_loss([re_image, re_cap, indx], inf_iter_train, inf_rate_train, image_buffer, epoch)
        
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss Batch', ebm_model.train_loss_batch.result(), step=epoch)
            tf.summary.scalar('Re-Energy Batch', ebm_model.re_batch.result(), step=epoch)
            tf.summary.scalar('Unre-Energy Batch', ebm_model.unre_batch.result(), step=epoch)
            tf.summary.scalar('LD-Energy Batch', ebm_model.lde_batch.result(), step=epoch)
            tf.summary.scalar('MSE Batch', ebm_model.mse_batch.result(), step=epoch)
        
        if epoch % 5 == 0:
            print('Loss Batch: {}, MSE Batch: {}'.format(ebm_model.train_loss_batch.result(), ebm_model.mse_batch.result()), flush=True)

            print("Re-Energy Batch: {}, Unre-Energy Batch: {}, LD-Energy Batch: {}".format(ebm_model.re_batch.result(), 
                                                                                           ebm_model.unre_batch.result(), 
                                                                                           ebm_model.lde_batch.result()), flush=True)
        ebm_model.re_batch.reset_states()
        ebm_model.unre_batch.reset_states()
        ebm_model.lde_batch.reset_states()
        ebm_model.train_loss_batch.reset_states()
        ebm_model.mse_batch.reset_states()
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    print('Validating EBM', flush=True)
    for re_image, re_cap, indx in val_combo:
        ebm_model.eval_loss([re_image, re_cap, indx], inf_iter_val, inf_rate_val, epoch)
        tf.keras.backend.clear_session()
        gc.collect()
        
    train_loss.append(ebm_model.train_loss_track.result());    val_loss.append(ebm_model.val_loss_track.result())
    re_energy_tr.append(ebm_model.re_train.result());   unre_energy_tr.append(ebm_model.unre_train.result());   ld_energy_tr.append(ebm_model.lde_train.result())
    re_energy_val.append(ebm_model.re_val.result());    unre_energy_val.append(ebm_model.unre_val.result());    ld_energy_val.append(ebm_model.lde_val.result())
    
    if epoch % 5 == 0:
        if epoch > 0:
            plot_loss3(epoch, train_loss, val_loss, 'EBM', save_path, save)
            plot_loss5(epoch, re_energy_tr, unre_energy_tr, ld_energy_tr, re_energy_val, unre_energy_val, ld_energy_val, 
                       'Energy', save_path, save, 're_tr', 'unre_tr', 'ld_tr', 're_val', 'unre_val', 'ld_val')
            ebm.save_weights(save_path + 'ebm_' + str(epoch) + '.h5')
    
    if epoch % 10 == 0:
        for re_image, re_cap, indx in val_combo.take(1):
            print('Example Image generation', flush=True)
            ebm_model.test_ebm([re_image, re_cap], epoch)
            tf.keras.backend.clear_session()
            gc.collect()
    
    with train_summary_writer.as_default():
        tf.summary.scalar('Train Loss', ebm_model.train_loss_track.result(), step=epoch)
        tf.summary.scalar('Re-Energy Train', ebm_model.re_train.result(), step=epoch)
        tf.summary.scalar('Unre-Energy Train', ebm_model.unre_train.result(), step=epoch)
        tf.summary.scalar('LD-Energy Train', ebm_model.lde_train.result(), step=epoch)
        tf.summary.scalar('MSE Train', ebm_model.mse_train.result(), step=epoch)
    
    with val_summary_writer.as_default():
        tf.summary.scalar('Val Loss', ebm_model.val_loss_track.result(), step=epoch)
        tf.summary.scalar('Re-Energy Train', ebm_model.re_val.result(), step=epoch)
        tf.summary.scalar('Unre-Energy Train', ebm_model.unre_val.result(), step=epoch)
        tf.summary.scalar('LD-Energy Train', ebm_model.lde_val.result(), step=epoch)
        tf.summary.scalar('MSE Val', ebm_model.mse_val.result(), step=epoch)
    
    print("Train Loss: {}, Val Loss: {}".format(ebm_model.train_loss_track.result(), ebm_model.val_loss_track.result()), flush=True)
    
    print("Re-Energy Train: {}, Unre-Energy Train: {}, LD-Energy Train: {}".format(ebm_model.re_train.result(), 
                                                                                   ebm_model.unre_train.result(), 
                                                                                   ebm_model.lde_train.result()), flush=True)
    
    print("Re-Energy Val: {}, Unre-Energy Val: {}, LD-Energy Val: {}".format(ebm_model.re_val.result(), 
                                                                             ebm_model.unre_val.result(), 
                                                                             ebm_model.lde_val.result()), flush=True)
    
    print("MSE Train: {}, MSE Val: {}".format(ebm_model.mse_train.result(), ebm_model.mse_val.result()), flush=True)
    
    ebm_model.train_loss_track.reset_states()
    ebm_model.val_loss_track.reset_states()
    
    ebm_model.mse_train.reset_states()
    ebm_model.mse_val.reset_states()
    
    ebm_model.re_train.reset_states()
    ebm_model.unre_train.reset_states()
    ebm_model.lde_train.reset_states()
    
    ebm_model.re_val.reset_states()
    ebm_model.unre_val.reset_states()
    ebm_model.lde_val.reset_states()
    
    epoch += 1
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    end = timer()
    print('Loop Time: ', end - start, 'secs', flush=True)
    del start, end


# In[ ]:


for re_image, re_cap, indx in val_combo.take(1):
    ebm_model.test_ebm([re_image, re_cap], epoch)
    tf.keras.backend.clear_session()
    gc.collect()