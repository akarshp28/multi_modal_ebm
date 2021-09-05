#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import json
import numpy as np
import tensorflow as tf
from scipy import spatial
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import train_test_split


# In[2]:

epoch = 500
lr = 0.0001
batch_size = 128

intr_dim = 256
latent_dim = int(intr_dim * 0.5)

droprate = 0.2
kl_weight = 0.01

max_length = 40 #time steps
emb_dim = 300

BASE_DIR = './annotations/'
use_all_data = True
num_sent = 10

directory = "output6"
parent_dir = "/users/apokkunu/trial/text/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
    

# In[ ]:


def load_data(dataset, use_all_data, num_sent, BASE_DIR):
    if 'train' in dataset:
        path = BASE_DIR + 'captions_train2014.json'
    else:
        path = BASE_DIR + 'captions_val2014.json'

    with open(path, 'r') as f:
        annotations = json.load(f)
    
    captions = []
    for c in annotations['annotations']:
        caption = f"<start> {c['caption']} <end>"
        captions.append(caption)
        
    if use_all_data:
        captions = captions
        print('Total data size: ', len(captions), flush=True)
    else:
        captions = captions[:num_sent]
        print('Temp data size: ', len(captions), flush=True)    
        
    return captions

def create_emb_ind(glove_path, emb_dim):
    total_path = glove_path + 'glove.6B.' + str(emb_dim) + 'd.txt'
    f = open(total_path, encoding='utf8')
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index), flush=True)
    return embeddings_index

def make_emb_mat(embeddings_index, total_words, emb_dim, word_index, dataset, save):
    glove_embedding_matrix = np.zeros((total_words, emb_dim))
    for word, i in word_index.items():
        if i < total_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                glove_embedding_matrix[i] = embedding_vector
            else:
                # if words not found in embedding index will be the word embedding of 'unk'
                glove_embedding_matrix[i] = embeddings_index.get('unk')

    print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0), flush=True)
    print('Check Null: ', np.isnan(np.sum(glove_embedding_matrix)), flush=True)
    print('Emb vector shape: ', glove_embedding_matrix.shape, flush=True)
    
    if save:
        np.save('./annotations/' + dataset + '.npy', glove_embedding_matrix)
    return glove_embedding_matrix


# In[3]:


train_captions = load_data('train', use_all_data, num_sent, BASE_DIR)
val_captions = load_data('val', use_all_data, num_sent, BASE_DIR)
joint_list = train_captions + val_captions
print(len(joint_list), flush=True)


# In[4]:

top_k = 15000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="unk", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(joint_list)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

word_index = tokenizer.word_index
index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index), flush=True)

sequences = tokenizer.texts_to_sequences(joint_list)

data_1 = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')
print('Shape of data tensor:', data_1.shape, flush=True)

vocab_size = top_k
print(vocab_size, flush=True)

# In[5]:


create = True
if create:
    embeddings_index = create_emb_ind(BASE_DIR, emb_dim)
    glove_embedding_matrix = make_emb_mat(embeddings_index, vocab_size, emb_dim, word_index, 'train', True)
else:
    glove_embedding_matrix = np.load(BASE_DIR + 'train.npy')
    print(glove_embedding_matrix.shape, flush=True)


# In[6]:


x_train, x_val = train_test_split(data_1, shuffle=False, test_size=len(val_captions)/len(joint_list), random_state=28)
print(x_train.shape, x_val.shape, flush=True)


# In[8]:


####################################################################################

class Sampling(tf.keras.layers.Layer):
    def __init__(self):
        super(Sampling, self).__init__()
        self.supports_masking = True
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.random.normal([batch, latent_dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class custom_lstm(tf.keras.layers.Layer):
    def __init__(self, intr_dim, droprate,  **kwargs):
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(intr_dim, recurrent_dropout=droprate, 
                                                                          return_sequences=False), merge_mode='concat')
        self.drop_layer = tf.keras.layers.Dropout(droprate)
        super(custom_lstm, self).__init__(**kwargs)
    
    def call(self, inputs):
        h = self.bi_lstm(inputs)
        h = self.drop_layer(h)
        return h
    
    def compute_mask(self, inputs, mask=None):
        return mask
    
x = tf.keras.layers.Input(shape=(max_length,))
embed_layer = tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=max_length, weights=[glove_embedding_matrix], 
                                        trainable=False, mask_zero=True)
encoder_layer = custom_lstm(intr_dim, droprate)

h = embed_layer(x)
h = encoder_layer(h)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(h)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(h)
z = Sampling()([z_mean, z_log_var])

####################################################################################

class custom_decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, intr_dim, max_length, droprate, **kwargs):
        self.rpv = tf.keras.layers.RepeatVector(max_length)
        self.lstm_layer_1 = tf.keras.layers.LSTM(intr_dim, return_sequences=True, recurrent_dropout=droprate)
        self.droplayer_2 = tf.keras.layers.Dropout(droprate)
        self.lstm_layer_2 = tf.keras.layers.LSTM(intr_dim*2, return_sequences=True, recurrent_dropout=droprate)
        self.droplayer_3 = tf.keras.layers.Dropout(droprate)
        self.decoded_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='linear'))
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

####################################################################################

class ELBO_Layer(tf.keras.layers.Layer):
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

####################################################################################

def zero_loss(y_true, y_pred):
    return tf.zeros_like(y_pred)

def kl_loss(x, fake_decoded_prob):
    kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var), axis=1)
    kl_loss = kl_weight * kl_loss
    return tf.reduce_mean(kl_loss)

vae = tf.keras.models.Model(x, fake_decoded_prob, name='VAE')
opt = tf.keras.optimizers.Adam(lr=lr)
vae.compile(optimizer=opt, loss=[zero_loss], metrics=[kl_loss])
vae.summary()

for i, l in enumerate(vae.layers):
    print(f'layer {i}: {l}', flush=True)
    print(f'has input mask: {l.input_mask}', flush=True)
    print(f'has output mask: {l.output_mask}', flush=True)

####################################################################################
    

# In[ ]:

print('\n', flush=True)

print('Sent Length: {}, Vocab size: {}, store_dir: {}'.format(max_length, vocab_size, directory), flush=True)

print('Epochs: {}, BS: {}, LR: {}, EMB DIM: {}, Z_DIM: {}, INTR_DIM: {}, Dropout: {}, KLW: {}'.format(epoch, batch_size, 
                                                                                                     lr, emb_dim, latent_dim, 
                                                                                                     intr_dim, droprate, 
                                                                                                     kl_weight), flush=True)
print('\n', flush=True)


def create_model_checkpoint(directory):
    if os.path.isdir(directory) == False:
        os.mkdir(path)
    filepath = path + '/' + "weights.{epoch:02d}-{val_loss:.2f}.h5"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, monitor='val_loss',
                                                      save_best_only=True, mode='min', save_weights_only=True)
    return checkpointer

checkpointer = create_model_checkpoint(directory)

callback_es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

history = vae.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=epoch, 
        batch_size=batch_size, shuffle=True, callbacks=[callback_es, checkpointer], verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./' + directory + '/plot_history.png')
