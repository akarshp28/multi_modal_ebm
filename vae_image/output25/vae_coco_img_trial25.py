#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from helper_functions2 import get_imgs, load_caps_img, create_image_gen, make_gallery, make_gallery2, plot_loss2

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


directory = "output25"
parent_dir = "/users/apokkunu/trial/img/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)

save_path = './' + directory + '/'
save = True

# In[4]:

use_all_data = True
batch_size = 128 # > 100 to draw image
max_epoch = 1000
latent_dim = 128
use_batch_norm = False
use_dropout = True
droprate = 0.3
kl_weight = 0.1
lr = 0.0002

# option_model = 'resnet'
# option_model = 'inception'
option_model = 'simple'

if option_model == 'resnet':
    model_type = tf.keras.applications.ResNet152V2
    preprocess = tf.keras.applications.resnet_v2
    img_dim1 = 49
    img_shape = 224 # 32 minimum
elif option_model == 'inception':
    model_type = tf.keras.applications.InceptionV3
    preprocess = tf.keras.applications.inception_v3
    img_dim1 = 64
    img_shape = 299 #75 minimum
else:
    model_type = None
    preprocess = None
    img_shape = 128


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

train_dataset = create_image_gen(train_imgs, batch_size, img_shape)
val_dataset = create_image_gen(val_imgs, batch_size, img_shape)
test_dataset = create_image_gen(test_imgs, batch_size, img_shape)

# # VAE

# In[8]:


class Sampling(tfl.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal([batch, dim])
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

input_dim = (img_shape, img_shape, 3)
encoder_inputs = tfl.Input(shape = input_dim, name = 'encoder_input')

conv_filters = [64, 128, 256, 512, 1024]
conv_kernel_size = [4, 4, 4, 4, 4]
conv_strides = [2, 2, 2, 2, 2]
n_layers = len(conv_filters)

x = encoder_inputs

for i in range(n_layers):
    x = tfl.Conv2D(filters = conv_filters[i], kernel_size = conv_kernel_size[i], strides = conv_strides[i], padding = 'same')(x)
    
    if use_batch_norm:
        x = tfl.BatchNormalization()(x)
    
    x = tfl.LeakyReLU()(x)
    
    if use_dropout:
        x = tfl.Dropout(rate=droprate)(x)

x_before = x.shape[1:]
x = tfl.Flatten()(x)
x_after = x.shape[1]
x = tfl.Dense(768, activation=tf.nn.leaky_relu)(x)

z_mean = tfl.Dense(latent_dim, name="z_mean")(x)
z_log_var = tfl.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# In[9]:


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
        x = sigmoid(x)

decoder = Model(latent_inputs, x, name="decoder")
decoder.summary()



# In[10]:


class VAE(Model):
    def __init__(self, encoder, decoder, optimizer):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = optimizer

        self.elbo = Mean(name="elbo")
        self.bce = Mean(name="reconstruction_loss")
        self.kld = Mean(name="kl_loss")
        
        self.elbo_val = Mean(name="elbo_val")
        self.bce_val = Mean(name="reconstruction_loss_val")
        self.kld_val = Mean(name="kl_loss_val")
        
        self.elbo_test = Mean(name="elbo_test")
        self.bce_test = Mean(name="reconstruction_loss_test")
        self.kld_test = Mean(name="kl_loss_test")
    
    @property
    def metrics(self):
        return [self.elbo, self.bce, self.kld, 
                self.elbo_val, self.bce_val, self.kld_val,
               self.elbo_test, self.bce_test, self.kld_test]

    @tf.function
    def train_step(self, data):
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
        
    def test_step(self, data, mode):
        self.encoder.trainable = False
        self.decoder.trainable = False

        z_mean, z_log_var, z = self.encoder.predict(data)
        reconstruction = self.decoder.predict(z)
        reconstruction = tf.convert_to_tensor(reconstruction, dtype=tf.float32)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), 
                                                           axis=(1, 2)))
        
        kl_loss = (-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var))) * kl_weight
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        
        if mode == 'val':
            self.elbo_val.update_state(total_loss)
            self.bce_val.update_state(reconstruction_loss)
            self.kld_val.update_state(kl_loss)
        else:
            self.elbo_test.update_state(total_loss)
            self.bce_test.update_state(reconstruction_loss)
            self.kld_test.update_state(kl_loss)

            
# # Visualization

# In[12]:

optimizer = tf.keras.optimizers.Adam(lr)
model = VAE(encoder, decoder, optimizer)


# In[9]:


elbo_train = []; bce_train = []; kld_train = [];
elbo_val = []; bce_val = []; kld_val = [];
elbo_test = []; bce_test = []; kld_test = [];
epoch = 0

print('\nModel Hyperparameters:', flush=True)
print('MAX_EPOCHS: {}, LR: {}, Dropout: {}, BS: {}, Z_DIM: {}, IMG_S: {}, KLW: {}'.format(max_epoch, lr, droprate, batch_size, latent_dim, img_shape, kl_weight), flush=True)
print('\n', flush=True)

while epoch < max_epoch:
    
    for batch in train_dataset:
        model.train_step(batch)
        
    for val_batch in val_dataset:
        model.test_step(val_batch, 'val')
        
    for test_batch in test_dataset:
        model.test_step(test_batch, 'test')
    
    elbo_train.append(model.elbo.result())
    bce_train.append(model.bce.result())
    kld_train.append(model.kld.result())
    
    elbo_val.append(model.elbo_val.result())
    bce_val.append(model.bce_val.result())
    kld_val.append(model.kld_val.result())
    
    elbo_test.append(model.elbo_test.result())
    bce_test.append(model.bce_test.result())
    kld_test.append(model.kld_test.result())
    
    print('\nEpoch: {}'.format(epoch), flush=True)
    print("Train ELBO: {}, Train BCE: {}, Train KLD: {}".format(model.elbo.result(), model.bce.result(), model.kld.result()), flush=True)
    print("Val ELBO: {}, Val BCE: {}, Val KLD: {}".format(model.elbo_val.result(), model.bce_val.result(), model.kld_val.result()), flush=True)
    print("Test ELBO: {}, Test BCE: {}, Test KLD: {}".format(model.elbo_test.result(), model.bce_test.result(), model.kld_test.result()), flush=True)
    
    if epoch % 10 == 0:
        if epoch > 0:
            plot_loss2(epoch, elbo_train, elbo_val, elbo_test, 'ELBO', save_path, save)
            plot_loss2(epoch, bce_train, bce_val, bce_test, 'BCE', save_path, save)
            plot_loss2(epoch, kld_train, kld_val, kld_test, 'KLD', save_path, save)
        
        mode = 'direct' #'vae'
        for test_batch in test_dataset.take(1):
            make_gallery(test_batch.numpy()[:100], 10, epoch, 'test_orig', save_path, save)
            make_gallery2(test_batch, model.encoder, model.decoder, mode, epoch, 'test_pred', save_path, save, latent_dim)
        
        model.encoder.save_weights(save_path + 'en_im_' + str(epoch) + '.h5')
        model.decoder.save_weights(save_path + 'de_im_' + str(epoch) + '.h5')
    
    model.elbo.reset_states();    model.bce.reset_states();    model.kld.reset_states()
    model.elbo_val.reset_states();    model.bce_val.reset_states();    model.kld_val.reset_states()
    model.elbo_test.reset_states();    model.bce_test.reset_states();    model.kld_test.reset_states()
    epoch += 1
    