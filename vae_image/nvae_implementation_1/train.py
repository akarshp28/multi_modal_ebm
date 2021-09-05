#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import random
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf

from nvae_model import NVAE
from helper_functions2 import get_imgs, load_caps_img, create_image_gen, make_gallery, plot_loss2


directory = "output2"
parent_dir = "/users/apokkunu/trial/img/nvae/"
path = os.path.join(parent_dir, directory)
if os.path.isdir(directory) == False:
    os.mkdir(path)
save_path = './' + directory + '/'
save = True


# In[ ]:


image_shape = 128
use_all_data = False

max_epoch = 500                 # Max Epochs
batch_size = 64                 # Batch size
lr = 0.01                      # Initial learning rate

n_encoder_channels = 8         # Number of initial channels in encoder
n_decoder_channels = 8         # Number of initial channels in decoder

res_cells_per_group = 2         # Number of residual cells to use within each group

n_preprocess_blocks = 2         # Number of blocks to use in the preprocessing layers
n_preprocess_cells = 3          # Number of cells to use within each preprocessing block

n_postprocess_blocks = 2        # Number of blocks to use in the postprocessing layers
n_postprocess_cells = 3         # Number of cells to use within each postprocessing block

n_latent_per_group = 10         # Number of latent stochastic variables to sample in each group
n_groups_per_scale = [4, 4, 4]    # Number of groups to include in each resolution scale

sr_lambda = 0.01                 # Spectral regularisation strength
scale_factor = 2                # Factor to rescale image with in each scaling step
step_based_warmup = False       # Base warmup on batches trained instead of epochs

use_multigpu = False
drop_remain = False
dataset = 'coco'


# In[ ]:


dataset_name = 'train'
train_capspath, train_imgspath = get_imgs(dataset_name)
train_caps, train_imgs = load_caps_img(train_capspath, train_imgspath, dataset_name, use_all_data)

dataset_name = 'val'
val_capspath, val_imgspath = get_imgs(dataset_name)
val_caps, val_imgs = load_caps_img(val_capspath, val_imgspath, dataset_name, use_all_data)

dataset_name = 'test'
test_capspath, test_imgspath = get_imgs(dataset_name)
test_imgs = load_caps_img(test_capspath, test_imgspath, dataset_name, use_all_data)

train_dataset = create_image_gen(set(train_imgs), batch_size, image_shape, drop_remain)
val_dataset = create_image_gen(set(val_imgs), batch_size, image_shape, drop_remain)
test_dataset = create_image_gen(set(test_imgs), batch_size, image_shape, drop_remain)
    
    
def save_reconstructions(epoch, model, test_data, dataset, name):
    reconstruction_logits, *_ = model(test_data)
    
    if dataset == 'mnist':
        distribution = distributions.Bernoulli(logits=reconstruction_logits, dtype=tf.float32, allow_nan_stats=False)
        images = distribution.mean()
    else:
        images = model.sample_from_discretized_mix_logistic(reconstruction_logits)
        
    make_gallery(images.numpy(), 10, epoch, name, color_mode, save_path, save)
    

# In[ ]:


batches_per_epoch = (len(set(train_imgs)) + batch_size - 1) // batch_size
lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=lr, decay_steps = max_epoch * batches_per_epoch)
optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

model = NVAE(n_encoder_channels = n_encoder_channels,
             n_decoder_channels = n_decoder_channels,
             res_cells_per_group = res_cells_per_group,
             n_preprocess_blocks = n_preprocess_blocks,
             n_preprocess_cells = n_preprocess_cells,
             n_postprocess_blocks= n_postprocess_blocks,
             n_post_process_cells = n_postprocess_cells,
             n_latent_per_group = n_latent_per_group,
             n_latent_scales = len(n_groups_per_scale),
             n_groups_per_scale = n_groups_per_scale,
             sr_lambda = sr_lambda,
             scale_factor = scale_factor,
             n_total_iterations = (batches_per_epoch * max_epoch),  # for balance kl
             step_based_warmup = step_based_warmup,
             input_shape = (image_shape, image_shape, 3),
             dataset_option = dataset,
             use_multigpu = use_multigpu,
             GLOBAL_BATCH_SIZE = batch_size,
             optimizer = optimizer)


# Begin Training 

elbo_train = []; bce_train = []; kld_train = []; bn_train = []
elbo_val = []; bce_val = []; kld_val = []; bn_val = []
elbo_test = []; bce_test = []; kld_test = []; bn_test = []
mse_tr = []; mse_vl=[]; mse_tst = []

epoch = 0
best_mse = 1000000
color_mode = True
best_epoch = 0

print('\nHyperparameters:', flush=True)
print("Max Epochs: {}, BatchSize: {}, Batches per Epoch: {}, Initial LR: {}".format(max_epoch, batch_size, batches_per_epoch, lr))
print("EN CH: {}, DEC CH: {}".format(n_encoder_channels, n_decoder_channels))
print("Pre-process Blocks: {}, Pre-process Cells: {}".format(n_preprocess_blocks, n_preprocess_cells))
print("Post-process Blocks: {}, Post-process Cells: {}".format(n_postprocess_blocks, n_postprocess_cells))
print("Num Groups: {}, Latent Vars per Group: {}, Res Blocks per Group: {}".format(n_groups_per_scale, n_latent_per_group, res_cells_per_group))
print("Spectral Reg: {}, IMG Scale: {}, Step based KL-Warmup: {}, Multi_GPU: {}".format(sr_lambda, scale_factor, step_based_warmup, use_multigpu))

while epoch < max_epoch:
    print('\nEpoch: {}'.format(epoch), flush=True)
    start = timer()
    
    print('Training NVAE', flush=True)
    step = 0
    for x in train_dataset:
        step += 1
        model.train_step(x, step, epoch)
    
    print('Validating NVAE', flush=True)
    step = 0
    for x in val_dataset:
        step += 1
        model.test_step(x, step, epoch, 'val')

    print('Testing NVAE', flush=True)
    step = 0
    for x in test_dataset:
        step += 1
        model.test_step(x, step, epoch, 'test')

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

    if model.mse_val.result().numpy() < best_mse:
        best_mse = mse_val_value
        best_epoch = epoch
        model.save_weights(save_path + 'nvae_' + str(epoch) + '.tf')

    temp = "Train ELBO: {}, Train BCE: {}, Train KLD: {}, Train BN: {}"
    print(temp.format(elbo_tr_value, bce_tr_value, kld_tr_value, bnl_tr_value), flush=True)
    
    temp = "Val ELBO: {}, Val BCE: {}, Val KLD: {}, Val BN: {}"
    print(temp.format(elbo_val_value, bce_val_value, kld_val_value, bnl_val_value), flush=True)

    temp = "Test ELBO: {}, Test BCE: {}, Test KLD: {}, Test BN: {}"
    print(temp.format(elbo_test_value, bce_test_value, kld_test_value, bnl_test_value), flush=True)

    print("Train MSE: {}, Val MSE: {}, Test MSE: {}".format(mse_tr_value, mse_val_value, mse_test_value), flush=True)
    print("Best MSE: {}, Best Epoch: {}".format(best_mse, best_epoch), flush=True)

    if epoch % 10 == 0:
        if epoch > 0:
            plot_loss2(epoch, elbo_train, elbo_val, elbo_test, 'ELBO', save_path, save)
            plot_loss2(epoch, bce_train, bce_val, bce_test, 'BCE', save_path, save)
            plot_loss2(epoch, kld_train, kld_val, kld_test, 'KLD', save_path, save)
            plot_loss2(epoch, bn_train, bn_val, bn_test, 'BN', save_path, save)
            plot_loss2(epoch, mse_tr, mse_vl, mse_tst, 'MSE', save_path, save)

        # samples from p(z)
        for temperature in [0.7, 0.8, 0.9, 1.0]:
            images, *_ = model.sample(100, temperature=temperature)
            make_gallery(images.numpy(), 10, epoch, 'pz_samples_temp_' + str(temperature) + '_', color_mode, save_path, save)

        # samples from q(z|x)
        for test_batch in test_dataset.take(1):
            if batch_size > 64:
                make_gallery(test_batch.numpy()[:100], 10, epoch, 'orig_', color_mode, save_path, save)
                save_reconstructions(epoch, model, test_batch[:100], dataset, 'pred_')
            elif batch_size == 32:
                ncols = 10; num_imgs = 30
                make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'orig_', color_mode, save_path, save)
                save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'pred_')
            elif batch_size < 32:
                ncols = 10; num_imgs = 10
                make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'orig_', color_mode, save_path, save)
                save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'pred_')
            else:
                ncols = 10; num_imgs = 50
                make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'orig_', color_mode, save_path, save)
                save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'pred_')

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

# samples from p(z)
for temperature in [0.7, 0.8, 0.9, 1.0]:
    images, *_ = model.sample(100, temperature=temperature)
    make_gallery(images.numpy(), 10, epoch, 'final_pz_samples_temp_' + str(temperature) + '_', color_mode, save_path, save)

# samples from q(z|x)
for test_batch in test_dataset.take(1):
    if batch_size > 64:
        make_gallery(test_batch.numpy()[:100], 10, epoch, 'final_orig_', color_mode, save_path, save)
        save_reconstructions(epoch, model, test_batch[:100], dataset, 'final_pred_')
    elif batch_size == 32:
        ncols = 10; num_imgs = 30
        make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'final_orig_', color_mode, save_path, save)
        save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'final_pred_')
    elif batch_size < 32:
        ncols = 10; num_imgs = 10
        make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'final_orig_', color_mode, save_path, save)
        save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'final_pred_')
    else:
        ncols = 10; num_imgs = 50
        make_gallery(test_batch.numpy()[:num_imgs], ncols, epoch, 'final_orig_', color_mode, save_path, save)
        save_reconstructions(epoch, model, test_batch[:num_imgs], dataset, 'final_pred_')