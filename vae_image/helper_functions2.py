#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import json
import random
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# # Download Data

# In[2]:


def get_imgs(dataset_name, option='load'):
    image_folder = '/users/apokkunu/trial/img/' + dataset_name + '2014/'
    annotation_folder = '/users/apokkunu/trial/img/annotations/'

    if option == 'DL_BOTH':
        if not os.path.exists(os.path.abspath('.') + image_folder):
            image_zip = tf.keras.utils.get_file(dataset_name + '2014.zip',
                                              cache_subdir=os.path.abspath('.'),
                                              origin = 'http://images.cocodataset.org/zips/' + dataset_name + '2014.zip',
                                              extract = True)
            IMGPATH = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
        
        if not os.path.exists(os.path.abspath('.') + annotation_folder):
            annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                  cache_subdir=os.path.abspath('.'),
                                                  origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                  extract = True)
            
            CAPPATH = os.path.dirname(annotation_zip) + annotation_folder + 'captions_' + dataset_name + '2014.json'
            os.remove(annotation_zip)
            
        print(CAPPATH, IMGPATH)
            
    elif option == 'DL_IMG':
        if not os.path.exists(os.path.abspath('.') + image_folder):
            image_zip = tf.keras.utils.get_file(dataset_name + '2014.zip',
                                              cache_subdir=os.path.abspath('.'),
                                              origin = 'http://images.cocodataset.org/zips/' + dataset_name + '2014.zip',
                                              extract = True)
            IMGPATH = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
            
        if 'test' in dataset_name:
            CAPPATH = annotation_folder + 'image_info_' + dataset_name + '2014.json'
        else:
            CAPPATH = annotation_folder + 'captions_' + dataset_name + '2014.json'
        print(CAPPATH, IMGPATH)
    
    else:
        IMGPATH = image_folder
        
        if 'test' in dataset_name:
            CAPPATH = annotation_folder + 'image_info_' + dataset_name + '2014.json'
        else:
            CAPPATH = annotation_folder + 'captions_' + dataset_name + '2014.json'
        
        print(CAPPATH, IMGPATH)
    return CAPPATH, IMGPATH


# In[ ]:


def load_caps_img(caption_path, img_path, dataset_name, use_all_data, num_imgs=100):
    # load annotation file
    with open(caption_path, 'r') as f:
        annotations = json.load(f)
    
    if 'test' in dataset_name:
        image_paths = []
        for val in annotations['images']:
            image_path = img_path + 'COCO_' + dataset_name + '2014_' + '%012d.jpg' % (val['id'])
            image_paths.append(image_path)

        # shuffle data
        random.shuffle(image_paths)
        print('\ndataset size:', len(image_paths), flush=True)

        # randomly select some images if not all
        if use_all_data:
            train_image_paths = image_paths
        else:
            train_image_paths = image_paths[:num_imgs]
        print('\ntemp dataset size:', len(train_image_paths), flush=True)

        # return path names of selected images
        img_name_vector = [i for i in train_image_paths]
        
        return img_name_vector
        
    else:
        # Group all captions together having the same image ID.
        image_path_to_caption = collections.defaultdict(list)
        for val in annotations['annotations']:
            caption = f"<start> {val['caption']} <end>"
            image_path = img_path + 'COCO_' + dataset_name + '2014_' + '%012d.jpg' % (val['image_id'])
            image_path_to_caption[image_path].append(caption)
        
        # shuffle data
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)
        print('\ndataset size:', len(image_paths), flush=True)

        # randomly select some images if not all
        if use_all_data:
            train_image_paths = image_paths
        else:
            train_image_paths = image_paths[:num_imgs]
        print('\ntemp dataset size:', len(train_image_paths), flush=True)

        # return path names of selected images
        train_captions = [];    img_name_vector = []
        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))
        
    return train_captions, img_name_vector


# # Visualization

# In[3]:

save_file_type = '.png'

def plot_loss(epoch, train_loss, name, save_path, save):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(train_loss), label = "train")
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()

def plot_loss2(epoch, train_loss, val_loss, test_loss, name, save_path, save):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(train_loss), label = "train")
    plt.plot(np.arange(epoch+1), np.array(val_loss), label = "val")
    plt.plot(np.arange(epoch+1), np.array(test_loss), label = "test")
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    
def plot_loss3(epoch, train_loss, val_loss, name, save_path, save):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(train_loss), label = "train")
    plt.plot(np.arange(epoch+1), np.array(val_loss), label = "val")
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    
def plot_loss4(epoch, train_loss, name, save_path, save):
    plt.figure()
    plt.plot(np.arange(len(train_loss)), np.array(train_loss), label = "train")
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    
def plot_loss5(epoch, loss1, loss2, loss3, loss4, loss5, loss6, name, save_path, save, label1, label2, label3, label4, label5, label6):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(loss1), label = label1)
    plt.plot(np.arange(epoch+1), np.array(loss2), label = label2)
    plt.plot(np.arange(epoch+1), np.array(loss3), label = label3)
    
    plt.plot(np.arange(epoch+1), np.array(loss4), label = label4)
    plt.plot(np.arange(epoch+1), np.array(loss5), label = label5)
    plt.plot(np.arange(epoch+1), np.array(loss6), label = label6)
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    
    
def plot_loss6(epoch, loss1, loss2, loss3, name, save_path, save, label1, label2, label3):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(loss1), label = label1)
    plt.plot(np.arange(epoch+1), np.array(loss2), label = label2)
    plt.plot(np.arange(epoch+1), np.array(loss3), label = label3)
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    
def plot_loss7(epoch, loss1, loss2, loss3, loss4, name, save_path, save, label1, label2, label3, label4):
    plt.figure()
    plt.plot(np.arange(epoch+1), np.array(loss1), label = label1)
    plt.plot(np.arange(epoch+1), np.array(loss2), label = label2)
    plt.plot(np.arange(epoch+1), np.array(loss3), label = label3)
    plt.plot(np.arange(epoch+1), np.array(loss4), label = label4)
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    if save:
        plt.savefig(save_path + "loss_" + name + "_" + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()
    

def make_gallery(data, ncols, epoch, name, color_mode, save_path, save):
    nindex, height, width, intensity = data.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    arr = (data.reshape(nrows, ncols, height, width, intensity)).swapaxes(1,2)
    img_gallery = arr.reshape(height*nrows, width*ncols, intensity)
    plt.figure(figsize=(ncols, ncols))
    
    if color_mode == True:
        plt.imshow(img_gallery)
    else:
        plt.imshow(img_gallery, cmap=plt.cm.gray)
    plt.xticks([]); plt.yticks([])
    
    if save:
        plt.savefig(save_path + name + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close()

def make_img(data, ncols, epoch, color_mode, save_path, save, name):
    data = data[:, 0, :, :, :]
    make_gallery(data[1:], ncols, epoch, 'ld_step_' + name, color_mode, save_path, save)
    
def predict_direct(model_en, model_de, data):
    model_en.trainable = False
    model_de.trainable = False
    z_mean, z_var, z = model_en.predict(data)
    reconstruction = model_de.predict(z)
    return reconstruction

def plot_landscape_sum(cum_energy, cum_zmag, save_path, save):
    flat_e = sum(cum_energy, [])
    flat_z = sum(cum_zmag, [])
    cum_e = np.squeeze(np.array(flat_e))
    cum_z = np.squeeze(np.array(flat_z))
    plot_loss(len(cum_e)-1, cum_e, 'energy values', save_path, save)
    plot_loss(len(cum_z)-1, cum_z, 'z mag', save_path, save)
    
def plot_landscape(epoch, cum_energy, cum_zmag, save_path, save, name):
    plot_loss4(epoch, cum_energy, 'energy_' + name, save_path, save)
    plot_loss4(epoch, cum_zmag, 'z_mag_' + name, save_path, save)

def make_gallery2(test_data, model_en, model_de, mode, epoch, name, color_mode, save_path, save, latent_dim, ncols=10, num_imgs=100):
    if mode == 'LD':
        x_d = []
        cum_energy = []
        cum_zmag = []
        for i in range(num_imgs):
            z = tf.random.normal([1, latent_dim])
            x_decoded, ld_images, energy_arrs, z_mag = predict_inf(model_de, z, inf_iter, inf_rate)
            x_d.append(x_decoded)
            
            cum_energy.append(energy_arrs)
            cum_zmag.append(z_mag)
            
            if i == 0:
                # plot energy landscape
                plot_landscape(cum_energy, cum_zmag, save_path, save)
        
        array = np.array(x_d)[:, 0, :, :, :]
        
        print('\nStep wise LD Output', flush=True)
        make_img(ld_images, ncols, epoch, name, color_mode, save_path, save)
    
    elif mode == 'direct':
        # predict using test data
        array = predict_direct(model_en, model_de, test_data.numpy()[:num_imgs])[:num_imgs, :, :, :]
        array = array[:num_imgs]
        
    else:
        # normal vae predict
        x_d = []
        for i in range(num_imgs):
            z = tf.random.normal([1, latent_dim])
            x_decoded = model_de.predict(z)
            x_d.append(x_decoded)
        array = np.array(x_d)[:, 0, :, :, :]
        
        draw(array[0], epoch, name, save_path, save)
    
    make_gallery(array, ncols, epoch, name, color_mode, save_path, save)
 
    
def draw(image, epoch, name, save_path, save):
    fig, ax = plt.subplots(1,1) 
    ax.axis('off')
    ax.imshow(image)
    if save:
        plt.savefig(save_path + "single_" + name + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close(fig)


def draw2(image, recon, color_mode, save_path, save, epoch, name):
    fig, (ax1,ax2) = plt.subplots(1,2) 
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image)
    ax2.imshow(recon)
    if save:
        plt.savefig(save_path + "comparison_" + name + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close(fig)

def draw3(image, recon, ld, color_mode, save_path, save, epoch, name):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3) 
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    if color_mode == True:
        ax1.imshow(image)
        ax2.imshow(recon)
        ax3.imshow(ld)
    else:
        ax1.imshow(image, cmap=plt.cm.gray)
        ax2.imshow(recon, cmap=plt.cm.gray)
        ax3.imshow(ld, cmap=plt.cm.gray)
    if save:
        plt.savefig(save_path + "comparison_" + name + str(epoch) + save_file_type)
    else:
        plt.show()
    plt.close(fig)
    
    

# # Transfer Learning

# In[4]:
# def load_image(image_path, img_shape, preprocess):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, (img_shape, img_shape))
#     img = preprocess.preprocess_input(img)
#     return img

# def create_image_gen_tl(img_paths, batch_size, img_shape, preprocess, buffer_size=10000, autotune=False):
#     encode_train = sorted(set(img_paths))
#     print('\ndataset size:', len(encode_train), flush=True)
    
#     dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    
#     if autotune:
#         dataset = dataset.map(lambda x: load_image(x, img_shape, preprocess), num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.shuffle(buffer_size).batch(batch_size)
#         dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#     else:
#         dataset = dataset.map(lambda x: load_image(x, img_shape, preprocess), num_parallel_calls=8)
#         dataset = dataset.shuffle(buffer_size).batch(batch_size)
#         dataset = dataset.prefetch(buffer_size=buffer_size)
#     return dataset


# # Dataset Preparation

def simple_load_image(image_path, img_shape=128):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_shape, img_shape))
    img = tf.math.divide(img, 255.0)
    return img

def map_func_new(image_path, cap, indexs):
    img_tensor = simple_load_image(image_path)
    return (image_path, img_tensor), cap, indexs

def create_embd_dataset(img_paths, padded_seqs, indexs, batch_size, buffer_size=10000):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, padded_seqs, indexs))
    dataset = dataset.map(lambda item1, item2, item3: map_func_new(item1, item2, item3), num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset

def map_func_embd_train(image_path):
    return np.load(image_path.decode('utf-8') + '.npy')

def map_func_embd_val(image_path):
    embd = np.load(image_path.decode('utf-8') + '.npy')
    img_tensor = simple_load_image(image_path)
    return (embd, img_tensor)

def create_tfgenerator(img_paths, batch_size, mode, buffer_size=10000):
    if mode == 'train':
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset = dataset.map(lambda item1: tf.numpy_function(map_func_embd_train, [item1], [tf.float32]), 
                              num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset = dataset.map(lambda item1: tf.numpy_function(map_func_embd_train, [item1], (tf.float32)), 
                              num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset



# In[5]:


def clean_text_data(captions_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="unk", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions_list)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    word_index = tokenizer.word_index
    index2word = {v: k for k, v in word_index.items()}
    print('Found %s unique tokens' % len(word_index), flush=True)

    sequences = tokenizer.texts_to_sequences(captions_list)

    padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50, padding='post')
    print('Shape of data tensor:', padded_seqs.shape, flush=True)

    vocab_size = len(word_index)
    print('vocab size:', vocab_size, flush=True)
    return padded_seqs, vocab_size, word_index, index2word, tokenizer

# def create_text_gen(padded_seqs, batch_size, buffer_size=10000):
#     dataset = tf.data.Dataset.from_tensor_slices(padded_seqs)
#     dataset = dataset.shuffle(buffer_size).batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=buffer_size)
#     return dataset

def simple_load_image(image_path, img_shape=128):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_shape, img_shape))
    img = tf.math.divide(img, 255.0)
    return img

def map_func(image_path, cap, indexs):
    img_tensor = simple_load_image(image_path)
    return img_tensor, cap, indexs

def create_image_gen(img_paths, batch_size, img_shape, drop_remain = False, buffer_size=10000, autotune=False):
    encode_train = sorted(set(img_paths))
    print('\ndataset size:', len(encode_train), flush=True)

    dataset = tf.data.Dataset.from_tensor_slices(encode_train)

    if autotune:
        dataset = dataset.map(lambda x: simple_load_image(x, img_shape), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=drop_remain)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: simple_load_image(x, img_shape), num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=drop_remain)
        dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset

def create_dataset(img_paths, padded_seqs, indexs, batch_size, buffer_size=10000):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, padded_seqs, indexs))
    dataset = dataset.map(lambda item1, item2, item3: map_func(item1, item2, item3), num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset

# def create_combo_gen(img_paths, padded_seqs, batch_size, buffer_size=10000):
#     dataset = tf.data.Dataset.from_tensor_slices((img_paths, padded_seqs))
#     dataset = dataset.map(map_func, num_parallel_calls=8).shuffle(buffer_size).batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=buffer_size)
#     return dataset

# def map_func_ultra(re_image_path, re_cap, un_image_path, un_cap):
#     re_img_tensor = simple_load_image(re_image_path)
#     un_img_tensor = simple_load_image(un_image_path)
#     return re_img_tensor, re_cap, un_img_tensor, un_cap

# def all_combo(re_img_paths, re_padded_seqs, un_img_paths, un_padded_seqs, batch_size, buffer_size=10000):
#     dataset = tf.data.Dataset.from_tensor_slices((re_img_paths, re_padded_seqs, un_img_paths, un_padded_seqs))
#     dataset = dataset.map(map_func_ultra, num_parallel_calls=8).shuffle(buffer_size).batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=buffer_size)
#     return dataset

# def map_func_ultra(re_image_path, re_cap, un_image_path, un_cap, re_image_path1, re_cap1, un_image_path1, un_cap1):
#     re_img_tensor = simple_load_image(re_image_path)
#     un_img_tensor = simple_load_image(un_image_path)
    
#     re_img_tensor1 = simple_load_image(re_image_path1)
#     un_img_tensor1 = simple_load_image(un_image_path1)
#     return re_img_tensor, re_cap, un_img_tensor, un_cap, re_img_tensor1, re_cap1, un_img_tensor1, un_cap1

# def all_combo(re_img_paths, re_padded_seqs, un_img_paths, un_padded_seqs, re_img_paths1, re_padded_seqs1, un_img_paths1, un_padded_seqs1, batch_size, buffer_size=10000):
#     dataset = tf.data.Dataset.from_tensor_slices((re_img_paths, re_padded_seqs, un_img_paths, un_padded_seqs, 
#                                                   re_img_paths1, re_padded_seqs1, un_img_paths1, un_padded_seqs1))
#     dataset = dataset.map(map_func_ultra, num_parallel_calls=8).shuffle(buffer_size).batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=buffer_size)
#     return dataset




# # Test time cleaning

# In[6]:



def clean_sent(indexs, ind):
    reconstructed_ind = []
    for i in indexs:
        reconstructed_ind.append(i)
        if i == ind: # just remove the padding while printing output sentence
            break
    return reconstructed_ind
