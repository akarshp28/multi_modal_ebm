#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Tuple
from functools import partial
from util import calculate_log_p, softclamp5
from common import RescaleType, Rescaler, SqueezeExcitation

from enum import Enum, auto

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras import activations, Sequential, layers
from tensorflow_addons.layers import SpectralNormalization


# In[2]:


# image_shape = (128, 128, 3)
# n_encoder_channels = 32
# n_decoder_channels = 32

# n_preprocess_blocks = 2
# n_preprocess_cells = 3

# n_postprocess_blocks = 2
# n_postprocess_cells = 3

# mult = 1
# scale_factor = 2

# n_latent_per_group = 20
# res_cells_per_group = 2
# n_groups_per_scale = [4, 4, 4]
# n_latent_scales = len(n_groups_per_scale)


# # Preprocess

# In[3]:


class SkipScaler(tf.keras.Model):
    def __init__(self, n_channels, **kwargs):
        super().__init__(**kwargs)
        
        # Each convolution handles a quarter of the channels
        self.conv1 = SpectralNormalization(layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same"))
        self.conv2 = SpectralNormalization(layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same"))
        self.conv3 = SpectralNormalization(layers.Conv2D(n_channels // 4, (1, 1), strides=(2, 2), padding="same"))
        
        # This convolotuion handles the remaining channels
        self.conv4 = SpectralNormalization(layers.Conv2D(n_channels - 3 * (n_channels // 4), (1, 1), strides=(2, 2), padding="same"))

    def call(self, x):
        out = activations.swish(x)
        # Indexes are offset as we stride by 2x2, this way we cover all pixels
        conv1 = self.conv1(out)
        conv2 = self.conv2(out[:, 1:, 1:, :])
        conv3 = self.conv3(out[:, :, 1:, :])
        conv4 = self.conv4(out[:, 1:, :, :])
        
        # Combine channels
        out = tf.concat((conv1, conv2, conv3, conv4), axis=3)
        return out


class BNSwishConv(tf.keras.Model):
    def __init__(self, n_nodes, n_channels, stride, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.nodes = Sequential()
        
        if stride == (1, 1):
            self.skip = tf.identity
        elif stride == (2, 2):
            # We have to rescale the input in order to combine it
            self.skip = SkipScaler(n_channels)
            
        for i in range(n_nodes):
            self.nodes.add(layers.BatchNormalization(momentum=0.05, epsilon=1e-5))
            self.nodes.add(layers.Activation(activations.swish))
            
            # Only apply rescaling on first node
            self.nodes.add(SpectralNormalization(layers.Conv2D(n_channels, (3, 3), stride if i == 0 else (1, 1), padding="same")))
            
        self.se = SqueezeExcitation()
    
    def call(self, inputs):
        skipped = self.skip(inputs)
        x = self.nodes(inputs)
        x = self.se(x)
        return skipped + 0.1 * x
    
def pre_process_block(image_shape, n_encoder_channels, n_preprocess_blocks, n_preprocess_cells, scale_factor, mult=1):
    # input is expected to be in [-1, 1] range
    in_put = layers.Input(shape=image_shape, name='image')
    x = SpectralNormalization(layers.Conv2D(n_encoder_channels, (3, 3), padding="same"))(in_put)
    
    for block in range(n_preprocess_blocks):
        for cell in range(n_preprocess_cells - 1):
            n_channels = mult * n_encoder_channels
            x = BNSwishConv(2, n_channels, stride=(1, 1))(x)
        
        # Rescale channels on final cell
        n_channels = mult * n_encoder_channels * scale_factor
        x = BNSwishConv(2, n_channels, stride=(2, 2))(x)
        mult *= scale_factor
    
    model = tf.keras.models.Model(in_put, x, name='preprocess')
    model.summary()
    
    get_out_shape = x.shape[1:]
    print(mult, get_out_shape)
    return model, mult, get_out_shape


# # Encoder

# In[4]:


def create_encoder_layers(pp_mult, n_latent_scales, n_groups_per_scale, res_cells_per_group, n_encoder_channels, scale_factor):
    # create encoder layers as a list
    mult = pp_mult
    enc_layers = []
    for scale in range(n_latent_scales):
        n_groups = n_groups_per_scale[scale]
        print('\nGroup: ', scale)

        for group_idx in range(n_groups):
            output_channels = n_encoder_channels * mult
            print('Output_channels: ', output_channels)

            for rb in range(res_cells_per_group):
                enc_layers.append(EncodingResidualCell(output_channels, name='res_block_' + str(scale) + '_' + str(group_idx) + '_' + str(rb)))
                print('res block')

            if not (scale == n_latent_scales - 1 and group_idx == n_groups - 1):
                print('combiner')
                enc_layers.append(EncoderDecoderCombiner(output_channels))
        
        # We downsample in the end of each scale except last
        if scale < n_latent_scales - 1:
            output_channels = n_encoder_channels * mult * scale_factor
            enc_layers.append(Rescaler(output_channels, scale_factor=scale_factor, rescale_type=RescaleType.DOWN))
            print('Rescaler')
            print('New output_channels: ', output_channels)
            mult *= scale_factor

    enc_layers.append(layers.ELU())
    enc_layers.append(SpectralNormalization(layers.Conv2D(n_encoder_channels * mult, (1, 1), padding="same")))
    enc_layers.append(layers.ELU())
    
    return enc_layers, mult


class EncodingResidualCell(tf.keras.Model):
    """Encoding network residual cell in NVAE architecture"""
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm1 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.conv1 = SpectralNormalization(layers.Conv2D(output_channels, (3, 3), padding="same"))
        self.batch_norm2 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.conv2 = SpectralNormalization(layers.Conv2D(output_channels, (3, 3), padding="same"))
        self.se = SqueezeExcitation()

    def call(self, inputs):
        x = activations.swish(self.batch_norm1(inputs))
        x = self.conv1(x)
        x = activations.swish(self.batch_norm2(x))
        x = self.conv2(x)
        x = self.se(x)
        return 0.1 * inputs + x
    

class EncoderDecoderCombiner(tf.keras.Model):
    def __init__(self, n_channels, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decoder_conv = SpectralNormalization(layers.Conv2D(n_channels, (1, 1)))

    def call(self, encoder_x, decoder_x):
        x = self.decoder_conv(decoder_x)
        return encoder_x + x


# # Sampler

# In[5]:


class Sampling(layers.Layer):
    def call(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class Sampler(tf.keras.Model):
    def __init__(self, n_latent_scales, n_groups_per_scale, n_latent_per_group, scale_factor, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Initialize sampler
        self.enc_sampler = []
        self.dec_sampler = []
        self.n_latent_scales = n_latent_scales
        self.n_groups_per_scale = n_groups_per_scale
        self.n_latent_per_group = n_latent_per_group
        
        for scale in range(self.n_latent_scales):
            n_groups = self.n_groups_per_scale[scale]
            
            for group in range(n_groups):
                # NVLabs use padding 1 here?
                self.enc_sampler.append(SpectralNormalization(layers.Conv2D(2 * self.n_latent_per_group, kernel_size=(3, 3), padding="same")))
                
                if scale == 0 and group == 0:
                    # Dummy value to maintain indexing
                    self.dec_sampler.append(None)
                else:
                    sampler = Sequential()
                    sampler.add(layers.ELU())
                    
                    # NVLabs use padding 0 here?
                    sampler.add(SpectralNormalization(layers.Conv2D(2 * self.n_latent_per_group, kernel_size=(1, 1))))
                    self.dec_sampler.append(sampler)
    
    def sample(self, mu, sigma):
        # reparametrization trick
        return Sampling()(mu, sigma)

    def get_params(self, sampler, z_idx, prior):
        params = sampler[z_idx](prior)
        mu, log_sigma = tf.split(params, 2, axis=-1)
        return mu, log_sigma
    
    def call(self, prior, z_idx, enc_prior=None):
        # Get encoder offsets
        if enc_prior is None:
            enc_prior = prior
        enc_mu_offset, enc_log_sigma_offset = self.get_params(self.enc_sampler, z_idx, enc_prior)
        
        if z_idx == 0:
            # Prior is standard normal distribution
            enc_mu = softclamp5(enc_mu_offset)
            enc_sigma = tf.math.exp(softclamp5(enc_log_sigma_offset)) + 1e-2
            z = self.sample(enc_mu, enc_sigma)
            params = [enc_mu, enc_sigma, tf.zeros_like(enc_mu), tf.ones_like(enc_sigma)]
            return z, params
        
        # Get decoder parameters
        raw_dec_mu, raw_dec_log_sigma = self.get_params(self.dec_sampler, z_idx, prior)
        
        dec_mu = softclamp5(raw_dec_mu)
        dec_sigma = tf.math.exp(softclamp5(raw_dec_log_sigma)) + 1e-2
        
        enc_mu = softclamp5(enc_mu_offset + raw_dec_mu)
        enc_sigma = (tf.math.exp(softclamp5(raw_dec_log_sigma + enc_log_sigma_offset)) + 1e-2)
        
        params = [enc_mu, enc_sigma, dec_mu, dec_sigma]
        z = self.sample(enc_mu, enc_sigma)
        return z, params


class RescaleType(Enum):
    UP = auto()
    DOWN = auto()


class SqueezeExcitation(tf.keras.Model):
    """Squeeze and Excitation block as defined by Hu, et al. (2019)
    See Also
    ========
    Source paper https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, ratio=16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        batch_size, h, w, c = input_shape
        self.gap = layers.GlobalAveragePooling2D(data_format="channels_last")
        num_hidden = max(c / self.ratio, 4)
        self.dense1 = layers.Dense(units=num_hidden)
        self.dense2 = layers.Dense(units=c)

    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = activations.relu(x)
        x = self.dense2(x)
        x = activations.sigmoid(x)
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 2)
        return x * inputs


class Rescaler(tf.keras.Model):
    def __init__(self, n_channels, scale_factor, rescale_type, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.bn = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.mode = rescale_type
        self.factor = scale_factor
        
        if rescale_type == RescaleType.UP:
            self.conv = SpectralNormalization(layers.Conv2D(n_channels, (3, 3), strides=(1, 1), padding="same"))
            
        elif rescale_type == RescaleType.DOWN:
            self.conv = SpectralNormalization(layers.Conv2D(n_channels, (3, 3), strides=(self.factor, self.factor), padding="same"))

    def call(self, input):
        x = self.bn(input)
        x = activations.swish(x)
        
        if self.mode == RescaleType.UP:
            _, height, width, _ = x.get_shape()
            x = tf.image.resize(x, size=(self.factor * height, self.factor * width), method="nearest")
        x = self.conv(x)
        return x


# # Decoder

# In[6]:


def create_decoder_layers(encoder_mult, n_latent_scales, n_groups_per_scale, res_cells_per_group, n_decoder_channels, scale_factor):
    m = encoder_mult
    dec_layers = []
    
    for scale in range(n_latent_scales):
        print('\nGroup: ', scale)
        n_groups = n_groups_per_scale[scale]

        for group in range(n_groups):
            if scale == group == 0:
                output_channels = int(n_decoder_channels * m)
            print('Output channels', output_channels)

            if not (scale == 0 and group == 0):
                for res in range(res_cells_per_group):
                    dec_layers.append(GenerativeResidualCell(output_channels))
                    print('Gen Res block', flush=True)

            dec_layers.append(DecoderSampleCombiner(output_channels))
            print('Decoder Combiner', flush=True)

        if scale < n_latent_scales - 1:
            output_channels = int(n_decoder_channels * m / scale_factor)

            dec_layers.append(Rescaler(output_channels, scale_factor=scale_factor, rescale_type=RescaleType.UP))
            print('Rescaler', flush=True)

            m /= scale_factor
    return dec_layers, m


class DecoderSampleCombiner(tf.keras.Model):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        
        self.conv = SpectralNormalization(layers.Conv2D(output_channels, (1, 1), strides=(1, 1), padding="same"))
    
    def call(self, x, z):
        output = tf.concat((x, z), axis=3)
        output = self.conv(output)
        return output


class GenerativeResidualCell(tf.keras.Model):
    """Generative network residual cell in NVAE architecture"""
    def __init__(self, output_channels, expansion_ratio=6, **kwargs):
        super().__init__(**kwargs)
        
        self.batch_norm1 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.conv1 = SpectralNormalization(layers.Conv2D(expansion_ratio * output_channels, (1, 1), padding="same"))
        self.batch_norm2 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.depth_conv = layers.DepthwiseConv2D((5, 5), padding="same")
        self.batch_norm3 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.conv2 = SpectralNormalization(layers.Conv2D(output_channels, (1, 1), padding="same"))
        self.batch_norm4 = layers.BatchNormalization(momentum=0.05, epsilon=1e-5)
        self.se = SqueezeExcitation()

    def call(self, inputs):
        x = self.batch_norm1(inputs)
        x = self.conv1(x)
        x = activations.swish(self.batch_norm2(x))
        x = self.depth_conv(x)
        x = activations.swish(self.batch_norm3(x))
        x = self.conv2(x)
        x = self.batch_norm4(x)
        x = self.se(x)
        return 0.1 * inputs + x


# In[7]:


def create_enc_dec_model(pp_mult, pp_output_shape, n_latent_per_group, n_latent_scales, n_groups_per_scale, res_cells_per_group, 
                         n_encoder_channels, n_decoder_channels, scale_factor, nll=False):
    
    # create tensorflow encoder/decoder end-to-end model
    x_in = layers.Input(shape=pp_output_shape, name='encoder_input')

    #############################################################
    # Encoder
    print('\nENCODER')
    #############################################################
    
    enc_layers, mult = create_encoder_layers(pp_mult, n_latent_scales, n_groups_per_scale, res_cells_per_group, n_encoder_channels, scale_factor)
    x = enc_layers[0](x_in)
    enc_dec_combiners = []
    for group in enc_layers[1:]:
        if isinstance(group, EncoderDecoderCombiner):
            # We are stepping between groups, need to save results
            enc_dec_combiners.append([group, x])
        else:
            x = group(x)
    enc_dec_combiners.reverse()
    en_out_shape = x.shape[1:]
    
    #############################################################
    # Decoder
    print('\nDECODER')
    #############################################################
    
    # call latent sapce sampler class
    sampler = Sampler(n_latent_scales=n_latent_scales, 
                      n_groups_per_scale=n_groups_per_scale, 
                      n_latent_per_group=n_latent_per_group, 
                      scale_factor=scale_factor)
    
    # create decoder layers
    dec_layers, decoder_mult = create_decoder_layers(mult, n_latent_scales, n_groups_per_scale, res_cells_per_group, n_decoder_channels, scale_factor)
    
    z_params = []
    
    if nll:
        all_log_p = []
        all_log_q = []
    
    z0, params = sampler(x, z_idx=0)
    z_params.append(params)
    
    if nll:
        all_log_q.append(calculate_log_p(z0, params.enc_mu, params.enc_sigma))
        all_log_p.append(calculate_log_p(z0, params.dec_mu, params.dec_sigma))
    
    z0_shape = tf.convert_to_tensor([en_out_shape[0], en_out_shape[1], n_latent_per_group], dtype=tf.int32)
    
    h_var = tf.Variable(tf.random.uniform([en_out_shape[0], en_out_shape[1], n_decoder_channels], minval=0, maxval=1), trainable=True)
    h = tf.expand_dims(h_var, 0)
    h = tf.tile(h, [tf.shape(z0)[0], 1, 1, 1])
    
    x = dec_layers[0](h, z0)
    
    combine_idx = 0
    for group in dec_layers[1:]:
        if isinstance(group, DecoderSampleCombiner):
            enc_prior = enc_dec_combiners[combine_idx][0](enc_dec_combiners[combine_idx][1], x)
            z_sample, params = sampler(x, z_idx=combine_idx + 1, enc_prior=enc_prior)
            
            if nll:
                all_log_q.append(calculate_log_p(z_sample, params.enc_mu, params.enc_sigma))
                all_log_p.append(calculate_log_p(z_sample, params.dec_mu, params.dec_sigma))
            
            z_params.append(params)
            x = group(x, z_sample)
            combine_idx += 1
        else:
            x = group(x)
    
    if nll:
        log_p = tf.zeros((tf.shape(x)[0]))
        log_q = tf.zeros((tf.shape(x)[0]))
        
        for p, q in zip(all_log_p, all_log_q):
            log_p += tf.reduce_sum(p, axis=[1, 2, 3])
            log_q += tf.reduce_sum(q, axis=[1, 2, 3])
        
        model_decoder = tf.keras.models.Model(x_in, [x, z_params, log_p, log_q], name='decoder')
        model_decoder.summary()
    else:
        model_decoder = tf.keras.models.Model(x_in, [x, z_params], name='decoder')
        model_decoder.summary()
    
    return model_decoder, int(decoder_mult), x.shape[1:], z0_shape, sampler, h_var, dec_layers


# # Post process

# In[8]:


class PostprocessCell(tf.keras.Model):
    def __init__(self, n_channels, n_nodes, scale_factor, upscale=False, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.sequence = Sequential()
        
        if upscale:
            self.skip = Rescaler(n_channels, scale_factor=scale_factor, rescale_type=RescaleType.UP)
        else:
            self.skip = tf.identity
            
        for _ in range(n_nodes):
            self.sequence.add(PostprocessNode(n_channels, upscale=upscale, scale_factor=scale_factor))
            
            if upscale:
                # Only scale once in each cells
                upscale = False

    def call(self, inputs):
        return self.skip(inputs) + 0.1 * self.sequence(inputs)

class PostprocessNode(tf.keras.Model):
    def __init__(self, n_channels, scale_factor, upscale=False, expansion_ratio=6, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.sequence = Sequential()
        
        if upscale:
            self.sequence.add(Rescaler(n_channels, scale_factor, rescale_type=RescaleType.UP))
            
        self.sequence.add(layers.BatchNormalization(momentum=0.05, epsilon=1e-5))
        hidden_dim = n_channels * expansion_ratio
        self.sequence.add(ConvBNSwish(hidden_dim, kernel_size=(1, 1), stride=(1, 1)))
        self.sequence.add(ConvBNSwish(hidden_dim, kernel_size=(5, 5), stride=(1, 1)))
        self.sequence.add(SpectralNormalization(layers.Conv2D(n_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False)))
        self.sequence.add(layers.BatchNormalization(momentum=0.05, epsilon=1e-5))
        self.sequence.add(SqueezeExcitation())

    def call(self, inputs):
        return self.sequence(inputs)


class ConvBNSwish(tf.keras.Model):
    def __init__(self, n_channels, kernel_size, stride, groups=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence = Sequential()
        self.sequence.add(SpectralNormalization(layers.Conv2D(n_channels, kernel_size=kernel_size, strides=stride, use_bias=False, padding="same")))
        self.sequence.add(layers.BatchNormalization(momentum=0.05, epsilon=1e-5))
        self.sequence.add(layers.Activation(activations.swish))

    def call(self, inputs):
        return self.sequence(inputs)
    
    
def create_post_process_layers(mult, n_decoder_channels, n_postprocess_blocks, n_postprocess_cells, scale_factor, dataset_option='coco'):
    print('\nPost-process')
    sequence = []
    for block in range(n_postprocess_blocks):
        # First cell rescales
        mult /= scale_factor
        output_channels = n_decoder_channels * mult

        for cell_idx in range(n_postprocess_cells):
            print('add post process cell')
            sequence.append(PostprocessCell(output_channels, n_nodes=1, upscale=cell_idx == 0, scale_factor=scale_factor))

    sequence.append(layers.Activation(activations.elu))
    if dataset_option == 'mnist':
        sequence.append(SpectralNormalization(layers.Conv2D(1, kernel_size=(3, 3), padding="same")))
    else:
        sequence.append(SpectralNormalization(layers.Conv2D(100, kernel_size=(3, 3), padding="same"))) # 10 logistic distributions
    print('add elu-conv')
    
    return sequence

def post_process(input_shapes, n_postprocess_blocks, n_postprocess_cells, n_decoder_channels, scale_factor, mult):
    # input_shapes and mult output comes from decoder output
    post_layers = create_post_process_layers(mult, n_decoder_channels, n_postprocess_blocks, n_postprocess_cells, scale_factor)
    
    in_put = layers.Input(shape=input_shapes, name='post_process')
    x = post_layers[0](in_put)
    
    for layer in post_layers[1:]:
        x = layer(x)
    
    model = tf.keras.models.Model(in_put, x, name='post_process')
    model.summary()
    return model


# In[9]:


def main_(image_shape, n_preprocess_blocks, n_preprocess_cells, n_postprocess_blocks, n_postprocess_cells, 
          n_encoder_channels, n_decoder_channels, n_latent_per_group, n_latent_scales, n_groups_per_scale, res_cells_per_group, scale_factor):

    # preprocess
    model_preprocess, pp_mult, pp_output_shape = pre_process_block(image_shape, 
                                                                   n_encoder_channels, 
                                                                   n_preprocess_blocks, 
                                                                   n_preprocess_cells, 
                                                                   scale_factor)
    
    
    # encoder / decoder
    model_decoder, decoder_mult, decoder_out_shape, z0_shape, sampler, h_var, dec_layers = create_enc_dec_model(pp_mult, 
                                                                                                                pp_output_shape, 
                                                                                                                n_latent_per_group, 
                                                                                                                n_latent_scales, 
                                                                                                                n_groups_per_scale, 
                                                                                                                res_cells_per_group,
                                                                                                                n_encoder_channels, 
                                                                                                                n_decoder_channels, 
                                                                                                                scale_factor)
    
    
    # post process
    model_postprocess = post_process(decoder_out_shape, 
                                     n_postprocess_blocks, 
                                     n_postprocess_cells, 
                                     n_decoder_channels, 
                                     scale_factor, 
                                     decoder_mult)
    
    return model_preprocess, model_decoder, model_postprocess


# In[10]:


# model_preprocess, model_decoder, model_postprocess = main_(image_shape, 
#                                                            n_preprocess_blocks, 
#                                                            n_preprocess_cells, 
#                                                            n_postprocess_blocks, 
#                                                            n_postprocess_cells, 
#                                                            n_encoder_channels, 
#                                                            n_decoder_channels, 
#                                                            n_latent_per_group,
#                                                            n_latent_scales, 
#                                                            n_groups_per_scale, 
#                                                            res_cells_per_group, 
#                                                            scale_factor)


# In[11]:


class NVAE(tf.keras.Model):
    def __init__(self,
                 n_encoder_channels,
                 n_decoder_channels,
                 res_cells_per_group,
                 n_preprocess_blocks,
                 n_preprocess_cells,
                 n_postprocess_blocks,
                 n_post_process_cells,
                 n_latent_per_group,
                 n_latent_scales,
                 n_groups_per_scale,
                 sr_lambda,
                 scale_factor,
                 n_total_iterations,
                 step_based_warmup,
                 input_shape,
                 dataset_option,
                 use_multigpu,
                 GLOBAL_BATCH_SIZE,
                 optimizer,
                 **kwargs):
        super().__init__(**kwargs)
        
        # multi gpu params
        self.use_multigpu = use_multigpu
        self.GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE
        
        # Updated for each gradient pass, training step
        self.sr_lambda = sr_lambda
        self.step_based_warmup = step_based_warmup
        
        self.optimizer = optimizer
        
        self.n_latent_per_group = n_latent_per_group
        self.n_latent_scales = n_latent_scales
        self.n_groups_per_scale = n_groups_per_scale
        self.n_total_iterations = n_total_iterations
        self.dataset_option = dataset_option
        
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
        
        # preprocess
        model_preprocess, pp_mult, pp_output_shape = pre_process_block(input_shape, 
                                                                       n_encoder_channels, 
                                                                       n_preprocess_blocks, 
                                                                       n_preprocess_cells, 
                                                                       scale_factor)

        # encoder / decoder
        model_decoder, decoder_mult, decoder_out_shape, z0_shape, sampler, h_var, dec_layers = create_enc_dec_model(pp_mult, 
                                                                                                                    pp_output_shape, 
                                                                                                                    n_latent_per_group, 
                                                                                                                    n_latent_scales, 
                                                                                                                    n_groups_per_scale,
                                                                                                                    res_cells_per_group,
                                                                                                                    n_encoder_channels, 
                                                                                                                    n_decoder_channels, 
                                                                                                                    scale_factor)
        # post process
        model_postprocess = post_process(decoder_out_shape, 
                                         n_postprocess_blocks, 
                                         n_post_process_cells, 
                                         n_decoder_channels, 
                                         scale_factor, 
                                         decoder_mult)
        
        self.model_preprocess = model_preprocess
        self.model_decoder = model_decoder
        self.model_postprocess = model_postprocess
        self.z0_shape = z0_shape
        self.sampler = sampler
        self.h_var = h_var
        self.dec_layers = dec_layers
    
    # main method
    def call(self, inputs):
        x = self.model_preprocess(inputs)
        x, z_params = self.model_decoder(x)
        reconstruction = self.model_postprocess(x)
        return reconstruction, z_params
    
    def train_step(self, data, steps, epoch):
        if self.dataset_option =='mnist':
            if isinstance(data, tuple):
                data = data[0] #Remove the label.
        
        with tf.GradientTape() as tape:
            reconstruction_logits, z_params = self(data)
            
            recon_loss = self.calculate_recon_loss(data, reconstruction_logits)
            bn_loss = self.calculate_bn_loss()
            
            # warming up KL term for first 30% of training
            warmup_metric = steps if self.step_based_warmup else epoch
            beta = min(warmup_metric / (0.3 * self.n_total_iterations), 1)
            activate_balancing = beta < 1
            kl_loss = beta * self.calculate_kl_loss(z_params, activate_balancing)
            
            if self.use_multigpu:
                loss = tf.nn.compute_average_loss(recon_loss + kl_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)
            else:
                loss = tf.math.reduce_mean(recon_loss + kl_loss)
            total_loss = loss + bn_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.elbo.update_state(total_loss)
        self.bce.update_state(recon_loss)
        self.kld.update_state(kl_loss)
        self.bnloss.update_state(bn_loss)
        
        reconstruction = self.sample_from_discretized_mix_logistic(reconstruction_logits)
        mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
        if self.use_multigpu:
            self.mse_train.update_state(tf.nn.compute_average_loss(mse, global_batch_size=self.GLOBAL_BATCH_SIZE))
        else:
            self.mse_train.update_state(tf.reduce_mean(mse))
            
            
    def test_step(self, data, steps, epoch, mode):
        reconstruction_logits, z_params, *_ = self(data)
        recon_loss = self.calculate_recon_loss(data, reconstruction_logits)
        bn_loss = self.calculate_bn_loss()
        
        # warming up KL term for first 30% of training
        warmup_metric = steps if self.step_based_warmup else epoch
        beta = min(warmup_metric / (0.3 * self.n_total_iterations), 1)
        activate_balancing = beta < 1
        kl_loss = beta * self.calculate_kl_loss(z_params, activate_balancing)
        
        if self.use_multigpu:
            loss = tf.nn.compute_average_loss(recon_loss + kl_loss, global_batch_size=self.GLOBAL_BATCH_SIZE)
        else:
            loss = tf.math.reduce_mean(recon_loss + kl_loss)
        total_loss = loss + bn_loss
        
        reconstruction = self.sample_from_discretized_mix_logistic(reconstruction_logits)
        
        if mode == 'test':
            self.elbo_test.update_state(total_loss)
            self.bce_test.update_state(recon_loss)
            self.kld_test.update_state(kl_loss)
            self.bnloss_test.update_state(bn_loss)
            mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
            if self.use_multigpu:
                self.mse_test.update_state(tf.nn.compute_average_loss(mse, global_batch_size=self.GLOBAL_BATCH_SIZE))
            else:
                self.mse_test.update_state(tf.reduce_mean(mse))
        else:
            self.elbo_val.update_state(total_loss)
            self.bce_val.update_state(recon_loss)
            self.kld_val.update_state(kl_loss)
            self.bnloss_val.update_state(bn_loss)
            mse = tf.reduce_sum(tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2))
            if self.use_multigpu:
                self.mse_val.update_state(tf.nn.compute_average_loss(mse, global_batch_size=self.GLOBAL_BATCH_SIZE))
            else:
                self.mse_val.update_state(tf.reduce_mean(mse))
            
    
    def sample(self, n_samples, temperature=1.0):
        s = tf.expand_dims(self.h_var, 0)
        s = tf.tile(s, [n_samples, 1, 1, 1])
        
        z0_shape = tf.concat([[n_samples], self.z0_shape], axis=0)
        mu = softclamp5(tf.zeros(z0_shape))
        sigma = tf.math.exp(softclamp5(tf.zeros(z0_shape))) + 1e-2
        
        if temperature != 1.0:
            sigma *= temperature
        z = self.sampler.sample(mu, sigma)

        decoder_index = 0
        last_s = None
        
        for layer in self.dec_layers:
            if isinstance(layer, DecoderSampleCombiner):
                if decoder_index > 0:
                    mu, log_sigma = self.sampler.get_params(self.sampler.dec_sampler, decoder_index, s)
                    mu = softclamp5(mu)
                    sigma = tf.math.exp(softclamp5(log_sigma)) + 1e-2
                    z = self.sampler.sample(mu, sigma)
                last_s = s
                s = layer(s, z)
                decoder_index += 1
            else:
                s = layer(s)
        
        reconstruction = self.model_postprocess(s)
        
        if self.dataset_option == 'mnist':
            distribution = distributions.Bernoulli(logits=reconstruction, dtype=tf.float32, allow_nan_stats=False)
            images = distribution.probs_parameter()
        else:
            images = self.sample_from_discretized_mix_logistic(reconstruction)
        
        z1 = self.sampler.sample(mu, sigma)
        z2 = self.sampler.sample(mu, sigma)
        # return images and mu, sigma, s used for sampling last hierarchical z in turn enabling sampling of images
        return images, last_s, z1, z2

    
    # As sample(), but starts from a fixed last hierarchical z given by mu, sigma and s. See sample() for details.
    def sample_with_z(self, z, s):
        last_gen_layer = self.dec_layers[-1]
        s = last_gen_layer(s, z)
        
        if self.dataset_option == 'mnist':
            reconstruction = self.postprocess(s)
            distribution = distributions.Bernoulli(logits=reconstruction, dtype=tf.float32, allow_nan_stats=False)
            images = distribution.mean()
        else:
            images = self.sample_from_discretized_mix_logistic(reconstruction)
        return images
    
    
    def calculate_kl_loss(self, z_params: List, balancing):
        # -KL(q(z1|x)||p(z1)) - sum[ KL(q(zl|x,z<l) || p(z|z<l))]
        kl_per_group = []
        
        # n_groups x batch_size x 4
        loss = 0
        for g in z_params:
            # [enc_mu, enc_sigma, dec_mu, dec_sigma]
            # term1 = (g.enc_mu - g.dec_mu) / g.dec_sigma
            # term2 = g.enc_sigma / g.dec_sigma
            
            term1 = (g[0] - g[2]) / g[3]
            term2 = g[1] / g[3]
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - tf.math.log(term2)
            kl_per_group.append(tf.math.reduce_sum(kl, axis=[1, 2, 3]))
        
        # balance kl
        if balancing:
            
            # Requires different treatment for encoder and decoder?
            kl_alphas = self.calculate_kl_alphas(self.n_latent_scales, self.n_groups_per_scale)
            
            kl_all = tf.stack(kl_per_group, 0)
            kl_coeff_i = tf.reduce_mean(tf.math.abs(kl_all), 1) + 0.01
            total_kl = tf.reduce_sum(kl_coeff_i)
            kl_coeff_i = kl_coeff_i / kl_alphas * total_kl
            kl_coeff_i = kl_coeff_i / tf.reduce_mean(kl_coeff_i, 0, keepdims=True)
            temp = tf.stack(kl_per_group, 1)
            
            # We stop gradient through kl_coeff_i because we are only interested
            # in changing the magnitude of the loss, not the direction of the
            # gradient.
            loss = tf.reduce_sum(temp * tf.stop_gradient(kl_coeff_i), axis=[1])
        else:
            loss = tf.math.reduce_sum(tf.convert_to_tensor(kl_per_group, dtype=tf.float32), axis=[0])
        return loss
    
    # Calculates the balancer coefficients alphas. The coefficient decay for later groups,
    # for which original paper offer several functions. Here, a square function is used.
    def calculate_kl_alphas(self, num_scales, groups_per_scale):
        coeffs = []
        for i in range(num_scales):
            aa = np.square(2 ** i)
            bb = groups_per_scale[num_scales - i - 1]
            cc = tf.ones([groups_per_scale[num_scales - i - 1]], tf.float32)
            coeffs.append(aa / bb  * cc)
        coeffs = tf.concat(coeffs, 0)
        coeffs /= tf.reduce_min(coeffs)
        return coeffs
    
    def calculate_recon_loss(self, inputs, reconstruction, crop_output=False):
        if self.dataset_option == 'mnist':
            if crop_output:
                inputs = inputs[:, 2:30, 2:30, :]
                reconstruction = reconstruction[:, 2:30, 2:30, :]
            log_probs = distributions.Bernoulli(logits=reconstruction, dtype=tf.float32, allow_nan_stats=False).log_prob(inputs)
            recons_loss = -tf.math.reduce_sum(log_probs, axis=[1, 2, 3])
        else:
            recons_loss = self.discretized_mix_logistic_loss(inputs, reconstruction)
        return recons_loss
    
    def calculate_bn_loss(self):
        bn_loss = 0.0
        for ind, model in enumerate(self.model_decoder.layers):
            if isinstance(model, layers.InputLayer) or isinstance(model, layers.ELU) or isinstance(model, SpectralNormalization) or isinstance(model, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
                pass
            else:
                for layer in model.layers:
                    if isinstance(layer, layers.BatchNormalization):
                        bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
                    elif hasattr(layer, "layers"):
                        for inner_layer in layer.layers:
                            if isinstance(inner_layer, layers.BatchNormalization):
                                bn_loss += tf.math.reduce_max(tf.math.abs(inner_layer.weights[0]))
        return self.sr_lambda * bn_loss
    
    def int_shape(self, x):
        return list(map(int, x.get_shape()))
    
    def log_sum_exp(self, x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis)
        m2 = tf.reduce_max(x, axis, keepdims=True)
        return m + tf.math.log(tf.reduce_sum(tf.exp(x-m2), axis))

    def log_prob_from_logits(self, x):
        """ numerically stable log_softmax implementation that prevents overflow """
        axis = len(x.get_shape()) - 1
        m = tf.reduce_max(x, axis, keepdims=True)
        return x - m - tf.math.log(tf.reduce_sum(tf.math.exp(x-m), axis, keepdims=True))
    
    def discretized_mix_logistic_loss(self, x, l, sum_all=True):
        x = 2.0 * x - 1.0
        
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        xs = self.int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
        ls = self.int_shape(l) # predicted distribution, e.g. (B,32,32,100)
        nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics 10 groups
        logit_probs = l[:, :, :, :nr_mix]
        
        l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
        means = l[:,:,:,:,:nr_mix]
        log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
        coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
        x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
        m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
        m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
        
        # reshaping final image
        means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)

        centered_x = x - means
        inv_stdv = tf.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1./255.)
        cdf_plus = tf.nn.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1./255.)
        cdf_min = tf.nn.sigmoid(min_in)
        log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
        cdf_delta = cdf_plus - cdf_min # probability for all other cases
        mid_in = inv_stdv * centered_x
        log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
        
        # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)
        
        # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.math.log(cdf_delta)))
        
        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, 
        # we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
        
        log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, 
                                                                tf.where(cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
        
        log_probs = tf.reduce_sum(log_probs,3) + self.log_prob_from_logits(logit_probs)
        
        if sum_all:
            return -tf.reduce_sum(self.log_sum_exp(log_probs)), 
        else:
            return -tf.reduce_sum(self.log_sum_exp(log_probs),[1,2])

    def sample_from_discretized_mix_logistic(self, l, nr_mix=10):
        # l is the reconstruction vector before processing
        ls = self.int_shape(l)
        xs = ls[:-1] + [3]
        
        # unpack parameters
        logit_probs = l[:, :, :, :nr_mix]
        l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])

        # sample mixture indicator from softmax
        sel = tf.one_hot(tf.argmax(logit_probs - tf.math.log(-tf.math.log(tf.random.uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), 
                         depth=nr_mix, dtype=tf.float32)
        sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])

        # select logistic parameters
        means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
        log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
        coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
        
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
        x = means + tf.exp(log_scales)*(tf.math.log(u) - tf.math.log(1. - u))
        x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
        x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
        x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
        return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)

