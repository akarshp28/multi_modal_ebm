from common import DistributionParams, Rescaler
from typing import List
from postprocess import Postprocess
from util import tile_images, softclamp5

from decoder import Decoder, DecoderSampleCombiner
from encoder import Encoder
import tensorflow as tf
from tensorflow.keras import layers
from preprocess import Preprocess
from tensorflow_probability import distributions
import numpy as np


class NVAE(tf.keras.Model):
    def __init__(
        self,
        n_encoder_channels,
        n_decoder_channels,
        res_cells_per_group,
        n_preprocess_blocks,
        n_preprocess_cells,
        n_latent_per_group,
        n_latent_scales,
        n_groups_per_scale,
        n_postprocess_blocks,
        n_post_process_cells,
        sr_lambda,
        scale_factor,
        total_epochs,
        n_total_iterations,
        step_based_warmup,
        input_shape,
        **kwargs):
        
        super().__init__(**kwargs)
        self.sr_lambda = sr_lambda
        self.preprocess = Preprocess(
            n_encoder_channels,
            n_preprocess_blocks,
            n_preprocess_cells,
            scale_factor,
            input_shape)
        
        self.n_latent_per_group = n_latent_per_group
        self.n_latent_scales = n_latent_scales
        self.n_groups_per_scale = n_groups_per_scale
        self.n_total_iterations = n_total_iterations
        self.n_preprocess_blocks = n_preprocess_blocks

        mult = self.preprocess.mult
        self.encoder = Encoder(
            n_encoder_channels=n_encoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=n_groups_per_scale,
            mult=mult,
            scale_factor=scale_factor,
            input_shape=self.preprocess.output_shape_)
        
        mult = self.encoder.mult
        self.decoder = Decoder(
            n_decoder_channels=n_decoder_channels,
            n_latent_per_group=n_latent_per_group,
            res_cells_per_group=res_cells_per_group,
            n_latent_scales=n_latent_scales,
            n_groups_per_scale=list(reversed(n_groups_per_scale)),
            mult=mult,
            scale_factor=scale_factor,
            input_shape=self.encoder.output_shape_)
        
        mult = self.decoder.mult
        self.postprocess = Postprocess(
            n_postprocess_blocks,
            n_post_process_cells,
            scale_factor=scale_factor,
            mult=mult,
            n_channels_decoder=n_decoder_channels)
        
        # Updated at start of each epoch
        self.epoch = 0
        self.total_epochs = total_epochs
        self.step_based_warmup = step_based_warmup
        # Updated for each gradient pass, training step
        self.steps = 0

    def call(self, inputs, nll=False):
        x = self.preprocess(inputs)
        enc_dec_combiners, final_x = self.encoder(x)
        
        # Flip bottom-up to top-down
        enc_dec_combiners.reverse()
        
        reconstruction, z_params, log_p, log_q = self.decoder(final_x, enc_dec_combiners, nll=nll)
        reconstruction = self.postprocess(reconstruction)
        return reconstruction, z_params, log_p, log_q

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, z_params, *_ = self(data)
            recon_loss = self.calculate_recon_loss(data, reconstruction)
            bn_loss = self.calculate_bn_loss()
            
            # warming up KL term for first 30% of training
            warmup_metric = self.steps if self.step_based_warmup else self.epoch
            beta = min(warmup_metric / (0.3 * self.n_total_iterations), 1)
            activate_balancing = beta < 1
            
            kl_loss = beta * self.calculate_kl_loss(z_params, activate_balancing)
            loss = tf.math.reduce_mean(recon_loss + kl_loss)
            total_loss = loss + bn_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.steps += 1
        return {"loss": total_loss, "reconstruction_loss": recon_loss, "kl_loss": kl_loss, "bn_loss": bn_loss}
    
    def sample(self, n_samples=16, temperature=1.0, greyscale=True):
        s = tf.expand_dims(self.decoder.h, 0)
        s = tf.tile(s, [n_samples, 1, 1, 1])
        z0_shape = tf.concat([[n_samples], self.decoder.z0_shape], axis=0)
        mu = softclamp5(tf.zeros(z0_shape))
        sigma = tf.math.exp(softclamp5(tf.zeros(z0_shape))) + 1e-2
        if temperature != 1.0:
            sigma *= temperature
        z = self.decoder.sampler.sample(mu, sigma)

        decoder_index = 0
        last_s = None
        # s should have shape 16,4,4,32
        # z should have shape 8,4,4,20
        for layer in self.decoder.groups:
            if isinstance(layer, DecoderSampleCombiner):
                if decoder_index > 0:
                    mu, log_sigma = self.decoder.sampler.get_params(self.decoder.sampler.dec_sampler, decoder_index, s)
                    mu = softclamp5(mu)
                    sigma = tf.math.exp(softclamp5(log_sigma)) + 1e-2
                    z = self.decoder.sampler.sample(mu, sigma)
                last_s = s
                s = layer(s, z)
                decoder_index += 1
            else:
                s = layer(s)

        reconstruction = self.postprocess(s)
        images = self.sample_from_discretized_mix_logistic(reconstruction)
        
        z1 = self.decoder.sampler.sample(mu, sigma)
        z2 = self.decoder.sampler.sample(mu, sigma)
        # return images and mu, sigma, s used for sampling last hierarchical z in turn enabling sampling of images
        return images, last_s, z1, z2

    # As sample(), but starts from a fixed last hierarchical z given by mu, sigma and s. See sample() for details.
    def sample_with_z(self, z, s):
        last_gen_layer = self.decoder.groups[-1]
        s = last_gen_layer(s, z)
        reconstruction = self.postprocess(s)
        images = self.sample_from_discretized_mix_logistic(reconstruction)
        return images

    def calculate_kl_loss(self, z_params: List[DistributionParams], balancing):
        # -KL(q(z1|x)||p(z1)) - sum[ KL(q(zl|x,z<l) || p(z|z<l))]
        kl_per_group = []
        # n_groups x batch_size x 4
        loss = 0

        for g in z_params:
            term1 = (g.enc_mu - g.dec_mu) / g.dec_sigma
            term2 = g.enc_sigma / g.dec_sigma
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
            coeffs.append(np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * tf.ones([groups_per_scale[num_scales - i - 1]], tf.float32,))
        coeffs = tf.concat(coeffs, 0)
        coeffs /= tf.reduce_min(coeffs)
        return coeffs

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def calculate_recon_loss(self, inputs, reconstruction):
        recons_loss = self.discretized_mix_logistic_loss(inputs, reconstruction)
        return recons_loss

    def calculate_bn_loss(self):
        bn_loss = 0

        def update_loss(layer):
            nonlocal bn_loss
            if isinstance(layer, layers.BatchNormalization):
                bn_loss += tf.math.reduce_max(tf.math.abs(layer.weights[0]))
            elif hasattr(layer, "layers"):
                for inner_layer in layer.layers:
                    update_loss(inner_layer)

        for model in [self.encoder, self.decoder]:
            for layer in model.groups:
                update_loss(layer)

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
        # convert samples to be in [-1, 1]
        # l = 2.0 * l - 1.0
        
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
        #we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
        
        log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, 
                                                                tf.where(cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
        
        log_probs = tf.reduce_sum(log_probs,3) + self.log_prob_from_logits(logit_probs)
        
        if sum_all:
            return -tf.reduce_sum(self.log_sum_exp(log_probs))
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