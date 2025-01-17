from common import RescaleType, Rescaler, SqueezeExcitation
from typing import List
from tensorflow.keras import layers, Sequential, activations
from functools import partial
import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization


class Encoder(tf.keras.Model):
    def __init__(self,
        n_encoder_channels,
        n_latent_per_group: int,
        res_cells_per_group,
        n_latent_scales: int,
        n_groups_per_scale: List[int],
        mult: int,
        scale_factor: int,
        input_shape,
        **kwargs):
        super().__init__(**kwargs)
        
        # Initialize encoder tower
        self.groups = []
        for scale in range(n_latent_scales):
            n_groups = n_groups_per_scale[scale]
            
            for group_idx in range(n_groups):
                output_channels = n_encoder_channels * mult
                group = Sequential()
                
                for _ in range(res_cells_per_group):
                    group.add(EncodingResidualCell(output_channels))
                    
                self.groups.append(group)
                if not (scale == n_latent_scales - 1 and group_idx == n_groups - 1):
                    
                    # We apply a convolutional between each group except the final output
                    self.groups.append(EncoderDecoderCombiner(output_channels))
                    
            # We downsample in the end of each scale except last
            if scale < n_latent_scales - 1:
                output_channels = n_encoder_channels * mult * scale_factor
                self.groups.append(Rescaler(output_channels, scale_factor=scale_factor, rescale_type=RescaleType.DOWN))
                
                mult *= scale_factor
                input_shape *= [1, 1 / scale_factor, 1 / scale_factor, scale_factor]
                
        self.final_enc = Sequential([layers.ELU(),
                                     SpectralNormalization(layers.Conv2D(n_encoder_channels * mult, (1, 1), padding="same")),
                                     layers.ELU() ])
        self.mult = mult
        self.output_shape_ = input_shape
    
    def call(self, x):
        # 8x26x26x32
        enc_dec_combiners = []
        for group in self.groups:
            if isinstance(group, EncoderDecoderCombiner):
                # We are stepping between groups, need to save results
                enc_dec_combiners.append(partial(group, x))
            else:
                x = group(x)
        final = self.final_enc(x)
        return enc_dec_combiners, final


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
    