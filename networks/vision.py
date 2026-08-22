from dataclasses import asdict, dataclass, is_dataclass
from typing import Union, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import pytorch_lightning as pl
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
from networks.activations import Activation
from networks.layers import PatchEmbed, TransformerLayer, ml_MLPLayer, ConvNormActivation
from utils.config import CNNConfig, TransformerConfig, ViTConfig, ResNetConfig, ConvNetConfig, ml_MLPConfig, LinearConfig, SpatialSoftmaxConfig
from utils.utils import conv_out_shape, output_padding_shape
from torchgeometry.contrib import SpatialSoftArgmax2d


class Encoder(pl.LightningModule):
    """ Encoder + reparameterization

    ただのEncoderよりモデルを軽く
    self.sequenceでは
        (C, W, H) -> (batch_size, W/2 * H/2 * 256)
        -> (batch_size, middle_layer_dim)
    self.fc_for_mu/ligvarで
        (batch_size, middle_layer_dim) -> (batch_size, latent_dim)
    になる

    """

    def __init__(
        self,
        obs_shape: tuple,
        embed_obs_dim: int,
        encoder_cfg: Union[CNNConfig, ViTConfig],
    ):
        super().__init__()

        self.obs_shape = obs_shape

        if isinstance(encoder_cfg, ViTConfig):
            encoder = ViTEncoder
            self.encoder_type = "ViT"
        elif isinstance(encoder_cfg, CNNConfig):
            encoder = CNN
            self.encoder_type = "CNN"
        else:
            raise NotImplementedError(
                f"{type(encoder_cfg)} is not implemented")
        if is_dataclass(encoder_cfg):
            encoder_cfg = asdict(encoder_cfg)

        self.encoder = encoder(
            embed_obs_dim,
            obs_shape,
            **encoder_cfg
        )

        self.embed_dim = embed_obs_dim
        self.conved_size = self.encoder.conved_size
        self.conved_shape = self.encoder.conved_shape
        self.last_channel = self.encoder.last_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]

        x = x.reshape([-1, *self.obs_shape])
        x = self.encoder(x)

        x = x.reshape([*batch_shape, *x.shape[1:]])
        return x


class Decoder(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple,
        decoder_cfg: Union[CNNConfig, ViTConfig, ResNetConfig],
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        if isinstance(decoder_cfg, ViTConfig):
            decoder = ViTDecoder
        elif isinstance(decoder_cfg, CNNConfig):
            decoder = ConvTranspose
        elif isinstance(decoder_cfg, ResNetConfig):
            decoder = ObservationResNetDecoder
        else:
            raise NotImplementedError(
                f"{type(decoder_cfg)} is not implemented")
        if is_dataclass(decoder_cfg):
            decoder_cfg = asdict(decoder_cfg)

        self.decoder = decoder(
            latent_dim,
            obs_shape,
            **decoder_cfg
        )
        self.decode_edge = decoder_cfg.get("decode_edge", False)

    def make_specific_projection(self, input_dim: int) -> nn.Sequential:
        self.decoder.make_specific_projection(input_dim)

    def decode_from_specific_projection(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.decode_from_specific_projection(x)

    def freeze_fc(self):
        self.decoder.freeze_fc()

    def unfreeze_fc(self):
        self.decoder.unfreeze_fc()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_dim:
            batch_shape = x.shape[:-1]
        else:
            batch_shape = x.shape[:-2]
        x = self.decoder(x)

        return x.reshape([*batch_shape, *self.obs_shape])


class ViTEncoder(nn.Module):
    def __init__(
        self,
        embed_obs_dim: int,
        obs_shape: tuple,
        patch_size: int = 8,
        vit_cfg: TransformerConfig = None,
        **kwargs
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.embed_dim = embed_obs_dim
        self.patch_size = patch_size
        self.vit_cfg = vit_cfg

        self.vit = TransformerLayer(
            self.embed_dim, max_len=self.n_patches, **vit_cfg
        )
        self.patch_embed = PatchEmbed(
            emb_dim=self.vit.d_model,
            patch_size=self.patch_size,
            obs_shape=self.obs_shape
        )
        self.last_channel = self.vit.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        x = self.vit(x)
        return x

    def patchify(self, imgs: torch.Tensor):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape([imgs.shape[0], 3, h, p, w, p])
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape([imgs.shape[0], h * w, p**2 * 3])
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.obs_shape[1] // p
        w = self.obs_shape[2] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, 3, p, p))
        x = torch.einsum("nhwcpq->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @property
    def conved_size(self):
        return self.n_patches

    @property
    def conved_shape(self):
        return (self.obs_shape[1] // self.patch_size, self.obs_shape[2] // self.patch_size)

    @property
    def n_patches(self):
        return (self.obs_shape[1] // self.patch_size) * (self.obs_shape[2] // self.patch_size)

    @property
    def patch_dim(self):
        return self.patch_size ** 2 * self.obs_shape[0]


class ViTDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple,
        patch_size: int = 8,
        proj_activation: str = 'Mish',
        vit_cfg: TransformerConfig = None,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.patch_size = patch_size

        self.vit_cfg = vit_cfg
        self.vit = TransformerLayer(
            self.patch_dim*self.n_patches, max_len=self.n_patches, **vit_cfg
        )
        if self.latent_dim:
            self.proj_activation = Activation(proj_activation)
            self.projector = nn.Sequential(
                nn.Linear(self.latent_dim, self.vit.d_model*self.n_patches),
                self.proj_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = False
        if self.latent_dim:
            if x.ndim == 3:
                B, L = x.shape[:2]
                x = x.reshape(B*L, -1)
                reshaped = True
            x = self.projector(x)
            x = x.reshape([-1, self.n_patches, self.vit.d_model])
        else:
            if x.ndim == 4:
                B, L = x.shape[:2]
                reshaped = True
            data_shape = x.shape[-2:]
            x = x.reshape([-1, *data_shape])
            assert x.shape[1] == self.n_patches and x.shape[2] == self.vit.d_model, f"{x.shape} != {self.n_patches} and {self.vit.d_model}"
        x = self.vit(x)
        x = x.reshape([-1, self.n_patches, self.patch_dim])
        x = self.unpatchify(x)
        if reshaped:
            x = x.reshape(B, L, *x.shape[1:])
        return x

    def make_specific_projection(self, input_dim: int) -> nn.Sequential:
        self.new_projector = nn.Sequential(
            nn.Linear(input_dim, self.vit.d_model*self.n_patches),
            self.proj_activation)
    def decode_from_specific_projection(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = False
        if self.latent_dim:
            if x.ndim == 3:
                B, L = x.shape[:2]
                x = x.reshape(B*L, -1)
                reshaped = True
            x = self.new_projector(x)
            x = x.reshape([-1, self.n_patches, self.vit.d_model])
        else:
            if x.ndim == 4:
                B, L = x.shape[:2]
                reshaped = True
            data_shape = x.shape[-2:]
            x = x.reshape([-1, *data_shape])
            assert x.shape[1] == self.n_patches and x.shape[2] == self.vit.d_model, f"{x.shape} != {self.n_patches} and {self.vit.d_model}"
        x = self.vit(x)
        x = x.reshape([-1, self.n_patches, self.patch_dim])
        x = self.unpatchify(x)
        if reshaped:
            x = x.reshape(B, L, *x.shape[1:])
        return x

    def freeze_fc(self):
        for param in self.projector.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.projector.parameters():
            param.requires_grad = True

    def patchify(self, imgs: torch.Tensor):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape([imgs.shape[0], 3, h, p, w, p])
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape([imgs.shape[0], h * w, p**2 * 3])
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.obs_shape[1] // p
        w = self.obs_shape[2] // p
        assert h * \
            w == x.shape[1], f"{h*w} != {x.shape[1]}, please check the shape {x.shape} and obs_shape {self.obs_shape}"

        x = x.reshape(shape=(x.shape[0], h, w, 3, p, p))
        x = torch.einsum("nhwcpq->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @property
    def n_patches(self):
        return (self.obs_shape[1] // self.patch_size) * (self.obs_shape[2] // self.patch_size)

    @property
    def patch_dim(self):
        return self.patch_size ** 2 * self.obs_shape[0]


class CNN(nn.Module):

    def __init__(
        self,
        embed_obs_dim: int,
        obs_shape: tuple,
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        hidden_activation: str = 'Mish',
        output_activation: str = 'Tanh',
        fc_hidden: tuple = None,
        spatial: bool = False,
        batch_norm: bool = False,
        spatial_temprature: float = 0.0,
        n_res_blocks: int = 0,
        **kwargs
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.embed_dim = embed_obs_dim
        self.channels = [obs_shape[0], *channels]
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.conv_activation = Activation(hidden_activation)
        self.output_activation = Activation(output_activation)
        self.batch_norm = batch_norm
        self.n_res_blocks = n_res_blocks

        self.fc_hidden = fc_hidden
        self.spatial = spatial
        self.spatial_temp = spatial_temprature

        self.conv = self._build_conv()

        if self.embed_dim:
            self.flatten = self._build_dense()

        if self.n_res_blocks:

            # Residual blocks
            res_blocks = []
            for _ in range(n_res_blocks):
                res_blocks.append(ResidualBlock(
                    channels[-1], 
                    paddings[-1], 
                    getattr(nn, hidden_activation),
                    batch_norm, 
                    use_mask=False))
            self.res_blocks = nn.Sequential(*res_blocks)

            # Second conv layer post residual blocks
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=channels[-1], 
                             out_channels=channels[-1], 
                             kernel_size=paddings[-1]*2+1, 
                             stride=1, 
                             padding=paddings[-1]), 
                nn.BatchNorm2d(channels[-1], 0.8) if batch_norm else nn.Identity())

        self.last_channel = self.channels[-1]

    def _build_conv(self):
        convs = []
        for i in range(len(self.channels)-1):
            convs += [nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=self.kernels[i],
                                stride=self.strides[i], padding=self.paddings[i])]
            if self.batch_norm:
                convs += [nn.BatchNorm2d(self.channels[i+1])]
            convs += [self.conv_activation]

        return nn.Sequential(*convs)

    def _build_dense(self):
        if self.spatial:

            if self.spatial_temp:
                print("spatial expectation")
                return nn.Sequential(SpatialSoftExpectation2d(self.spatial_temp), nn.Flatten())
            else:
                print("spatial")
                return nn.Sequential(SpatialSoftArgmax2d(), nn.Flatten())

        elif self.fc_hidden is None:
            if self.conved_size == self.embed_dim:
                print("skip linear in CNN")
                return nn.Sequential(nn.Flatten())
            else:
                return nn.Sequential(nn.Flatten(), nn.Linear(self.channels[-1]*self.conved_size, self.embed_dim), self.output_activation)
        else:
            dense = []
            dense += [nn.Flatten()]
            dense += [nn.Linear(self.conved_size * self.channels[-1], self.fc_hidden[0])]
            dense += [self.conv_activation]
            for i in range(len(self.fc_hidden)-1):
                dense += [nn.Linear(self.fc_hidden[i], self.fc_hidden[i+1])]
                dense += [self.conv_activation]
            dense += [nn.Linear(self.fc_hidden[-1], self.embed_dim),
                      self.output_activation]
            return nn.Sequential(*dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.n_res_blocks:
            out = self.res_blocks(x)
            out2 = self.conv2(out)
            x = torch.add(x, out2)

        if self.embed_dim:
            x = self.flatten(x)
        else:
            x = x.flatten(-2)
            x = x.transpose(1, 2)
            assert x.shape[-1] == self.channels[-1], f"{x.shape} != {self.channels[-1]}"
        return x

    @property
    def conved_size(self):
        if self.spatial:
            return self.channels[-1] * 2
        else:
            conv_shape = self.obs_shape[1:]
            for i in range(len(self.channels)-1):
                conv_shape = conv_out_shape(
                    conv_shape, self.paddings[i], self.kernels[i], self.strides[i])

            conved_size = np.prod(conv_shape).item()
            return conved_size
    @property
    def conved_shape(self):
        conv_shape = self.obs_shape[1:]
        for i in range(len(self.channels)-1):
            conv_shape = conv_out_shape(
                conv_shape, self.paddings[i], self.kernels[i], self.strides[i])

        return conv_shape


class ObservationResNetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 12,
        obs_shape: list = [3, 24, 32],
        conv_channel: int = 64,
        conv_padding: int = 1,
        final_padding: int = 4,
        conv_activation: str = "Mish",
        mlp_activation: str = "ELU",
        out_activation: str = "Sigmoid",
        n_res_blocks: int = 8,
        batch_norm: bool = True,
        upscale_factor: int = 2,
        n_upsampling: int = 2,
        use_mask: bool = False,
        decode_edge: bool = False,
        init_channel: int = 1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_shape = obs_shape
        self.conv_channel = conv_channel
        self.conv_padding = conv_padding
        self.conv_activation = getattr(nn, conv_activation)
        self.mlp_activation = getattr(nn, mlp_activation)
        self.out_activation = getattr(nn, out_activation)
        self.batch_norm = batch_norm
        self.n_res_blocks = n_res_blocks
        self.upscale_factor = upscale_factor
        self.n_upsampling = n_upsampling
        self.decode_edge = decode_edge
        self.init_channel = init_channel
        if self.batch_norm:
            print("decoder batch nomalization enabled")
        else:
            print("decoder batch nomalization disabled")
        if use_mask:
            print("decoder use mask")
            self.conv_fn = MaskedConv2d
        else:
            self.conv_fn = nn.Conv2d

        _scaling_factor = upscale_factor ** n_upsampling

        height = obs_shape[1]
        width = obs_shape[2]

        out_channels = obs_shape[0] * 3 if decode_edge else obs_shape[0]
        self.input_height, self.input_width = height//_scaling_factor, width//_scaling_factor

        if latent_dim:
            self.mlp = nn.Sequential(
                nn.Linear(latent_dim, (height//_scaling_factor)*(width//_scaling_factor)), 
                self.mlp_activation(), 
                nn.Linear((height//_scaling_factor)*(width//_scaling_factor), (height//_scaling_factor)*(width//_scaling_factor) * init_channel)
                )

        # First layer
        if use_mask:
            self.conv1 = nn.Sequential(
                self.conv_fn(False, in_channels=init_channel, out_channels=conv_channel, kernel_size=final_padding*2+1, stride=1, padding=final_padding), self.conv_activation())
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(init_channel, conv_channel, kernel_size=final_padding*2+1, stride=1, padding=final_padding), self.conv_activation())

        # Residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks.append(ResidualBlock(
                conv_channel, conv_padding, self.conv_activation, batch_norm, use_mask=use_mask))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            self.conv_fn(in_channels=conv_channel, out_channels=conv_channel, kernel_size=conv_padding*2+1, stride=1, padding=conv_padding), nn.BatchNorm2d(conv_channel, 0.8) if batch_norm else nn.Identity())

        # Upsampling layers
        upsampling = []
        for _ in range(self.n_upsampling):
            upsampling += [
                self.conv_fn(in_channels=conv_channel, out_channels=conv_channel*(upscale_factor**2), kernel_size=conv_padding *
                          2+1, stride=1, padding=conv_padding),
                nn.BatchNorm2d(conv_channel*(upscale_factor**2)
                               ) if batch_norm else nn.Identity(),
                nn.PixelShuffle(upscale_factor),
                self.conv_activation(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            self.conv_fn(in_channels=conv_channel, out_channels=out_channels, kernel_size=final_padding*2+1, stride=1, padding=final_padding), self.out_activation())
        print("Decoder initialized")
    def make_specific_projection(self, input_dim: int) -> torch.Tensor:
        self.new_mlp = nn.Sequential(
            nn.Linear(input_dim, (self.input_height)*(self.input_width)), 
            self.mlp_activation(), 
            nn.Linear((self.input_height)*(self.input_width), (self.input_height)*(self.input_width) * self.init_channel)
            )

    def decode_from_specific_projection(self, x: torch.Tensor) -> D.distribution:
        reshaped = False
        if self.latent_dim:
            if x.ndim == 3:
                B, L = x.shape[:2]
                x = x.reshape(B*L, -1)
                reshaped = True
            x = self.new_mlp(x)
        else:
            if x.ndim == 4:
                B, L = x.shape[:2]
                data_shape = x.shape[-2:]
                x = x.reshape(B*L, *data_shape)
                reshaped = True
            x = x.transpose(1, 2)
            assert x.shape[-1] == self.init_channel * self.input_height * self.input_width, f"{x.shape} != {self.init_channel} and {self.input_height * self.input_width}"
        x = x.reshape((-1, self.init_channel, self.input_height, self.input_width))
        x = self.conv1(x)
        out = self.res_blocks(x)
        out2 = self.conv2(out)
        out = torch.add(x, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        if reshaped:
            out = out.reshape(B, L, out.shape[1], out.shape[2], out.shape[3])
        return out

    def freeze_fc(self):
        for param in self.mlp.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.mlp.parameters():
            param.requires_grad = True


    def forward(self, x):
        reshaped = False

        if self.latent_dim:
            if x.ndim == 3:
                B, L = x.shape[:2]
                x = x.reshape(B*L, -1)
                reshaped = True
            x = self.mlp(x)
        else:
            if x.ndim == 4:
                B, L = x.shape[:2]
                data_shape = x.shape[-2:]
                x = x.reshape(B*L, *data_shape)
                reshaped = True
            x = x.transpose(1, 2)
            assert x.shape[-1] == self.input_height * self.input_width and x.shape[-2] == self.init_channel, f"{x.shape} != {self.init_channel} and {self.input_height * self.input_width}"
        x = x.reshape((-1, self.init_channel, self.input_height, self.input_width))

        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        if reshaped:
            out = out.reshape(B, L, out.shape[1], out.shape[2], out.shape[3])
        return out

class ConvTranspose(nn.Module):
    """

    Vision Decoder

    self.sequenceで
        (batch_size, latent_dim) -> (batch_size, middle_layer_dim)
        -> (batch_size, W/2 * H/2 * 256) -> (batch_size, C, W, H))
    になる

    """

    def __init__(
        self,
        latent_dim: int,
        obs_shape: tuple,
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        hidden_activation: str = 'Mish',
        output_activation: str = 'Tanh',
        fc_hidden: tuple = None,
        batch_norm: bool = False,
        decode_edge: bool = False,
        init_channel: int = 1,
        **options
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_shape = obs_shape
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.conv_activation = Activation(hidden_activation)
        self.output_activation = Activation(output_activation)
        self.fc_hidden = fc_hidden
        self.batch_norm = batch_norm
        self.conv_shapes = []
        self.conv_outpads = []
        self.decode_edge = decode_edge
        self.init_channel = init_channel
        if self.init_channel != channels[0]:
            self.first_conv = nn.Conv2d(self.init_channel, channels[0], kernel_size=1, stride=1, padding=0)
            self.have_first_conv = True
        else:
            self.have_first_conv = False

        for i in reversed(range(len(channels))):

            self.conv_shapes += [conv_out_shape(
                obs_shape[1:] if i == len(channels)-1 else self.conv_shapes[len(channels)-2-i], paddings[i], kernels[i], strides[i])]
            self.conv_outpads += [output_padding_shape(
                obs_shape[1:] if i == len(channels)-1 else self.conv_shapes[len(channels)-2-i], self.conv_shapes[len(channels)-1-i], paddings[i], kernels[i], strides[i])]
        if self.latent_dim:
            self.linear = self._build_dense(self.latent_dim)

        self.conv = self._build_conv()
    def make_specific_projection(self, input_dim: int) -> nn.Sequential:
        self.new_linear = self._build_dense(input_dim)

    def decode_from_specific_projection(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = False
        if self.latent_dim:
            if x.ndim == 3:
                B, L = x.shape[:2]
                x = x.reshape(B*L, -1)
                reshaped = True
            x = self.new_linear(x)
        else:
            if x.ndim == 4:
                B, L = x.shape[:2]
                data_shape = x.shape[-2:]
                x = x.reshape(B*L, *data_shape)
                reshaped = True
            x = x.transpose(1, 2)
            assert x.shape[-2] == self.channels[0], f"{x.shape} != {self.channels[0]}"
        z = x.reshape([-1, self.channels[0], *self.conv_shapes[-1]])
        if self.have_first_conv:
            z = self.first_conv(z)
        reconstruct = self.conv(z)

        if reshaped:
            reconstruct = reconstruct.reshape(B, L, *reconstruct.shape[1:])
        return reconstruct
            
    def freeze_fc(self):
        for param in self.linear.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.linear.parameters():
            param.requires_grad = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reshaped = False
        if self.latent_dim:
            if z.ndim == 3:
                B, L = z.shape[:2]
                z = z.reshape(B*L, -1)
                reshaped = True
            z = self.linear(z)
        else:
            if z.ndim == 4:
                B, L = z.shape[:2]
                data_shape = z.shape[-2:]
                z = z.reshape(B*L, *data_shape)
                reshaped = True
            z = z.transpose(1, 2)
            assert z.shape[-2] == self.init_channel, f"{z.shape} != {self.channels[0]}"
        z = z.reshape([-1, self.init_channel, *self.conv_shapes[-1]])
        if self.have_first_conv:
            z = self.first_conv(z)
        reconstruct = self.conv(z)
        if reshaped:
            reconstruct = reconstruct.reshape(B, L, *reconstruct.shape[1:])

        return reconstruct

    def _build_conv(self):
        convs = []
        for i in range(len(self.channels)-1):
            convs += [nn.ConvTranspose2d(self.channels[i], self.channels[i+1], kernel_size=self.kernels[i],
                                         stride=self.strides[i], padding=self.paddings[i], output_padding=self.conv_outpads[i])]
            if self.batch_norm:
                convs += [nn.BatchNorm2d(self.channels[i])]
            convs += [self.conv_activation]
        convs += [nn.ConvTranspose2d(self.channels[-1], 
                                     self.obs_shape[0]*3 if self.decode_edge else self.obs_shape[0], 
                                     kernel_size=self.kernels[-1],
                                     stride=self.strides[-1], 
                                     padding=self.paddings[-1], 
                                     output_padding=self.conv_outpads[-1])]
        convs += [self.output_activation]

        return nn.Sequential(*convs)

    def _build_dense(self, input_dim: int):
        dense = []
        if self.fc_hidden is None:
            dense += [nn.Linear(input_dim, self.conved_size)]
            dense += [self.conv_activation]
        else:
            dense += [nn.Linear(self.latent_dim, self.fc_hidden[0])]
            dense += [self.conv_activation]
            for i in range(len(self.fc_hidden)-1):
                dense += [nn.Linear(self.fc_hidden[i], self.fc_hidden[i+1])]
                dense += [self.conv_activation]
            dense += [nn.Linear(self.fc_hidden[-1], self.conved_size)]
            dense += [self.conv_activation]
        return nn.Sequential(*dense)

    @property
    def conved_size(self):

        conved_size = self.init_channel * np.prod(self.conv_shapes[-1]).item()
        print(f"conved_size: {conved_size}")
        return conved_size


class SpatialSoftExpectation2d(nn.Module):
    def __init__(self, temperature=1.0, normalized_coordinates=True):
        super().__init__()
        self.temperature = torch.tensor(temperature).float()
        self.normalized_coordinates = normalized_coordinates

    def forward(self, x):
        x = spatial_softmax2d(x, self.temperature)
        x = spatial_expectation2d(x, self.normalized_coordinates)

        return x


class BothSpatialSoftmax(nn.Module):
    def __init__(self, temperature=1.0, normalized_coordinates=True):
        super().__init__()
        self.temperature = torch.tensor(temperature).float()
        self.normalized_coordinates = normalized_coordinates
        self.spatial = SpatialSoftArgmax2d()

    def forward(self, x):
        expectaion = spatial_softmax2d(x, self.temperature)
        expectaion = spatial_expectation2d(
            expectaion, self.normalized_coordinates)
        argmax = self.spatial(x)

        x = torch.cat([expectaion, argmax], dim=-1)

        return x

class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_self: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        h = self.weight.size()[2]
        w = self.weight.size()[3]
        self.mask.fill_(1)
        # マスクタイプによる場合分け
        if not include_self:
          self.mask[:, :, h // 2, w // 2:] = 0
          self.mask[:, :, h // 2 + 1:] = 0
        else: # 自分自身は見る
          self.mask[:, :, h // 2, w // 2 + 1:] = 0
          self.mask[:, :, h // 2 + 1:] = 0
 
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, conv_padding: int = 1, conv_activation: nn.Module = nn.Mish(), batch_norm: bool = True, use_mask: bool = False):
        super(ResidualBlock, self).__init__()
        if use_mask:
            self.conv_block = nn.Sequential(
                MaskedConv2d(True, in_features, in_features,
                             kernel_size=conv_padding*2+1, stride=1, padding=conv_padding),
                nn.BatchNorm2d(in_features, 0.8) if batch_norm else nn.Identity(),
                conv_activation(),
                MaskedConv2d(True, in_features, in_features,
                             kernel_size=conv_padding*2+1, stride=1, padding=conv_padding),
                nn.BatchNorm2d(in_features, 0.8) if batch_norm else nn.Identity(),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_features, in_features,
                          kernel_size=conv_padding*2+1, stride=1, padding=conv_padding),
                nn.BatchNorm2d(in_features, 0.8) if batch_norm else nn.Identity(),
                conv_activation(),
                nn.Conv2d(in_features, in_features,
                          kernel_size=conv_padding*2+1, stride=1, padding=conv_padding),
                nn.BatchNorm2d(in_features, 0.8) if batch_norm else nn.Identity(),
            )

    def forward(self, x):
        return x + self.conv_block(x)



class ml_Encoder(pl.LightningModule):
    """
    Encoder with various architectures.

    Parameters
    ----------
    feature_dim: Union[int, tuple[int, int, int]]
        Dimension of the feature tensor.
        If int, Encoder includes full connection layer to downsample the feature tensor.
        Otherwise, Encoder does not include full connection layer and directly process with backbone network.
    obs_shape: tuple[int, int, int]
        shape of the input tensor
    backbone_cfg: Union[ViTConfig, ConvNetConfig, ResNetConfig]
        configuration of the network
    fc_cfg: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]
        configuration of the full connection layer. If feature_dim is tuple, fc_cfg is ignored.
        If feature_dim is int, fc_cfg must be provided. Default is None.


    Examples
    --------
    >>> feature_dim = 128
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[16, 32, 64],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> fc_cfg = LinearConfig(
    ...     activation="ReLU",
    ...     bias=True
    ... )
    >>> encoder = Encoder(feature_dim, obs_shape, cfg, fc_cfg)
    >>> x = torch.randn(2, *obs_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 128])

    >>> encoder
    Encoder(
      (encoder): ConvNet(
        (conv): Sequential(
          (0): ConvNormActivation(
            (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (1): ConvNormActivation(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (2): ConvNormActivation(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
        )
      )
      (fc): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): LinearNormActivation(
          (linear): Linear(in_features=4096, out_features=128, bias=True)
          (norm): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
      )
    )

    """

    def __init__(
        self,
        feature_dim: Union[int,Tuple[int, int, int]],
        obs_shape: Tuple[int, int, int],
        backbone_cfg: Union[ViTConfig , ConvNetConfig , ResNetConfig],
        fc_cfg: Union[ml_MLPConfig , LinearConfig , SpatialSoftmaxConfig , None] = None,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape

        if isinstance(backbone_cfg, ViTConfig):
            self.encoder = ViT(obs_shape, backbone_cfg)
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.encoder = ConvNet(obs_shape, backbone_cfg)
        # elif isinstance(backbone_cfg, ResNetConfig):
        #     self.encoder = ResNetPixUnshuffle(obs_shape, backbone_cfg)
        else:
            msg = f"{type(backbone_cfg)} is not implemented"
            raise NotImplementedError(msg)

        self.feature_dim = feature_dim
        self.conved_size = self.encoder.conved_size
        self.conved_shape = self.encoder.conved_shape
        self.last_channel = self.encoder.last_channel

        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is provided"
        else:
            assert feature_dim == (self.encoder.last_channel, *self.encoder.conved_shape), (
                f"{feature_dim} != {(self.encoder.last_channel, *self.encoder.conved_shape)}"
            )
        if isinstance(fc_cfg, ml_MLPConfig):
            self.fc = nn.Sequential(
                nn.Flatten(),
                ml_MLPLayer(self.conved_size, feature_dim, fc_cfg),
            )
        # elif isinstance(fc_cfg, LinearConfig):
        #     self.fc = nn.Sequential(
        #         nn.Flatten(),
        #         LinearNormActivation(self.conved_size, feature_dim, fc_cfg),
        #     )
        # elif isinstance(fc_cfg, AdaptiveAveragePoolingConfig):
        #     self.fc = nn.Sequential(
        #         nn.AdaptiveAvgPool2d(fc_cfg.output_size),
        #         nn.Flatten(),
        #         LinearNormActivation(
        #             int(self.last_channel * np.prod(fc_cfg.output_size)),
        #             feature_dim,
        #             fc_cfg.additional_layer
        #         ) if isinstance(
        #                 fc_cfg.additional_layer, LinearConfig
        #         ) else MLPLayer(
        #             int(self.last_channel * np.prod(fc_cfg.output_size)),
        #             feature_dim,
        #             fc_cfg.additional_layer
        #         ) if isinstance(
        #                 fc_cfg.additional_layer, MLPConfig
        #         ) else nn.Identity(),
        #     )
        #     if fc_cfg.additional_layer is None:
        #         self.feature_dim = self.last_channel * (fc_cfg.output_size**2) if isinstance(
        #             fc_cfg.output_size, int
        #         ) else self.last_channel * np.prod(fc_cfg.output_size)
        # elif isinstance(fc_cfg, SpatialSoftmaxConfig):
        #     self.fc = nn.Sequential(
        #         SpatialSoftmax(fc_cfg),
        #         nn.Flatten(),
        #         LinearNormActivation(
        #             self.last_channel * 2,
        #             self.feature_dim,
        #             fc_cfg.additional_layer
        #         ) if isinstance(
        #                 fc_cfg.additional_layer, LinearConfig
        #         ) else MLPLayer(
        #             self.last_channel * 2,
        #             self.feature_dim,
        #             fc_cfg.additional_layer
        #         ) if isinstance(
        #                 fc_cfg.additional_layer, MLPConfig
        #         ) else nn.Identity(),
        #     )
        #     if fc_cfg.additional_layer is None:
        #         self.feature_dim = self.last_channel * 2
                
        else:
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *obs_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *feature_dim)
        """
        batch_shape = x.shape[:-3]

        x = x.reshape([-1, *self.obs_shape])
        x = self.encoder(x)
        x = x.view(-1, self.last_channel, *self.conved_shape)
        x = self.fc(x)
        return x.reshape([*batch_shape, *x.shape[1:]])
    
    
class ConvNet(nn.Module):
    """
    Convolutional Neural Network for Encoder.

    Parameters
    ----------
    obs_shape: tuple[int, int, int]
        shape of input tensor
    cfg: ConvNetConfig
        configuration of the network

    Examples
    --------
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[16, 32, 64],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> encoder = ConvNet(obs_shape, cfg)
    >>> encoder
    ConvNet(
      (conv): Sequential(
        (0): ConvNormActivation(
          (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (1): ConvNormActivation(
          (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (2): ConvNormActivation(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
      )
    )
    >>> x = torch.randn(2, *obs_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 64, 8, 8])
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        cfg: ConvNetConfig,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.channels = [obs_shape[0], *cfg.channels]
        self.cfg = cfg

        self.conv = self._build_conv()

        self.last_channel = self.channels[-1]

    def _build_conv(self) -> nn.Module:
        convs = []
        for i in range(len(self.channels) - 1):
            convs += [ConvNormActivation(self.channels[i], self.channels[i + 1], self.cfg.conv_cfgs[i])]

        return nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *obs_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, self.last_channel, *self.conved_shape)

        """
        return self.conv(x)

    @property
    def conved_shape(self) -> Tuple[int, int]:
        """
        Get the shape of the output tensor after convolutional layers.

        Returns
        -------
        tuple[int, int]
            shape of the output tensor

        Examples
        --------
        >>> obs_shape = (3, 64, 64)
        >>> cfg = ConvNetConfig(
        ...     channels=[64, 32, 16],
        ...     conv_cfgs=[
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...     ]
        ... )
        >>> encoder = ConvNet(obs_shape, cfg)
        >>> encoder.conved_shape
        (8, 8)

        """
        conv_shape = self.obs_shape[1:]
        for i in range(len(self.channels) - 1):
            padding, kernel, stride, dilation = (
                self.cfg.conv_cfgs[i].padding,
                self.cfg.conv_cfgs[i].kernel_size,
                self.cfg.conv_cfgs[i].stride,
                self.cfg.conv_cfgs[i].dilation,
            )
            conv_shape = conv_out_shape(conv_shape, padding, kernel, stride, dilation)

        return conv_shape

    @property
    def conved_size(self) -> int:
        """
        Get the size of the output tensor after convolutional layers.

        Returns
        -------
        int
            size of the output tensor

        """
        return self.last_channel * np.prod(self.conved_shape).item()

