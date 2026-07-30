import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.activations import Activation
from utils.config import MLPConfig, States2Map1dConfig, SoftmaxTransConfig, Map2StatesConfig, ConvConfig, ml_MLPConfig, LinearConfig
from typing import Tuple, Literal, Any,Optional
from einops import rearrange
from copy import deepcopy

class MLPLayer(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 cfg: MLPConfig
                 ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.hidden_activation = Activation(cfg.hidden_activation)
        self.output_activation = Activation(cfg.output_activation)

        self.dense = self._build_dense() if cfg.n_layers > 0 else nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim * 2 if cfg.output_activation == "GLU" else self.output_dim),
            self.output_activation
        )

    def forward(self, *x):
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-1)
        return self.dense(x)

    def _build_dense(self):
        layers = []
        layers += [nn.Linear(self.input_dim, self.hidden_dim * 2 if self.cfg.hidden_activation == "GLU" else self.hidden_dim)]
        if self.cfg.layer_norm:
            layers += [nn.LayerNorm(self.hidden_dim *2 if self.cfg.hidden_activation == "GLU" else self.hidden_dim)]
        layers += [self.hidden_activation]
        if self.cfg.dropout > 0:
            layers += [nn.Dropout(self.cfg.dropout)]
        for i in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden_dim,
                                 self.hidden_dim * 2 if self.cfg.hidden_activation == "GLU" else self.hidden_dim)]
            if self.cfg.layer_norm:
                layers += [nn.LayerNorm(self.hidden_dim * 2 if self.cfg.hidden_activation == "GLU" else self.hidden_dim)]
            layers += [self.hidden_activation]
            if self.cfg.dropout > 0:
                layers += [nn.Dropout(self.cfg.dropout)]
        layers += [nn.Linear(self.hidden_dim, self.output_dim * 2 if self.cfg.output_activation == "GLU" else self.output_dim),
                   self.output_activation]
        return nn.Sequential(*layers)
    

class LinearNormActivation(nn.Module):
    """
    Linear layer with normalization and activation, and dropouts.

    References
    ----------
    LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    RMSNorm: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : LinearConfig
        Linear layer configuration.

    Examples
    --------
    >>> cfg = LinearConfig(
    ...     activation="ReLU",
    ...     norm="layer",
    ...     norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...     dropout=0.1,
    ...     norm_first=False,
    ...     bias=True
    ... )
    >>> linear = LinearNormActivation(32, 16, cfg)
    >>> linear
    LinearNormActivation(
      (linear): Linear(in_features=32, out_features=16, bias=True)
      (norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
      (activation): Activation(
        (activation): ReLU()
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
    >>> x = torch.randn(1, 32)
    >>> output = linear(x)
    >>> output.shape
    torch.Size([1, 16])

    >>> cfg = LinearConfig(
    ...     activation="SiGLU",
    ...     norm="none",
    ...     norm_cfg={},
    ...     dropout=0.0,
    ...     norm_first=True,
    ...     bias=True
    ... )
    >>> linear = LinearNormActivation(32, 16, cfg)
    >>> # If actication includes "glu", linear output_dim is doubled to adjust actual output_dim.
    >>> linear
    LinearNormActivation(
      (linear): Linear(in_features=32, out_features=32, bias=True)
      (norm): Identity()
      (activation): Activation(
        (activation): SiGLU()
      )
      (dropout): Identity()
    )
    >>> x = torch.randn(1, 32)
    >>> output = linear(x)
    >>> output.shape
    torch.Size([1, 16])


    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: LinearConfig,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            output_dim * 2 if "glu" in cfg.activation.lower() else output_dim,
            bias=cfg.bias,
        )
        if cfg.norm_first:
            normalized_shape = input_dim 
        else:
            normalized_shape = output_dim * 2 if "glu" in cfg.activation.lower() else output_dim

        cfg.norm_cfg["normalized_shape"] = normalized_shape
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation)
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(cfg.dropout)
        else:
            self.dropout = nn.Identity()
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (*, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, output_dim)
        """
        if self.norm_first:
            x = self.norm(x)
            x = self.linear(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.linear(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x
    

class ml_MLPLayer(pl.LightningModule):
    """
    Multi-layer perceptron layer.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : MLPConfig

    Examples
    --------
    >>> cfg = MLPConfig(
    ...     hidden_dim=16,
    ...     n_layers=3,
    ...     output_activation="ReLU",
    ...     linear_cfg=LinearConfig(
    ...         activation="ReLU",
    ...         norm="layer",
    ...         norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...         dropout=0.1,
    ...         norm_first=False,
    ...         bias=True
    ...     )
    ... )
    >>> mlp = MLPLayer(32, 16, cfg)
    >>> x = torch.randn(1, 32)
    >>> output = mlp(x)
    >>> output.shape
    torch.Size([1, 16])

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.dense = self._build_dense()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (*, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, output_dim)

        """
        return self.dense(x)

    def _build_dense(self) -> nn.Module:
        """
        Build dense layers.

        Returns
        -------
        nn.Sequential
            Dense layers

        Examples
        --------
        >>> cfg = MLPConfig(
        ...     hidden_dim=64,
        ...     n_layers=2,
        ...     output_activation="ReLU",
        ...     linear_cfg=LinearConfig(
        ...         activation="ReLU",
        ...         norm="layer",
        ...         norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
        ...         dropout=0.1,
        ...         norm_first=False,
        ...         bias=True
        ...     )
        ... )
        >>> mlp = MLPLayer(32, 16, cfg)
        >>> mlp._build_dense()
        Sequential(
          (0): LinearNormActivation(
            (linear): Linear(in_features=32, out_features=64, bias=True)
            (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): LinearNormActivation(
            (linear): Linear(in_features=64, out_features=64, bias=True)
            (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): LinearNormActivation(
            (linear): Linear(in_features=64, out_features=16, bias=True)
            (norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        """
        layers = []
        layers += [LinearNormActivation(self.input_dim, self.hidden_dim, self.cfg.linear_cfg)]
        for _ in range(self.n_layers - 1):
            layers += [LinearNormActivation(self.hidden_dim, self.hidden_dim, self.cfg.linear_cfg)]
        last_cfg = self.cfg.linear_cfg
        last_cfg.activation = self.cfg.output_activation
        layers += [LinearNormActivation(self.hidden_dim, self.output_dim, last_cfg)]
        return nn.Sequential(*layers)


class States2Map1d(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 feature_size: int,
                 cfg: States2Map1dConfig
                 ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_size = feature_size
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.hidden_activation = Activation(cfg.hidden_activation)
        self.output_activation = Activation(cfg.output_activation)
        if cfg.conv_target == "state":
            self.output_shape = (self.output_dim, self.feature_size)
        elif cfg.conv_target == "map":
            self.output_shape = (self.feature_size, self.output_dim)
        else:
            raise ValueError("conv_target should be either 'state' or 'map'")

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, cfg.init_channel*self.output_shape[1]),
            self.hidden_activation
        )
        self.conv = self._build_conv(
                self.output_shape[0]
                )
        if cfg.transformer_cfg is not None:
            assert cfg.transformer_cfg.d_model == self.output_shape[1], "d_model should be the same as the hidden_dim"
            self.transformer = TransformerLayer(
                output_dim=0,
                max_len=self.output_shape[0],
                d_model=cfg.transformer_cfg.d_model,
                nhead=cfg.transformer_cfg.nhead,
                hidden_dim=cfg.transformer_cfg.hidden_dim,
                n_layers=cfg.transformer_cfg.n_layers,
                dropout=cfg.transformer_cfg.dropout,
                hidden_activation=cfg.transformer_cfg.hidden_activation,
                output_activation=cfg.transformer_cfg.output_activation
            )

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        x = self.projection(x.reshape(-1, self.input_dim))
        x = x.reshape(-1, self.cfg.init_channel, self.output_shape[1])
        x = self.conv(x)
        x = x.reshape(*batch_shape, *self.output_shape)
        if self.cfg.conv_target == "map":
            x = x.transpose(-1, -2)
        return x

    def _build_conv(self, out_channels: int = 1):
        layers = []
        if self.n_layers > 0:
            layers += [nn.Conv1d(self.cfg.init_channel, self.hidden_dim, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            if self.cfg.layer_norm:
                layers += [nn.LayerNorm(self.output_shape[1])]
            layers += [self.hidden_activation]
            for _ in range(self.n_layers - 1):
                layers += [nn.Conv1d(self.hidden_dim, self.hidden_dim, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
                if self.cfg.layer_norm:
                    layers += [nn.LayerNorm(self.output_shape[1])]
                layers += [self.hidden_activation]
            layers += [nn.Conv1d(self.hidden_dim, out_channels, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            layers += [self.output_activation]
        else:
            layers += [nn.Conv1d(self.cfg.init_channel, out_channels, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            layers += [self.output_activation]
        return nn.Sequential(*layers)

class States2Map2d(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 conv_shape: Tuple[int, int],
                 cfg: States2Map1dConfig
                 ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_shape = conv_shape
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.hidden_activation = Activation(cfg.hidden_activation)
        self.output_activation = Activation(cfg.output_activation)

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, cfg.init_channel*np.prod(conv_shape)),
            self.hidden_activation
        )
        self.conv = self._build_conv(
                cfg.transformer_cfg.d_model if cfg.transformer_cfg is not None else self.output_dim
                )
        if cfg.transformer_cfg is not None:
            self.transformer = TransformerLayer(
                output_dim=self.output_dim if self.output_dim != cfg.transformer_cfg.d_model else 0,
                max_len=np.prod(conv_shape),
                d_model=cfg.transformer_cfg.d_model,
                nhead=cfg.transformer_cfg.nhead,
                hidden_dim=cfg.transformer_cfg.hidden_dim,
                n_layers=cfg.transformer_cfg.n_layers,
                dropout=cfg.transformer_cfg.dropout,
                hidden_activation=cfg.transformer_cfg.hidden_activation,
                output_activation=cfg.transformer_cfg.output_activation,
                flatten=False
            )

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        x = self.projection(x.reshape(-1, self.input_dim))
        x = x.reshape(-1, self.cfg.init_channel, *self.output_shape)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.cfg.transformer_cfg is not None:
            x = self.transformer(x)
        x = x.reshape(*batch_shape, *x.shape[1:])
        return x

    def _build_conv(self, out_channels: int = 1):
        layers = []
        if self.n_layers > 0:
            layers += [nn.Conv2d(self.cfg.init_channel, self.hidden_dim, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            if self.cfg.layer_norm:
                layers += [nn.LayerNorm([self.hidden_dim, *self.output_shape])]
            layers += [self.hidden_activation]
            for _ in range(self.n_layers - 1):
                layers += [nn.Conv2d(self.hidden_dim, self.hidden_dim, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
                if self.cfg.layer_norm:
                    layers += [nn.LayerNorm([self.hidden_dim, *self.output_shape])]
                layers += [self.hidden_activation]
            layers += [nn.Conv2d(self.hidden_dim, out_channels, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            layers += [self.output_activation]
        elif self.n_layers == 0:
            layers += [nn.Conv2d(self.cfg.init_channel, self.hidden_dim, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            layers += [self.output_activation]
        else:
            layers += [nn.Conv2d(self.cfg.init_channel, out_channels, self.cfg.kernel_size, padding=(self.cfg.kernel_size-1)//2)]
            layers += [self.output_activation]
        return nn.Sequential(*layers)

class Map2States(pl.LightningModule):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        conv_shape: Tuple[int, int],
        cfg: Map2StatesConfig
        ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.conv_shape = conv_shape
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.hidden_activation = Activation(cfg.hidden_activation)
        self.output_activation = Activation(cfg.output_activation)
        self.is_1d = cfg.is_1d
        self.kernel_size = cfg.kernel_size
        self.stride = cfg.stride
        self.padding = cfg.padding

        self.d_out = output_dim
        self.conv = self._build_conv(
                output_dim

                )
    def forward(self, x: torch.Tensor):
        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        if self.is_1d:
            x = x.transpose(-1, -2)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.conv_shape[0], w=self.conv_shape[1])
        x = self.conv(x)
        return x.flatten(1).reshape(*batch_shape, -1)

    @property
    def output_shape(self):
        if self.is_1d:
            conv_size = np.prod(self.conv_shape)
            for _ in range(np.clip(self.n_layers, 1, None) + 1):
                conv_size = (conv_size - (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
            return (conv_size, self.d_out)
        else:
            conv_shape = np.array(self.conv_shape)
            for _ in range(np.clip(self.n_layers, 1, None) + 1):
                conv_shape = (conv_shape - (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
            return (conv_shape[0], conv_shape[1], self.d_out)
    
    @property
    def output_dim(self):
        return int(np.prod(self.output_shape))

    def _build_conv(self, out_channels: int = 1):
        if self.is_1d:
            layers = []
            if self.n_layers > 0:
                layers += [nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                if self.cfg.layer_norm:
                    layers += [nn.LayerNorm(self.hidden_dim)]
                layers += [self.hidden_activation]
                for _ in range(self.n_layers - 1):
                    layers += [nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                    if self.cfg.layer_norm:
                        layers += [nn.LayerNorm(self.hidden_dim)]
                    layers += [self.hidden_activation]
                layers += [nn.Conv1d(self.hidden_dim, out_channels, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
            elif self.n_layers == 0:
                layers += [nn.Conv1d(self.input_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
            else:
                layers += [nn.Conv1d(self.input_dim, out_channels, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
        else:
            layers = []
            if self.n_layers > 0:
                layers += [nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                if self.cfg.layer_norm:
                    layers += [nn.LayerNorm([self.hidden_dim, *self.conv_shape])]
                layers += [self.hidden_activation]
                for _ in range(self.n_layers - 1):
                    layers += [nn.Conv2d(self.hidden_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                    if self.cfg.layer_norm:
                        layers += [nn.LayerNorm([self.hidden_dim, *self.conv_shape])]
                    layers += [self.hidden_activation]
                layers += [nn.Conv2d(self.hidden_dim, out_channels, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
            elif self.n_layers == 0:
                layers += [nn.Conv2d(self.input_dim, self.hidden_dim, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
            else:
                layers += [nn.Conv2d(self.input_dim, out_channels, self.kernel_size, self.stride, self.padding)]
                layers += [self.output_activation]
        return nn.Sequential(*layers)
    



class TransformerLayer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        max_len: int,
        d_model: int,
        nhead: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        hidden_activation: str = "ReLU",
        output_activation: str = "GeLU",
        flatten: bool = True,
        cls_token: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.flatten = flatten

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout, max_len=max_len)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.hidden_dim, 
            activation=hidden_activation.lower(), 
            dropout=dropout, 
            batch_first=True
            )

        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=self.n_layers, enable_nested_tensor=True)
        if self.output_dim:
            self.linear = nn.Linear(
                self.d_model*max_len if self.flatten else self.d_model, 
                output_dim)
            self.output_activation = Activation(output_activation)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        if self.output_dim: 
            if self.flatten:
                x = x.flatten(1)
            x = self.linear(x)
            x = self.output_activation(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbed(nn.Module):
    def __init__(
        self, emb_dim: int = 384, patch_size: int = 2, obs_shape: list = [3, 24, 32]
    ):
        """
        引数:
            in_channels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        """
        super(PatchEmbed, self).__init__()
        self.emb_dim = emb_dim
        # パッチの数

        self.obs_shape = obs_shape

        # パッチの大きさ
        self.patch_size = patch_size
        self.patch_num = (obs_shape[1] // patch_size) * \
            (obs_shape[2] // patch_size)
        assert (
            self.patch_size * self.patch_size * self.patch_num
            == self.obs_shape[1] * self.obs_shape[2]
        ), "patch_num is not correct"

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.obs_shape[0],
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            z_0: ViTへの入力。形状は、(B, N, D)。
                B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
        """
        # パッチの埋め込み & flatten [式(3)]
        # パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        # ここで、Pはパッチ1辺の大きさ
        x = self.patch_emb_layer(x)

        # パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
        # ここで、Npはパッチの数(=H*W/Pˆ2)
        x = x.flatten(2)

        # 軸の入れ替え (B, D, Np) -> (B, Np, D)
        x = x.transpose(1, 2)

        return x

class LinearActivation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str):
        super(LinearActivation, self).__init__()
        self.activation = Activation(activation)
        self.linear = nn.Linear(
            input_dim, 
            output_dim *2 if isinstance(self.activation, nn.GLU) else output_dim
            )

    def forward(self, x):
        return self.activation(self.linear(x))

class SoftmaxTransformation:
    def __init__(
        self, 
        cfg: SoftmaxTransConfig
        ):
        super(SoftmaxTransformation, self).__init__()
        self.vector = cfg.vector
        self.sigma = cfg.sigma
        self.n_ignore = cfg.n_ignore
        self.max = cfg.max
        self.min = cfg.min
        self.k = torch.linspace(self.min, self.max, self.vector)


    def __call__(self, x: torch.Tensor):
        return self.transform(x)

    def get_transformed_dim(self, dim: int):
        return (dim - self.n_ignore) * self.vector + self.n_ignore

    def transform(self, x: torch.Tensor):
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, :-self.n_ignore], x[:, -self.n_ignore:]
        else:
            data = x

        negative = torch.stack(
            [torch.exp((-(data+self.k[v])**2)/self.sigma) for v in range(self.vector)])
        negative_sum = negative.sum(dim=0)
        
        transformed = negative/(negative_sum+1e-8)
        transformed = rearrange(transformed, 'v b d -> b (d v)')

        if self.n_ignore:
            transformed = torch.cat([transformed, ignored], dim=-1)
        else:
            transformed = transformed
        return transformed.reshape(*batch, self.get_transformed_dim(dim))

    def inverse(self, x: torch.Tensor):
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, :-self.n_ignore], x[:, -self.n_ignore:]
        else:
            data = x

        data = data.reshape([len(data), -1, self.vector])

        data = rearrange(data, 'b d v -> v b d')

        data = torch.stack([data[v]*self.k[v] for v in range(self.vector)]).sum(dim=0)

        if self.n_ignore:
            data = torch.cat([data, ignored], dim=-1)
        else:
            data = data
        return data.reshape(*batch, dim)



def get_norm(
    norm: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"],
    **kwargs: Any,
) -> nn.Module:
    """
    Get normalization layer.

    Parameters
    ----------
    norm : Literal["layer", "rms", "group", "batch", "none"]
        Normalization layer. If it's set to "none", normalization is not applied.
    kwargs : dict
        Normalization layer configuration.

    Returns
    -------
    nn.Module
        Normalization layer.

    Examples
    --------
    >>> cfg = {"normalized_shape": 1, "eps": 1e-05, "elementwise_affine": True, "bias": True}
    >>> norm = get_norm("layer", **cfg)
    >>> norm
    LayerNorm((1,), eps=1e-05, elementwise_affine=True)

    >>> cfg = {"normalized_shape": 1, "eps": 1e-05, "elementwise_affine": True}
    >>> norm = get_norm("rms", **cfg)
    >>> norm
    RMSNorm((1,), eps=1e-05, elementwise_affine=True)

    >>> cfg = {"num_groups": 1, "num_channels": 12, "eps": 1e-05, "affine": True}
    >>> norm = get_norm("group", **cfg)
    >>> norm
    GroupNorm(1, 12, eps=1e-05, affine=True)

    >>> cfg = {"num_features": 1, "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True}
    >>> norm = get_norm("batch2d", **cfg)
    >>> norm
    BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> cfg = {"num_features": 1, "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True}
    >>> norm = get_norm("batch1d", **cfg)
    >>> norm
    BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> norm = get_norm("none")
    >>> norm
    Identity()

    """
    if norm == "layer":
        return nn.LayerNorm(**kwargs)
    if norm == "rms":
        return nn.RMSNorm(**kwargs)
    if norm == "group":
        return nn.GroupNorm(**kwargs)
    if norm == "batch2d":
        return nn.BatchNorm2d(**kwargs)
    if norm == "batch1d":
        return nn.BatchNorm1d(**kwargs)
    return nn.Identity()


class ConvNormActivation(nn.Module):
    """
    Convolutional layer with normalization and activation, and dropouts.

    References
    ----------
    PixelShuffle: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    PixelUnshuffle: https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
    BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    GroupNorm: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    InstanceNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
    Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.

    Examples
    --------
    >>> cfg = ConvConfig(
    ...     activation="ReLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.1,
    ...     norm="batch",
    ...     norm_cfg={"affine": True, "track_running_stats": True},
    ...     scale_factor=0
    ... )
    >>> conv = ConvNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])

    >>> cfg = ConvConfig(
    ...     activation="SiGLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.0,
    ...     norm="none",
    ...     norm_cfg={},
    ...     scale_factor=2
    ... )
    >>> conv = ConvNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 64, 64])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
    ) -> None:
        super().__init__()

        out_channels_ = out_channels
        if "glu" in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor) ** 2
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor) ** 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels_,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode
        )
        if cfg.norm != "none" and cfg.norm != "group":
            cfg.norm_cfg["num_features"] = out_channels_
        elif cfg.norm == "group":
            cfg.norm_cfg["num_channels"] = in_channels if cfg.norm_first else out_channels_

        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        if cfg.scale_factor > 0:
            self.pixel_shuffle = nn.PixelShuffle(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            self.pixel_shuffle = nn.PixelUnshuffle(abs(cfg.scale_factor))
        else:
            self.pixel_shuffle = nn.Identity()
        self.activation = Activation(cfg.activation, dim=-3)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W) or (in_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H', W') or (out_channels, H', W')
        H' and W' are calculated as follows:
        H' = (H + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1
        H' = H' * scale_factor if scale_factor > 0 else H' // abs(scale_factor) if scale_factor < 0 else H'
        W' = (W + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1
        W' = W' * scale_factor if scale_factor > 0 else W' // abs(scale_factor) if scale_factor < 0 else W'

        """
        if self.norm_first:
            x = self.norm(x)
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.pixel_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class Attention2d(nn.Module):
    def __init__(
        self,
        channels: int,
        nhead: Optional[int] = None,
        patch_size: int = 1,
        attn_cfg=None,
    ):
        super().__init__()
        self.channels = channels

        if nhead is None or patch_size is None:
            assert attn_cfg is not None
            self.n_heads = attn_cfg.nhead
            self.patch_size = attn_cfg.patch_size
        else:
            self.n_heads = nhead
            self.patch_size = patch_size

        assert channels % self.n_heads == 0

        cfg = ConvConfig(
            kernel_size=1,
            padding=0,
            stride=1,
            activation="Identity",
            dropout=0.0,
        )
        first_cfg = deepcopy(cfg)
        first_cfg.norm_first = True

        self.qkv = ConvNormActivation(channels, channels * 3, first_cfg)
        self.proj_out = ConvNormActivation(channels, channels, cfg)

    def qkv_attn(self, qkv):
        bs, height, width, total_channels = qkv.shape
        from einops import rearrange

        # 最後の次元が (heads, qkv=3, ch) の順に並んでいると仮定
        q, k, v = rearrange(
            qkv, 
            "b h w (heads qkv ch) -> qkv (b heads) (h w) ch", 
            qkv=3, 
            heads=self.n_heads
        )

        ch = total_channels // (3 * self.n_heads)
        scale = 1.0 / np.sqrt(np.sqrt(ch))

        weight = torch.einsum("btc,bsc->bts", q * scale, k * scale)
        weight = weight - torch.max(weight, dim=-1, keepdim=True)[0]
        weight = F.softmax(weight, dim=-1)

        a = torch.einsum("bts,bsc->btc", weight, v)

        return rearrange(
            a, "(b heads) (h w) ch -> b h w (heads ch)", b=bs, heads=self.n_heads, h=height, w=width
        )

    def forward(self, x):
        # 1. 畳み込みは NCHW で実行
        qkv = self.qkv(x)

        # 2. Attentionの計算のために NHWC に変換
        qkv = qkv.permute(0, 2, 3, 1)

        # 3. Patch Rearrange
        qkv = rearrange(
            qkv,
            "b (h p1) (w p2) c -> b h w (c p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # 4. 核心のアテンション計算
        a = self.qkv_attn(qkv)

        # 5. Patchを元に戻す
        a = rearrange(
            a,
            "b h w (c p1 p2) -> b (h p1) (w p2) c",
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # 6. 畳み込み用に NCHW に戻す
        a = a.permute(0, 3, 1, 2)
        a = self.proj_out(a)

        return x + a