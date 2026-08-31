from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Union, Any, Optional

from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning import Callback


def load_config(path: str) -> DictConfig:
    """Convert model config `.yaml` to `Dictconfig` with custom resolvers."""
    path = Path(path)

    config = OmegaConf.load(path)

    return config

@dataclass
class RNNConfig:
    bias: bool = True


@dataclass
class ExperimentConfig:
    data_dir: str
    seed: int
    log_every: int
    embed_obs_dim: int
    obs_shape: tuple[int, ...]
    action_dim: int
    alpha: float
    device: int
    activation: str
    chunk_size: int
    idx_splitter: IndexConfig
    batch_size: BatchSizeConfig
    datamodule: DatasetConfig
    epochs: EpochConfig
    trainer: TrainerConfig
    # callbacks: list
    world: Union[WorldConfig, EstimaterConfig] = None

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)

    # def tuple_callbacks(self):
    #     self.callbacks = tuple(self.callbacks)

    def dc2dict(self):
        if self.world is not None:
            self.world.dc2dict()

@dataclass
class DatasetConfig:
    seed: int
    data_dir: str
    chunk_size: int
    action_diff: bool = False
    use_stoch: Literal["posterior", "prior"] = "posterior"
    noise: bool = True
    future_pred_len: int = 0
    data_length: int = 100
    unified_policy: bool = False
    interval: int = 1
    binarize_code: bool = False
    freeze_padding: int = False
    # Parameters used by the current RNN_experiment.py simulation loader.
    data_root: str = "data"
    behaviour: str = "adult"
    env: str = "box_messy"
    env_dim: float = 0.635
    stride: int = 10
    num_workers: int = 0


@dataclass
class IndexConfig:
    num: int
    change_point: tuple[int, ...]
    n_val_each: int

    def __post_init__(self):
        self.change_point = tuple(self.change_point)

@dataclass
class EpochConfig:
    world: int = 0

@dataclass
class BatchSizeConfig:
    world: int = 0

@dataclass
class TrainerConfig:
    accelerator: str
    devices: tuple[int, ...]
    deterministic: bool
    precision: int
    log_every_n_steps: int
    check_val_every_n_epoch: int

    def __post_init__(self):
        self.devices = tuple(self.devices)


@dataclass
class WorldConfig:
    obs_shape: tuple[int, ...]
    obs_dim: int
    action_dim: int
    alpha: float
    activation: str
    optimizer_cfg: DictConfig
    loss_cfg: DictConfig
    dynamics_cfg: Union[RSSMConfig, MTRSSMConfig, CRSSMV4Config]
    encoder_cfg: Union[ViTConfig, CNNConfig, MLPConfig]
    decoder_cfg: Union[ViTConfig, CNNConfig, MLPConfig, States2Map1dConfig] = None
    c_decoder_cfg: MLPConfig = None
    contrastive_class: int = 128
    contrastive_hidden: int = 128
    contrastive_activation: DictConfig = DictConfig({
        "activation": "Tanh",
        "reparametarize_fn": "gelu",
    })
    hypernetize: bool = False
    shared_decoder: bool = False
    shared_critic: bool = True
    load_name: str = None
    n_policy: int = 0
    use_vqvae: bool = True
    binarize_code: bool = False
    action_transformation: SoftmaxTransConfig = None
    finetune_decoder: int = False
    truncated: int = 0

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)

    def dc2dict(self):
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.contrastive_activation = dict(self.contrastive_activation)

    def dict2dc(self):
        self.optimizer_cfg = DictConfig(self.optimizer_cfg)
        self.loss_cfg = DictConfig(self.loss_cfg)
        self.contrastive_activation = DictConfig(self.contrastive_activation)



@dataclass
class GateL0RDConfig:
    activation: str
    dense_hidden_dim: int = None
    layers: int = 1
    fix_sigma: float = False
    straight_through: bool = False
    common_gate: bool = False

@dataclass
class SparseGateL0RDConfing:
    activation: str
    layers: int = 1
    n_class: int = 4
    dense_hidden_dim: int = 0
    fix_sigma: float = False
    straight_through: bool = False
    common_gate: bool = False
    apply_softmax: bool = True
    hidden_recurrence: bool = True
    
@dataclass
class CRSSMV4Config:
    determ_dim: int
    coarse_dim: int
    stoch_cfg: DistributionConfig
    coarse_stoch_cfg: DistributionConfig
    init_from_: str
    init_with_: str = "prior"
    precise_rnn: str = "GRU"
    coarse_rnn: str = "SparseGateL0RD"
    precise_cfg: Union[RNNConfig, MTRNNConfig] = field(default_factory=RNNConfig)
    coarse_cfg: Union[RNNConfig, MTRNNConfig] = field(default_factory=SparseGateL0RDConfing)
    coarse_obs: Literal["determ", "obs"] = "obs"

@dataclass
class CRSSMConfig:
    determ_dim: int
    coarse_dim: int
    stoch_cfg: DistributionConfig
    init_from_: str
    init_with_: str = "prior"
    precise_rnn: str = "GRU"
    coarse_rnn: str = "SparseGateL0RD"
    precise_cfg: Union[RNNConfig, MTRNNConfig, GateL0RDConfing] = field(default_factory=RNNConfig)
    coarse_cfg: Union[RNNConfig, MTRNNConfig, SparseGateL0RDConfing] = field(default_factory=SparseGateL0RDConfing)
    coarse_obs: Literal["determ", "obs"] = "obs"


@dataclass
class States2Map1dConfig:
    init_channel: int
    kernel_size: int
    hidden_dim: int
    n_layers: int
    hidden_activation: str
    output_activation: str
    layer_norm: bool = False
    conv_target: Literal["map", "state"] = "state"
    transformer_cfg: TransformerConfig = None

@dataclass
class States2Map2dConfig:
    init_channel: int
    kernel_size: int
    hidden_dim: int
    n_layers: int
    hidden_activation: str
    output_activation: str
    layer_norm: bool = False
    transformer_cfg: TransformerConfig = None

@dataclass
class Map2StatesConfig:
    hidden_dim: int
    n_layers: int
    hidden_activation: str
    output_activation: str
    kernel_size: int
    stride: int
    padding: int = 0
    layer_norm: bool = False
    is_1d: bool = False

@dataclass
class VQConfig:
    codebook_size: int
    feature_map_scale: int
    layers: int = 3
    # cnn_cfg: DictConfig

@dataclass
class VQWorldConfig:
    obs_shape: tuple[int, ...]
    obs_dim: int
    action_dim: int
    alpha: float
    activation: str
    output_activation: str
    vq_name: str 
    optimizer_cfg: DictConfig
    loss_cfg: DictConfig
    codebook_size: int
    feature_map_scale: int
    dynamics_cfg: Union[RSSMConfig, MTRSSMConfig]
    input_rssm: Literal["one_hot", "embed", "quantized"]
    encoder_cfg: Union[ViTConfig, CNNConfig] = None
    decoder_type: Literal["MLP", "1dconv1", "1dconv2","2dconv"] = "MLP"
    embed_size: int = 64
    tokens_dim: int = 1024
    contrastive_class: int = 128
    contrastive_hidden: int = 128
    contrastive_activation: DictConfig = DictConfig({
        "activation": "Tanh",
        "reparametarize_fn": "gelu",
    })
    hypernetize: bool = False
    mode: Literal["aif", "dreamer-v2", "dreamer-v3"] = "aif"
    reward_cfg: MLPConfig = None

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)

    def dc2dict(self):
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.contrastive_activation = dict(self.contrastive_activation)


@dataclass
class ptWorldConfig:
    obs_shape: tuple[int, ...]
    obs_dim: int
    action_dim: int
    alpha: float
    activation: str
    output_activation: str
    optimizer_cfg: DictConfig
    loss_cfg: DictConfig
    dynamics_cfg: Union[RSSMConfig, MTRSSMConfig]
    encoder_cfg: Union[ViTConfig, CNNConfig]
    decoder_cfg: Union[ViTConfig, CNNConfig] = None
    input_range: str = "norm"
    contrastive_class: int = 128
    contrastive_hidden: int = 128
    contrastive_activation: DictConfig = DictConfig({
        "activation": "Tanh",
        "reparametarize_fn": "gelu",
    })
    hypernetize: bool = False
    mode: Literal["aif", "dreamer-v2", "dreamer-v3"] = "aif"
    reward_cfg: MLPConfig = None
    decode: str = "MLP"

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)

    def dc2dict(self):
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.contrastive_activation = dict(self.contrastive_activation)

# @dataclass
# class PolicyConfig:
#     mlp_cfg: MLPConfig
#     optimizer_cfg: DictConfig
#     mean_scale: float = 5.0
#     init_std: float = 5.0
#     min_std: float = 1e-4

#     def dc2dict(self):
#         self.optimizer_cfg = dict(self.optimizer_cfg)


@dataclass
class AgentConfig:
    action_dim: int
    world_cfg: Union[WorldConfig, VQWorldConfig]
    

    alpha: float
    preferred_obs: str
    preferred_num: int
    pref_dist: str
    pref_std: float
    truncated: bool = False
    truncated_step: int = 20


    def dc2dict(self):
        # self.value_optimizer_cfg = dict(self.value_optimizer_cfg)
        self.world_cfg.dc2dict()
        # self.efe_policy_cfg.dc2dict()

@dataclass
class Agent_d_Config:
    diffusion_cfg: DiffusionConfig
    
    def dc2dict(self):
        self.diffusion_cfg.dc2dict()


@dataclass
class Agent_Config:
    action_dim: int
    # world_cfg: WorldConfig
    

    alpha: float
    preferred_obs: str
    preferred_num: int
    pref_dist: str
    pref_std: float
    vae_cfg: DictConfig

@dataclass
class ptConfig:
    VQtype: str

@dataclass
class CNNConfig:
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    paddings: tuple[int, ...]
    hidden_activation: str
    output_activation: str
    batch_norm: bool = False
    fc_hidden: tuple[int, ...] = None
    spatial: bool = False
    spatial_temprature: float = 0.0
    both_spatial: bool = False

    def __post_init__(self):
        assert len(self.channels) == len(self.kernels) == len(
            self.strides) == len(self.paddings)
        self.channels = tuple(self.channels)
        self.kernels = tuple(self.kernels)
        self.strides = tuple(self.strides)
        self.paddings = tuple(self.paddings)
        if self.fc_hidden is not None:
            self.fc_hidden = tuple(self.fc_hidden)


# @dataclass
# class CriticQuantizedConfig:
#     max: int
#     min: int
#     n_class: int


@dataclass
class MLPConfig:
    hidden_dim: int
    n_layers: int
    hidden_activation: str
    output_activation: str
    layer_norm: bool = False
    dropout: float = 0.0

@dataclass
class ml_MLPConfig:
    hidden_dim: int
    n_layers: int
    output_activation: str
    linear_cfg: LinearConfig = None
    


@dataclass
class TransformerConfig:
    d_model: int
    nhead: int
    hidden_dim: int
    n_layers: int
    dropout: float = 0.1
    hidden_activation: str = "ReLU"
    output_activation: str = "GeLU"
    hypernetize: bool = False
    attn_hypernetize: bool = False


@dataclass
class ViTConfig:
    patch_size: int
    vit_cfg: TransformerConfig
    proj_activation: str = "Mish"


@dataclass
class ResNetConfig:
    conv_channel: int
    conv_padding: int
    final_padding: int
    conv_activation: str
    mlp_activation: str
    out_activation: str
    n_res_blocks: int
    batch_norm: bool = False
    upscale_factor: int = 2
    n_upsampling: int = 2
    use_mask: bool = False
    init_channel: int = 1
    
@dataclass
class SRGANConfig:
    feature_dim: int
    num_res: int
    upscale: int
    in_channel: int = 1
    activation: str = "PReLU"
    output_activation: str = "Tanh"

@dataclass
class RNNConfig:
    bias: bool = True
    

@dataclass
class ml_ResNetConfig:
    conv_channel: int
    conv_kernel: int
    f_kernel: int
    conv_activation: str
    out_activation: str
    n_res_blocks: int
    scale_factor: int = 2
    n_scaling: int = 2
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.0
    init_channel: int = 16
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
    attention: AttentionConfig | None = None

    def __post_init__(self) -> None:
        """Set `norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(self.norm_cfg)

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `ResNetConfig`."""
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig | ListConfig | list | tuple | dict):
                setattr(self, key, convert_dictconfig_to_dict(value))

@dataclass
class AttentionConfig:
    """
    Attention configuration.

    Attributes
    ----------
    nhead : int
        Number of heads.
    patch_size : int
        Patch size.
    """

    nhead: int
    patch_size: int
    

@dataclass
class MTRNNConfig:
    tau: int
    tau_sample: bool = False
    bias: bool = True
    return_gate: bool = False
    apply_tanh: bool = True


@dataclass
class DistributionConfig:
    stoch_dim: int
    hidden_dim: int
    dist: str
    layers: int
    activation: str
    n_class: int = 1
    temprature: float = 1.0
    spherical: bool = False


@dataclass
class RSSMConfig:
    determ_dim: int
    stoch_cfg: DistributionConfig
    init_from_: str
    init_with_: str = "prior"
    rnn_name: str = "GRU"
    rnn_cfg: Union[RNNConfig, MTRNNConfig] = field(default_factory=RNNConfig)
    
@dataclass
class MTRSSMConfig:
    """Two-level RSSM: fast ``lower_cfg`` + slow ``higher_cfg`` (e.g. both MTRNN)."""

    lower_cfg: RSSMConfig
    higher_cfg: RSSMConfig
    temporal_abstraction: int = 5
    top_obs: Literal["determ", "stoch", "both"] = "determ"

@dataclass
class SoftmaxTransConfig:
    vector: int
    sigma: float
    n_ignore: int = 0
    max: float = 1.0
    min: float = -1.0
    
@dataclass
class ContrastiveLearningConfig:
    dim_feature: int
    eval_func: MLPConfig
    negative_alpha: int = 1
    dim_input2: Optional[int] = None
    cross_entropy_like: bool = False
    positive_range_self: int = None
    positive_range_tgt: int = None
    

@dataclass
class SpatialSoftmaxConfig:
    """
    Spatial softmax configuration.

    Attributes
    ----------
    temperature : float
        Softmax temperature. If it's set to 0.0, the layer outputs the coordinates of the maximum value.
        Otherwise, the layer outputs the expectation of the coordinates with softmax function.
        Default is 0.0.
    """

    temperature: float = 1.0
    eps: float = 1e-6
    is_argmax: bool = False
    is_straight_through: bool = False

@dataclass
class EstimaterConfig:
    obs_shape: Tuple[int, ...]
    encoder: EncoderConfig
    version: Literal['v1', 'v2', 'v3']
    optimizer_cfg: DictConfig
    loss_cfg: DictConfig

    def __post_init__(self):
        self.obs_shape = tuple(self.obs_shape)
        assert self.version in ['v1', 'v2', 'v3'], f"Invalid version: {self.version}"


    def dictcfg2dict(self) -> None:
        """Convert OmegaConf DictConfig to a dictionary."""
        self.optimizer_cfg = dict(self.optimizer_cfg)
        self.loss_cfg = dict(self.loss_cfg)
        self.encoder.dictcfg2dict()

@dataclass
class Dataset_d_Config:
    path: str
    batch_size: int
    n_iterations: int
    seed: int
    adjusting_methods: Optional[tuple[str]] = None
    max_difference: Optional[float] = None
    normalize_distance: bool = True

    def __post_init__(self):
        if isinstance(self.adjusting_methods, list) or isinstance(self.adjusting_methods, ListConfig):
            self.adjusting_methods = tuple(self.adjusting_methods)


@dataclass 
class EncoderConfig:
    """
    Encoder configuration.

    Attributes
    ----------
    backbone: Union[ConvNetConfig, ResNetConfig]
        Backbone configuration.
    full_connection: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]
        Full connection configuration.
    """

    backbone: Union[ConvNetConfig, ResNetConfig, ml_ResNetConfig]
    full_connection: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `EncoderConfig`.

        Returns
        -------
        dict
            Dictionary representation of EncoderConfig.
        """
        self.backbone.dictcfg2dict()
        if hasattr(self.full_connection, "dictcfg2dict"):
            self.full_connection.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


@dataclass
class ConvConfig:
    """
    A convolutional layer configuration.

    Attributes
    ----------
    activation : str
        Activation function.
    kernel_size : int
        Kernel size.
    stride : int
        Stride.
    padding : int
        Padding.
    output_padding : int
        Output padding, especially for transposed convolution. Default is 0.
    dilation : int
        Dilation.
    groups : int
        Number of groups. Default is 1. See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
    bias : bool
        Whether to use bias. Default is True.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.
    norm : Literal["batch", "group", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : dict
        Normalization layer configuration. If you want to use Instance, Layer, or Group normalization,
        set norm to "group" and set norm_cfg with "num_groups=$in_channel, 1, or any value". Default is {}.
    scale_factor : int
        Scale factor for upsample, especially for PixelShuffle or PixelUnshuffle.
        If it's set to >0, upsample is applied. If it's set to <0 downsample is applied.
        Otherwise, no upsample or downsample is applied. Default is 0.


    """

    activation: str
    kernel_size: int
    stride: int
    padding: int
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
    dropout: float = 0.0
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: Dict[str, Any] = field(default_factory=dict)
    norm_first: bool = False
    scale_factor: int = 0

    def __post_init__(self) -> None:
        """Set `.norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `ConvConfig`.

        Returns
        -------
        dict
            Dictionary representation of ConvConfig.
        """
        self.norm_cfg = dict(self.norm_cfg)

        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))
        


@dataclass
class ConvNetConfig:
    """
    Convolutional neural network layers configuration.

    Attributes
    ----------
    channels : Tuple[int, ...]
        Number of channels for each layer.
    conv_cfgs : Tuple[ConvConfig, ...]
        Convolutional layer configurations.
        The length of conv_cfgs should be the same as the length of channels.
    init_channel : int
        Initial number of channels, especially for transposed convolution.
    """

    channels: Tuple[int, ...]
    conv_cfgs: Tuple[ConvConfig, ...]
    init_channel: int = 16

    def __post_init__(self) -> None:
        """Set `channels` and `conv_cfgs` as tuple."""
        self.conv_cfgs = tuple(self.conv_cfgs)
        self.channels = tuple(self.channels)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `ConvNetConfig`.

        Returns
        -------
        dict
            Dictionary representation of ConvNetConfig.
        """
        self.channels = tuple(self.channels)
        conv_cfgs = []
        for conv_cfg in self.conv_cfgs:
            if isinstance(conv_cfg, DictConfig):
                conv_cfg = ConvConfig(**conv_cfg)
            conv_cfg.dictcfg2dict()
            conv_cfgs.append(conv_cfg)
        self.conv_cfgs = tuple(conv_cfgs)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))



@dataclass
class LinearConfig:
    """
    A linear layer configuration.

    Attributes
    ----------
    activation : str
        Activation function.
    norm : Literal["layer", "rms", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : dict
        Normalization layer configuration. Default is {}.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.
    norm_first : bool
        Whether to apply normalization before linear layer. Default is False.
    bias : bool
        Whether to use bias. Default is True.
    """

    activation: str
    norm: Literal["layer", "rms", "none"] = "none"
    norm_cfg: Dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.0
    norm_first: bool = False
    bias: bool = True

    def __post_init__(self) -> None:
        """Set `norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `LinearConfig`.

        Returns
        -------
        dict
            Dictionary representation of LinearConfig.
        """
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


#自分用にアレンジ
@dataclass
class DiffusionConfig:
    # Inputs / output structure.
    n_obs_steps: int = 2 #過去何ステップ分の観測画像を与えるか
    horizon: int = 16 #生成するアクションの長さ
    n_action_steps: int = 8 #実際に実行されるアクションの長さ

    input_shapes: tuple[int, ...] = (3, 120, 160)
    output_shapes: int = 2

    # # Normalization / Unnormalization 正規化手法の指定
    # input_normalization_modes: dict[str, str] = field(
    #     default_factory=lambda: {
    #         "observation.image": "mean_std",
    #         # "observation.state": "min_max",
    #     }
    # )
    # output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18" #画像のエンコーダ
    crop_shape: tuple[int, int] | None = (64, 64) #エンコーダに入れる前に画像をクロップするか
    crop_is_random: bool = True #クロップする領域をランダムにするか
    pretrained_backbone_weights: str | None = None #事前学習済みの重みを使うか
    use_group_norm: bool = True #バッチノームをグループノームに置き換えるか
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True #Unetのconditionの方法にFiLMを使うか
    activate : str = "ReLU"
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100 #拡散ステップ数
    beta_schedule: str = "squaredcos_cap_v2" #DDPMスケジューラ
    beta_start: float = 0.0001 #最初のbeta
    beta_end: float = 0.02 #最後のbeta
    prediction_type: str = "epsilon" #"epsilon" or "sample"
    clip_sample: bool = True #拡散ステップごとにサンプルを[-range, range]でクリップするか：アクションが正規化されている必要あり
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None #逆拡散ステップ数．特に指定がなければ学習時の拡散ステップ数と同じ

    # Loss computation
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        self.input_shapes = tuple(self.input_shapes)

    def dc2dict(self):
        pass
