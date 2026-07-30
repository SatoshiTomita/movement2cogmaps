@dataclass
class RSSMConfig:
    determ_dim: int
    stoch_cfg: DistributionConfig
    init_from_: str
    init_with_: str = "prior"
    rnn_name: str = "GRU"
    rnn_cfg: Union[RNNConfig, MTRNNConfig] = field(default_factory=RNNConfig)
    