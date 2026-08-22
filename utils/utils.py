import random
from glob import glob
from typing import Dict, Iterator, List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer
import yaml
from omegaconf import OmegaConf
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from schedulefree import RAdamScheduleFree
# import torchvision.transforms.functional as F
import blosc2

def save_blosc2(path: str, x: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(blosc2.pack_array2(x))


def load_blosc2(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())

def symlog(x: torch.Tensor):
    return torch.sign(x)*torch.log(torch.abs(x)+1)


def symexp(x: torch.Tensor):
    return torch.sign(x)*(torch.exp(torch.abs(x))-1)


def conv_out(h_in, padding, kernel_size, stride, dilation):
    return int((h_in + 2.0 * padding - dilation * (kernel_size - 1.0) - 1.0) / stride + 1.0)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride, dilation=1):
    return tuple(conv_out(x, padding, kernel_size, stride, dilation) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )


def re_tanh(x: torch.Tensor):
    return torch.clamp(x.tanh(), 0.0, 1.0)


def get_optimizer(
    param: Iterator[nn.Parameter], name: str, lr: float, amsgrad: bool = False
):
    if name == "RAdamScheduleFree":
        optimizer = RAdamScheduleFree
        return optimizer(param, lr=lr)
    elif hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)
    elif hasattr(torch_optimizer, name):
        optimizer = getattr(torch_optimizer, name)
    else:
        raise NotImplementedError
    kwargs = {"lr": lr}
    if name in {"Adam", "AdamW"}:
        kwargs["amsgrad"] = amsgrad
    return optimizer(param, **kwargs)


class HeavisideStepFnc(torch.autograd.Function):
    """
    Heaviside activation function with straight through estimator
    """

    @staticmethod
    def forward(ctx, input):
        return torch.ceil(input).clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ReTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        binary_gate = re_tanh(x)
        return binary_gate

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output * (1 - x.tanh() ** 2)
        return grad_input

class mytorch:
    @staticmethod
    def concat(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def stack(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.stack(tensor_list, dim=dim)

    @staticmethod
    @torch.jit.script
    def softmax(
        inputs: torch.Tensor, dim: int, temperature: torch.Tensor = torch.tensor(1.0)
    ):

        x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
        x = x / temperature

        x = torch.softmax(x, dim=dim)

        if torch.isinf(x).any() or torch.isnan(x).any():
            print("inputs", inputs)
            print("result", x)
            raise ValueError("softmax is inf or nan")

        return x


# 明るさを変更する関数
def change_brightness(img: torch.Tensor, std: float = 0.1, max: float = 1.0, min: float = -1.0, clamp: bool = True):
    """

    画像(tensor)明るさを変更する関数
    M1のコードそのまま

    Args:
        img(np.ndarray): 元画像
        alpha(float): コントラスト
        beta(float): 明るさ

    Returns:
        torch.Tensor: 明るさを変えた画像

    """
    alpha = torch.normal(1.0, std, size=(1,))

    beta = torch.normal(0.0, std, size=(1,))

    bright_img = alpha * img + beta

    if clamp:
        bright_img = torch.clamp(bright_img, min, max)

    return bright_img


@torch.jit.script
def add_noise(
    input_data: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    max: float = 0.95,
    min: float = -0.95,
    clamp: bool = True,
):
    """関節角度にノイズを加える

    Args:
        angles(np.ndarray): 元関節データ
        alpha(float): ノイズ
        beta(float): バイアス

    Returns:
        torch.Tensor: ノイズを加えた関節データ

    """

    noise = torch.normal(mean, std, size=input_data.shape)

    data = input_data + noise
    if clamp:

        clip_data = torch.clamp(data, min, max)

    else:
        clip_data = data
    return clip_data


def determine_loader(
    data: Dataset, seed: int, batch_size: int, shuffle: bool = True, collate_fn=None
):
    g = torch.Generator()
    g.manual_seed(seed)
    if collate_fn is not None:
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn,
        )
    else:
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=2,
            pin_memory=False,
        )
    return loader


def torch_fix_seed(seed=42):
    """乱数を固定する関数

    各行でやっていることは
        https://qiita.com/north_redwing/items/1e153139125d37829d2d
    などに詳細あり

    """

    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True


def seed_worker(worker_id):
    """

    DataLoaderのworkerの固定
    Dataloaderの乱数固定にはgeneratorの固定も必要らしい

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def remove_keys(d: Dict, _remove_keys: List[str] = None):
    if _remove_keys is None:
        raise ValueError("keys must be required.")
    for k in list(d.keys()):
        v = d[k]
        if k in _remove_keys:
            del d[k]
        if isinstance(v, dict):
            remove_keys(v, _remove_keys)
        elif isinstance(v, list):
            for vd in v:
                remove_keys(vd, _remove_keys)


def initialize_weight(
    weight: torch.Tensor,
    distribution: Optional[str],
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initialize weight tensor using the specified distribution and nonlinearity function.

    Args:
        weight (torch.Tensor): Tensor to be initialized.
        distribution (str, optional): Distribution to use for initialization.
        nonlinearity (str, optional): Nonlinearity function to use. Defaults to "LeakyReLU".

    Raises:
        ValueError: When the specified distribution is not supported.
    """
    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu", "mish", "tanhexp", "elu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(
            weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported weight distribution '{distribution}'")


def initialize_bias(bias: torch.Tensor, distribution: Optional[float] = 0.0) -> None:
    """
    Initializes the bias tensor of a layer using the given distribution.

    Args:
        bias (nn.Parameter): the bias tensor to be initialized
        distribution (float): the distribution to use for initialization, default is 0 (constant)

    Raises:
        ValueError: When the specified distribution is not supported.
    """
    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
        return

    raise ValueError(f"Unsupported bias distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0.0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initializes the weight and bias tensors of a linear or convolutional layer using the given distribution.

    Args:
    - layer (nn.Module): the linear or convolutional layer to be initialized
    - distribution (str): the distribution to use for initialization, default is 'kaiming_normal'
    - init_bias (float): the initial value of the bias tensor, default is 0
    - nonlinearity (str): the nonlinearity function to use for initialization, default is "LeakyReLU"

    Returns:
    - None
    """
    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(layer.bias, init_bias)

# def resize_im(
#     input_data:Torch.Tensor, #channel, w, h
#     ):
#     resized_im = F.resize(img = input_data, size=64,64)
#     return resized_im

def populate_queues(queue, tensor):
    if len(queue) != queue.maxlen: #条件付けする観測の長さor実際に使う行動の長さ分，データからキューを追加
        # initialize by copying the first observation several times until the queue is full
        while len(queue) != queue.maxlen:
            queue.append(tensor)
    else:
        # add latest observation to the queue
        queue.append(tensor)
    return queue


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype

class ChannelProcessor:
    def __init__(self, epsilon=1e-8):
        """
        Args:
            epsilon (float): ゼロ分割を防ぐための小さな値。
        """
        self.epsilon = epsilon
        self.max_min = np.load("/home/yokozawa/work/turtlebot4_ws/turtlebot4_ws/dataset/1023/stats/max_min.npy", allow_pickle=True).item()
        self.mean_std = np.load("/home/yokozawa/work/turtlebot4_ws/turtlebot4_ws/dataset/1023/stats/mean_std.npy", allow_pickle=True).item()
    
    def normalize(self, tensor):
        """
        Args:
            tensor (torch.Tensor): 入力テンソル (shape: [b*l, c, h, w])
        
        Returns:
            torch.Tensor: 正規化されたテンソル。
        """
        max_vals = self.max_min["max_val"]  # shape: [b*l, c, 1, 1]
        min_vals = self.max_min["min_val"]  # shape: [b*l, c, 1, 1]
        max_vals = max_vals.view(1, 16, 1, 1)
        min_vals = min_vals.view(1, 16, 1, 1)
        return (tensor - min_vals) / (max_vals - min_vals + self.epsilon)
    
    def unnormalize(self, tensor):
        """
        Args:
            tensor (torch.Tensor): 正規化されたテンソル (shape: [b*l, c, h, w])
        
        Returns:
            torch.Tensor: 元のスケールに戻したテンソル。
        """
        max_vals = self.max_min["max_val"]  # shape: [b*l, c, 1, 1]
        min_vals = self.max_min["min_val"]  # shape: [b*l, c, 1, 1]
        max_vals = max_vals.view(1, 16, 1, 1)
        min_vals = min_vals.view(1, 16, 1, 1)
        return tensor * (max_vals - min_vals + self.epsilon) + min_vals
    
    def standardize(self, tensor):
        """
        Args:
            tensor (torch.Tensor): 入力テンソル (shape: [b*l, c, h, w])
        
        Returns:
            torch.Tensor: 標準化されたテンソル。
        """
        mean_vals = self.mean_std["mean_val"]  # shape: [b*l, c, 1, 1]
        std_vals = self.mean_std["std_val"]  # shape: [b*l, c, 1, 1]
        mean_vals = mean_vals.view(1, 16, 1, 1)
        std_vals = std_vals.view(1, 16, 1, 1)
        return (tensor - mean_vals) / (std_vals + self.epsilon)

    def unstandardize(self, tensor):
        """
        Args:
            tensor (torch.Tensor): 標準化されたテンソル (shape: [b*l, c, h, w])
        
        Returns:
            torch.Tensor: 元のスケールに戻したテンソル。
        """
        mean_vals = self.mean_std["mean_val"]  # shape: [b*l, c, 1, 1]
        std_vals = self.mean_std["std_val"]  # shape: [b*l, c, 1, 1]
        mean_vals = mean_vals.view(1, 16, 1, 1)
        std_vals = std_vals.view(1, 16, 1, 1)
        return tensor * (std_vals + self.epsilon) + mean_vals

    def __call__(self, tensor, mode="normalize"):
        """
        Args:
            tensor (torch.Tensor): 入力テンソル (shape: [b*l, c, h, w])
            mode (str): "normalize" または "standardize" を指定。
        
        Returns:
            torch.Tensor: 処理されたテンソル。
        """
        if mode == "normalize":
            return self.normalize(tensor)
        elif mode == "standardize":
            return self.standardize(tensor)
        elif mode == "unnormalize":
            return self.unnormalize(tensor)
        elif mode == unstandardize(tensor):
            return self.unstandardize(tensor)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'normalize' or 'standardize'.")


def actions_to_positions(actions, delta_t=0.1):
    """
    行動テンソル[length, batch, dim]を相対位置[x, y, θ]に変換する。
    
    Args:
        actions (torch.Tensor): [length, batch, 2] のテンソル。
                               dim=2: [旋回速度 (ω), 直線速度 (v)]。
        delta_t (float): 各時刻間の時間差（デフォルト: 1.0）。
    
    Returns:
        positions (torch.Tensor): [length, batch, 3] の位置テンソル。
                                  dim=3: [x, y, θ]。
    """
    length, batch, _ = actions.shape

    # 初期位置 (x=0, y=0, θ=0) を設定
    positions = torch.zeros((length, batch, 3))  # dim=3: [x, y, θ]
    device = torch.device("cuda")
    delta_t = torch.tensor(delta_t).to(device)
    actions[:, :, 0] = actions[:, :, 0]*0.5
    actions[:, :, 1] = actions[:, :, 1]*0.2
    for t in range(1, length):
        # 直前の位置
        x_prev, y_prev, theta_prev = positions[t - 1, :, 0], positions[t - 1, :, 1], positions[t - 1, :, 2]
        
        # 現在の行動
        omega, velocity = actions[t - 1, :, 0], actions[t - 1, :, 1]

        # 現在の角度（θ）の更新
        theta = theta_prev.to(device) + omega.to(device) * delta_t
        
        # 位置 (x, y) の更新
        x = x_prev.to(device) + velocity.to(device) * torch.cos(theta).to(device) * delta_t
        y = y_prev.to(device) + velocity.to(device) * torch.sin(theta).to(device) * delta_t
        
        # 更新結果を格納
        positions[t, :, 0] = x
        positions[t, :, 1] = y
        positions[t, :, 2] = theta

    return positions

def positions_to_actions(positions: torch.Tensor) -> torch.Tensor:
    """
    Convert positions back to actions with zero-padding at t=length.

    Args:
        positions (torch.Tensor): A tensor of shape [length, batch, 3] representing positions (x, y, theta).

    Returns:
        torch.Tensor: A tensor of shape [length, batch, 2] representing actions (angular_velocity, linear_velocity).
    """
    # Compute the differences between consecutive positions
    delta_positions = positions[1:] - positions[:-1]  # Shape: [length-1, batch, 3]
    
    # Extract changes in theta, x, and y
    delta_theta = delta_positions[..., 2]
    delta_x = delta_positions[..., 0]
    delta_y = delta_positions[..., 1]
    
    # Calculate angular and linear velocities
    linear_velocity = torch.sqrt(delta_x**2 + delta_y**2)  # Magnitude of the displacement
    angular_velocity = delta_theta  # Change in orientation (theta)

    # Combine velocities into a single tensor
    actions = torch.stack([angular_velocity, linear_velocity], dim=-1)  # Shape: [length-1, batch, 2]

    # Zero-pad the final action to match the length of positions
    zero_action = torch.zeros(1, actions.size(1), 2, device=actions.device, dtype=actions.dtype)  # Shape: [1, batch, 2]
    actions_padded = torch.cat([actions, zero_action], dim=0)  # Shape: [length, batch, 2]

    return actions_padded
