from torch.utils.data import Dataset
import torch


class WindowedPredictionDataset(Dataset):
    """
    [video：シーン画像の時系列, velocity：線形速度, rot_velocity：角速度, positions：位置座標, thetas：方角,window_size：ウィンドウサイズ, n_future_pred：未来予測ステップ数] -> [inputs:時間窓のシーン画像, vel:時間窓の線形速度, rot_vel:時間窓の角速度, pos:時間窓の位置座標, thet:時間窓の方角, labels:時間窓の次ステップの画像]
    """

    def __init__(self, video, velocity, rot_velocity, positions, thetas,
                 window_size, n_future_pred=1):
        self.scene_in = torch.from_numpy(video) if video is not None else None
        self.scene_out = self.scene_in

        self.velocity = torch.from_numpy(velocity)
        self.rot_velocity = torch.from_numpy(rot_velocity)
        self.positions = torch.from_numpy(positions)
        self.thetas = torch.from_numpy(thetas)

        self.window_size = window_size
        self.n_future_pred = n_future_pred

    def __getitem__(self, index):
        """Return (inputs, vel, rot_vel, pos, theta, labels) for the given window index."""
        if not (0 <= index < len(self)):
            raise ValueError("Index out of range")

        s0 = index * self.window_size
        e0 = s0 + self.window_size
        inputs = self.scene_in[:, s0:e0] if self.scene_in is not None else torch.Tensor([])

        vel, rot_vel, pos, thet, label = [], [], [], [], []
        for f in range(self.n_future_pred):
            s, e = s0 + f, e0 + f
            vel.append(self.velocity[:, s:e])
            rot_vel.append(self.rot_velocity[:, s:e])
            pos.append(self.positions[:, s:e])
            thet.append(self.thetas[:, s:e])

            label.append(self.scene_out[:, s+1:e+1])

        vel = torch.stack(vel, dim=1)
        rot_vel = torch.stack(rot_vel, dim=1)
        pos = torch.stack(pos, dim=1)
        thet = torch.stack(thet, dim=1)
        label = torch.stack(label, dim=1)

        return inputs, vel, rot_vel, pos, thet, label

    def __len__(self):
        """Number of non-overlapping windows available."""
        # 最後の未来ラベルまで確保できる完全なウィンドウ数を返す。
        return (
            self.positions.shape[1] - self.n_future_pred
        ) // self.window_size
