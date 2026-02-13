from torch.utils.data import Dataset
import torch
import numpy as np

from utils.grid_cells import GridCells


class WindowedPredictionDataset(Dataset):
    """Dataset that splits time-series data into fixed-size windows for next-step prediction.

    Extends the base windowed dataset with grid-cell activations computed from
    positions. Each sample includes scene frames, velocities, grid-cell states,
    positions, headings, and shifted labels for multi-step future prediction.

    Args:
        video: Scene observations array [C, T].
        velocity: Linear velocity array [C, T].
        rot_velocity: Rotational velocity array [C, T].
        n_gridcells: Number of grid-cell units.
        gridcells_modules: Grid scales per module, or None for defaults.
        gridcells_orientations: Grid orientations per module, or None for defaults.
        gridcells_softmax: Whether to apply softmax to grid-cell outputs.
        positions: Position coordinates array [C, T].
        thetas: Heading angles array [C, T].
        window_size: Number of timesteps per window.
        n_future_pred: Number of future prediction steps (default: 1).
    """

    def __init__(self, video, velocity, rot_velocity,
                 n_gridcells, gridcells_modules, gridcells_orientations, gridcells_softmax,
                 positions, thetas, window_size, n_future_pred=1):
        self.scene_in = torch.from_numpy(video)
        self.scene_out = self.scene_in

        self.velocity = torch.from_numpy(velocity)
        self.rot_velocity = torch.from_numpy(rot_velocity)

        gc_params = {'n': n_gridcells, 'softmax': gridcells_softmax}
        if gridcells_modules is not None:
            gc_params['gridscale'] = tuple(gridcells_modules)
        if gridcells_orientations is not None:
            gc_params['orientation'] = tuple(gridcells_orientations)
        gc = GridCells(gc_params)
        self.grid_cells = torch.from_numpy(gc.get_state(positions).astype(np.float32))

        self.positions = torch.from_numpy(positions)
        self.thetas = torch.from_numpy(thetas)

        self.window_size = window_size
        self.n_future_pred = n_future_pred

    def __getitem__(self, index):
        """Return (inputs, vel, rot_vel, gc, pos, theta, labels) for the given window index."""
        if not (0 <= index < len(self)):
            raise ValueError("Index out of range")

        s0 = index * self.window_size
        e0 = s0 + self.window_size
        inputs = self.scene_in[:, s0:e0] if self.scene_in is not None else torch.Tensor([])

        vel, rot_vel, gc, pos, thet, label = [], [], [], [], [], []
        for f in range(self.n_future_pred):
            s, e = s0 + f, e0 + f
            vel.append(self.velocity[:, s:e])
            rot_vel.append(self.rot_velocity[:, s:e])
            gc.append(self.grid_cells[:, s:e])
            pos.append(self.positions[:, s:e])
            thet.append(self.thetas[:, s:e])

            label.append(self.scene_out[:, s+1:e+1])

        vel = torch.stack(vel, dim=1)
        rot_vel = torch.stack(rot_vel, dim=1)
        gc = torch.stack(gc, dim=1)
        pos = torch.stack(pos, dim=1)
        thet = torch.stack(thet, dim=1)
        label = torch.stack(label, dim=1)

        return inputs, vel, rot_vel, gc, pos, thet, label

    def __len__(self):
        """Number of non-overlapping windows available."""
        return self.positions.shape[1] // self.window_size - self.n_future_pred
