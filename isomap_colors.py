"""Position-based colours shared by the Isomap plotting scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Vivid colours for (lower-left, lower-right, upper-left, upper-right).
# Intermediate positions are bilinear mixtures.
CORNER_COLORS = np.asarray(
    [
        (0.000, 0.200, 1.000),  # vivid blue
        (1.000, 0.100, 0.000),  # vivid red
        (0.000, 0.800, 0.200),  # vivid green
        (1.000, 0.850, 0.000),  # vivid yellow
    ],
    dtype=float,
)


def xy_bounds(positions: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    """Return finite x/y limits spanning one or more position arrays."""
    if not positions:
        raise ValueError("At least one position array is required")

    xy_arrays = []
    for position in positions:
        position = np.asarray(position)
        if position.ndim != 2 or position.shape[1] < 2:
            raise ValueError("Positions must have shape (samples, >=2)")
        if len(position) == 0:
            raise ValueError("Position arrays must not be empty")
        xy = position[:, :2]
        if not np.all(np.isfinite(xy)):
            raise ValueError("Positions contain NaN or infinity")
        xy_arrays.append(xy)

    return (
        min(float(xy[:, 0].min()) for xy in xy_arrays),
        max(float(xy[:, 0].max()) for xy in xy_arrays),
        min(float(xy[:, 1].min()) for xy in xy_arrays),
        max(float(xy[:, 1].max()) for xy in xy_arrays),
    )


def position_corner_colors(
    positions: np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Map x/y positions to bilinear mixtures of four corner colours.

    ``bounds`` is ``(x_min, x_max, y_min, y_max)``. Supplying shared bounds
    makes colours directly comparable across multiple embeddings.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("Positions must have shape (samples, >=2)")
    if bounds is None:
        bounds = xy_bounds([positions])

    x_min, x_max, y_min, y_max = bounds
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Position bounds must be finite")
    if x_max < x_min or y_max < y_min:
        raise ValueError("Position bounds must be ordered min to max")

    x_scale = x_max - x_min
    y_scale = y_max - y_min
    x = (
        np.full(len(positions), 0.5)
        if x_scale == 0
        else np.clip((positions[:, 0] - x_min) / x_scale, 0.0, 1.0)
    )
    y = (
        np.full(len(positions), 0.5)
        if y_scale == 0
        else np.clip((positions[:, 1] - y_min) / y_scale, 0.0, 1.0)
    )
    weights = np.column_stack(
        ((1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y)
    )
    return np.clip(weights @ CORNER_COLORS, 0.0, 1.0)


def plot_xy_color_reference(axis, bounds, resolution: int = 128) -> None:
    """Draw the four-corner colour field used by the latent-space plots."""
    x_min, x_max, y_min, y_max = bounds
    # imshow cannot display a zero-width extent, so only its display extent is
    # expanded; colour calculation continues to use the true data bounds.
    x_display = (x_min, x_max if x_max > x_min else x_min + 1.0)
    y_display = (y_min, y_max if y_max > y_min else y_min + 1.0)
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_positions = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    image = position_corner_colors(grid_positions, bounds).reshape(
        resolution, resolution, 3
    )
    axis.imshow(
        image,
        origin="lower",
        extent=(*x_display, *y_display),
        aspect="equal",
        interpolation="bilinear",
    )
    axis.set_xlim(*x_display)
    axis.set_ylim(*y_display)
    corner_labels = (
        (0.02, 0.02, "x low, y low", "white", "left", "bottom"),
        (0.98, 0.02, "x high, y low", "white", "right", "bottom"),
        (0.02, 0.98, "x low, y high", "black", "left", "top"),
        (0.98, 0.98, "x high, y high", "black", "right", "top"),
    )
    for x, y, label, color, horizontal, vertical in corner_labels:
        axis.text(
            x, y, label, color=color, fontsize=8, fontweight="bold",
            ha=horizontal, va=vertical, transform=axis.transAxes,
        )
    axis.set_xlabel("X position")
    axis.set_ylabel("Y position")
    axis.set_title("Physical X-Y colour key")
