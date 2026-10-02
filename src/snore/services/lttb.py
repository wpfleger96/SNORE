"""Largest Triangle Three Buckets (LTTB) downsampling algorithm.

Downsamples time-series data while preserving visual shape. Based on the 2013
paper by Sveinn Steinarsson: "Downsampling Time Series for Visual Representation"
"""

import numpy as np

__all__ = ["lttb_downsample"]


def lttb_downsample(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample time-series data using Largest Triangle Three Buckets algorithm.

    LTTB preserves the visual shape of data by selecting points that form the
    largest triangles with neighboring points, maintaining peaks, valleys, and
    overall structure even at aggressive downsampling ratios.

    Args:
        timestamps: Array of timestamps (x-axis values)
        values: Array of data values (y-axis values)
        target_points: Desired number of points after downsampling

    Returns:
        Tuple of (downsampled_timestamps, downsampled_values)

    Raises:
        ValueError: If timestamps and values have different lengths
        ValueError: If target_points < 2

    Examples:
        >>> t = np.array([0, 1, 2, 3, 4, 5])
        >>> v = np.array([0, 10, 5, 15, 8, 12])
        >>> t_down, v_down = lttb_downsample(t, v, target_points=3)
        >>> len(t_down)
        3
    """
    if len(timestamps) != len(values):
        raise ValueError(
            f"timestamps ({len(timestamps)}) and values ({len(values)}) "
            "must have same length"
        )

    if target_points < 2:
        raise ValueError("target_points must be at least 2")

    if len(timestamps) <= target_points:
        return timestamps.copy(), values.copy()

    if target_points == 2:
        return timestamps[[0, -1]], values[[0, -1]]

    sampled_indices = np.zeros(target_points, dtype=int)
    sampled_indices[0] = 0
    sampled_indices[-1] = len(timestamps) - 1

    bucket_size = (len(timestamps) - 2) / (target_points - 2)

    a = 0

    for i in range(1, target_points - 1):
        avg_range_start = int(np.floor((i + 1) * bucket_size)) + 1
        avg_range_end = int(np.floor((i + 2) * bucket_size)) + 1
        avg_range_end = min(avg_range_end, len(timestamps) - 1)

        if avg_range_start >= len(timestamps):
            avg_range_start = len(timestamps) - 1
        if avg_range_end <= avg_range_start:
            avg_range_end = min(avg_range_start + 1, len(timestamps) - 1)

        avg_slice_t = timestamps[avg_range_start:avg_range_end]
        avg_slice_v = values[avg_range_start:avg_range_end]
        if len(avg_slice_t) == 0:
            avg_x = timestamps[avg_range_start]
            avg_y = values[avg_range_start]
        else:
            avg_x = np.mean(avg_slice_t)
            avg_y = np.mean(avg_slice_v)

        range_start = int(np.floor(i * bucket_size)) + 1
        range_end = int(np.floor((i + 1) * bucket_size)) + 1
        range_end = min(range_end, len(timestamps))

        point_a_x = timestamps[a]
        point_a_y = values[a]

        max_area = -1.0
        max_area_point = range_start

        for j in range(range_start, range_end):
            area = abs(
                (point_a_x - avg_x) * (values[j] - point_a_y)
                - (point_a_x - timestamps[j]) * (avg_y - point_a_y)
            )

            if area > max_area:
                max_area = area
                max_area_point = j

        sampled_indices[i] = max_area_point
        a = max_area_point

    return timestamps[sampled_indices], values[sampled_indices]
