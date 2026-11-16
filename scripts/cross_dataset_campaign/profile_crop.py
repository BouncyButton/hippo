"""Match a synthetic profiling target to the input patch, preserving its center."""


def center_crop_target(target, spatial_shape):
    """Crop the last three axes; reject shapes that would silently truncate."""
    source_shape = target.shape[-3:]
    if len(spatial_shape) != 3 or any(
        type(size) is not int or not 0 < size <= available
        for size, available in zip(spatial_shape, source_shape)
    ):
        raise ValueError(f"Invalid profile shape {spatial_shape} for target {source_shape}")
    slices = tuple(slice((available - size) // 2, (available - size) // 2 + size)
                   for size, available in zip(spatial_shape, source_shape))
    return target[(Ellipsis,) + slices].contiguous()
