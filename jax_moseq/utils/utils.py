import numpy as np
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from jax.scipy.linalg import cho_factor, cho_solve
from textwrap import fill
import functools
from math import ceil

_MIXED_MAP_ITERS = 1
_MIXED_MAP_GPUS = 1


def concatenate_stateseqs(stateseqs, mask=None):
    """
    Concatenate state sequences, optionally applying a mask.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    Returns
    -------
    stateseqs_flat: ndarray
        1d array containing all state sequences
    """
    if isinstance(stateseqs, dict):
        stateseq_flat = np.hstack(list(stateseqs.values()))
    elif isinstance(stateseqs, list):
        stateseq_flat = np.hstack(stateseqs)
    elif mask is not None:
        stateseq_flat = stateseqs[mask[:, -stateseqs.shape[1] :] > 0]
    else:
        stateseq_flat = stateseqs.flatten()
    return stateseq_flat


def get_durations(stateseqs, mask=None):
    """
    Get durations for a batch of state sequences.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    Returns
    -------
    durations: 1d array
        The duration of each each state (across all state sequences)

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])
    }
    >>> get_durations(stateseqs)
    array([2, 3, 1, 3, 1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)
    stateseq_padded = np.hstack([[-1], stateseq_flat, [-1]])
    changepoints = np.diff(stateseq_padded).nonzero()[0]
    return changepoints[1:] - changepoints[:-1]


def get_frequencies(stateseqs, mask=None, num_states=None, runlength=True):
    """
    Get state frequencies for a batch of state sequences.

    Parameters
    ----------
    stateseqs: ndarray of shape (..., t), or dict or list of such arrays
        Batch of state sequences where the last dim indexes time, or a
        dict/list containing state sequences as 1d arrays.

    mask: ndarray of shape (..., >=t), default=None
        Binary indicator for which elements of `stateseqs` are valid,
        used in the case where `stateseqs` is an ndarray. If `mask`
        contains more time-points than `stateseqs`, the initial extra
        time-points will be ignored.

    num_states: int, default=None
        Number of different states. If None, the number of states will
        be set to `max(stateseqs)+1`.

    runlength: bool, default=True
        Whether to count frequency by the number of instances of each
        state (True), or by the number of frames in each state (False).

    Returns
    -------
    frequencies: 1d array
        Frequency of each state across all state sequences

    Examples
    --------
    >>> stateseqs = {
        'name1': np.array([1, 1, 2, 2, 2, 3]),
        'name2': np.array([0, 0, 0, 1])}
    >>> get_frequencies(stateseqs, runlength=True)
    array([0.2, 0.4, 0.2, 0.2])
    >>> get_frequencies(stateseqs, runlength=False)
    array([0.3, 0.3, 0.3, 0.1])
    """
    stateseq_flat = concatenate_stateseqs(stateseqs, mask=mask).astype(int)

    if runlength:
        state_onsets = np.pad(np.diff(stateseq_flat).nonzero()[0] + 1, (1, 0))
        stateseq_flat = stateseq_flat[state_onsets]

    counts = np.bincount(stateseq_flat, minlength=num_states)
    frequencies = counts / counts.sum()
    return frequencies


def symmetrize(A):
    """Symmetrize a matrix."""
    return (A + A.swapaxes(-1, -2)) / 2


def psd_solve(A, B, diagonal_boost=1e-6):
    """
    Solves the linear system Ax=B, assuming A is positive semi-definite.

    Uses Cholesky decomposition for improved numerical stability and
    efficiency. A is symmetrized and diagonal elements are boosted by
    ``diagonal_boost`` to ensure positive definiteness.

    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix
    b: jax array, shape (...,n)

    Returns
    -------
    x: jax array, shape (...,n)
        Solution of the linear system Ax=b
    """
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), B)
    return x


def psd_inv(A, diagonal_boost=1e-6):
    """
    Invert a positive semi-definite matrix.

    Uses :py:func:`jax_moseq.utils.psd_solve` for numerical stability
    and ensures that the inverse matrix is symmetric.

    Parameters
    ----------
    A: jax array, shape (n,n)
        A positive semi-definite matrix

    Returns
    -------
    Ainv: jax array, shape (n,n)
        The inverse of A
    """
    Ainv = psd_solve(A, jnp.eye(A.shape[-1]), diagonal_boost=diagonal_boost)
    return symmetrize(Ainv)


def jax_io(fn):
    """
    Converts a function involving numpy arrays to one that inputs and
    outputs jax arrays.
    """
    return lambda *args, **kwargs: jax.device_put(
        fn(*jax.device_get(args), **jax.device_get(kwargs))
    )


def device_put_as_scalar(x):
    as_scalar = lambda arr: arr.item() if arr.shape == () else arr
    return jax.tree_map(as_scalar, jax.device_put(x))


def apply_affine(x, Ab):
    return jnp.einsum("...ij, ...j->...i", Ab, pad_affine(x))


def pad_affine(x):
    """
    Pad ``x`` with 1's so that it can be affine transformed with matrix
    multiplication.
    """
    padding = jnp.ones((*x.shape[:-1], 1))
    xpadded = jnp.concatenate((x, padding), axis=-1)
    return xpadded


def fit_pca(Y, mask, PCA_fitting_num_frames=1000000, verbose=False, **kwargs):
    """
    Fit a PCA model to transformed keypoint coordinates.

    Parameters
    ----------
    Y: jax array, shape (..., d)
        Keypoint coordinates
    mask: jax array
        Binary indicator for which elements of ``Y`` are valid
    PCA_fitting_num_frames: int, default=1000000
        Maximum number of frames to use for PCA. Frames will be sampled
        randomly if the input data exceed this size.
    verbose: bool, default=False
        Whether to print the number of sampled frames.
    Returns
    -------
    pca, sklearn.decomposition._pca.PCA
        An sklearn PCA model fit to Y
    """
    Y_flat = Y[mask > 0]

    N = Y_flat.shape[0]
    N_sample = min(PCA_fitting_num_frames, N)
    sample = np.random.choice(N, N_sample, replace=False)
    Y_sample = np.array(Y_flat)[sample]

    if verbose:
        print(f"PCA: Fitting PCA model to {N_sample} data points")
    pca = PCA().fit(Y_sample)
    return pca


def wrap_angle(x):
    """
    Wrap an angle to the range [-pi, pi].
    """
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def pad_along_axis(arr, pad_widths, axis=0, value=0):
    """
    Pad an array along a single axis

    Parameters
    -------
    arr: ndarray, Array to be padded
    pad_widths: tuple (int,int), Amount of padding on either end
    axis: int, Axis along which to add padding
    value: float, Value of padded array elements

    Returns
    _______
    padded_arr: ndarray
    """
    pad_left_shape = list(arr.shape)
    pad_right_shape = list(arr.shape)

    pad_left_shape[axis] = pad_widths[0]
    pad_right_shape[axis] = pad_widths[1]

    padding_left = jnp.ones(pad_left_shape, dtype=arr.dtype) * value
    padding_right = jnp.ones(pad_right_shape, dtype=arr.dtype) * value

    padded_arr = jnp.concatenate([padding_left, arr, padding_right], axis=axis)
    return padded_arr


def unbatch(data, keys, bounds):
    """
    Invert :py:func:`jax_moseq.utils.batch`

    Parameters
    ----------
    data: ndarray, shape (num_segs, seg_length, ...)
        Stack of segmented time-series

    keys: list or array of str, length num_segs
        Name of the time-series that each segment came from

    bounds: ndarray, shape (num_segs, 2)
        Start and end times for each segment, reflecting
        how the segments were extracted from the original
        time-series.

    Returns
    -------
    data_dict: dict
        Dictionary mapping names to reconstructed time-series
    """
    data_dict = {}
    for key in set(list(keys)):
        length = bounds[keys == key, 1].max()
        seq = np.zeros((int(length), *data.shape[2:]), dtype=data.dtype)
        for (s, e), d in zip(bounds[keys == key], data[keys == key]):
            seq[s:e] = d[: e - s]
        data_dict[key] = seq
    return data_dict


def batch(data_dict, keys=None, seg_length=None, seg_overlap=30):
    """
    Stack time-series data of different lengths into a single array for batch
    processing, optionally breaking up the data into fixed length segments. The
    data is padded so that the stacked array isn't ragged. The padding
    repeats the last frame of each time-series until the end of the segment.

    Parameters
    ----------
    data_dict: dict {str : ndarray}
        Dictionary mapping names to ndarrays, where the first dim
        represents time. All data arrays must have the same shape except
        for the first dim.

    keys: list of str, default=None
        Optional list of names specifying which datasets to include in
        the output and what order to put them in. Each name must be a
        key in ``data_dict``. If ``keys=None``, names will be sorted
        alphabetically.

    seg_length: int, default=None
        Break each time-series into segments of this length. If
        ``seg_length=None``, the final stacked array will be as long
        as the longest time-series.

    seg_overlap: int, default=30
        Amount of overlap between segments. For example, setting
        ``seg_length=N`` and ``seg_overlap=M`` will result in segments
        with start/end times (0, N+M), (N, 2*N+M), (2*N, 3*N+M),...

    Returns
    -------
    data: ndarray, shape (N, seg_length, ...)
        Stacked data array

    mask: ndarray, shape (N, seg_length)
        Binary indicator specifying which elements of ``data`` are not
        padding (``mask==0`` in padded locations)

    metadata: tuple (keys, bounds)
        Metadata for the rows of `data`, as a tuple with an array of keys
        and an array of (start,end) times.
    """
    if keys is None:
        keys = sorted(data_dict.keys())
    Ns = [len(data_dict[key]) for key in keys]
    if seg_length is None:
        seg_length = np.max(Ns)

    stack, mask, keys_out, bounds = [], [], [], []
    for key, N in zip(keys, Ns):
        for start in range(0, N, seg_length):
            arr = data_dict[key]
            end = min(start + seg_length + seg_overlap, N)
            pad_length = seg_length + seg_overlap - (end - start)
            padding = np.repeat(arr[end - 1 : end], pad_length, axis=0)
            mask.append(np.hstack([np.ones(end - start), np.zeros(pad_length)]))
            stack.append(np.concatenate([arr[start:end], padding], axis=0))
            keys_out.append(key)
            bounds.append((start, end))

    stack = np.stack(stack)
    mask = np.stack(mask)
    metadata = (np.array(keys_out), np.array(bounds))
    return stack, mask, metadata


def get_mixed_map_iters():
    """Get the number of iterations to use for jax.lax.map in
    :py:func:`jax_moseq.utils.mixed_map`."""
    return _MIXED_MAP_ITERS


def set_mixed_map_iters(iters):
    """Set the number of iterations to use for jax.lax.map in
    :py:func:`jax_moseq.utils.mixed_map`."""
    global _MIXED_MAP_ITERS
    _MIXED_MAP_ITERS = iters


def get_mixed_map_gpus():
    """Get the number of GPUs to use for jax.pmap in
    :py:func:`jax_moseq.utils.mixed_map`."""
    return _MIXED_MAP_GPUS


def set_mixed_map_gpus(gpus):
    """Set the number of GPUs to use for jax.pmap in
    :py:func:`jax_moseq.utils.mixed_map`."""
    global _MIXED_MAP_GPUS
    _MIXED_MAP_GPUS = gpus


def _reshape_args(args, axes):
    """Reshape args to (pmap dim, lax.map dim, vmap dim, [other dims])"""
    n_iters = get_mixed_map_iters()
    n_gpus = get_mixed_map_gpus()
    axis_size = args[0].shape[axes[0]]

    vmap_size = ceil(axis_size / n_iters / n_gpus)
    pmap_size = ceil(axis_size / vmap_size / n_iters)
    lmap_size = ceil(axis_size / vmap_size / pmap_size)
    padding = vmap_size * pmap_size * lmap_size - axis_size

    def _reshape(a, axis):
        if axis > 0:
            a = jnp.moveaxis(a, axis, 0)
        if padding > 0:
            padding_array = jnp.zeros((padding, *a.shape[1:]), dtype=a.dtype)
            a = jnp.concatenate((a, padding_array))
        return a.reshape(lmap_size, pmap_size, vmap_size, *a.shape[1:])

    args = [_reshape(arg, axis) for arg, axis in zip(args, axes)]
    return args, axis_size


def _reshape_outputs(outputs, axes, axis_size):
    """Reshape outputs from (lax.map dim, vmap dim, [other dims])"""

    def _reshape(a, axis):
        a = a.reshape(-1, *a.shape[3:])[:axis_size]
        if axis > 0:
            a = jnp.moveaxis(a, 0, axis)
        return a

    outputs = [_reshape(out, axis) for out, axis in zip(outputs, axes)]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def _partial(fun, other_args, mapped_argnums, other_argnums):
    def partial_fun(mapped_args):
        args = {}
        for i, arg in zip(mapped_argnums, mapped_args):
            args[i] = arg
        for i, arg in zip(other_argnums, other_args):
            args[i] = arg
        args = [args[i] for i in range(len(args))]
        return fun(*args)

    return partial_fun


def _sort_args(args, in_axes):
    """Sort arguments into mapped and unmapped arguments."""
    mapped_args, mapped_argnums = [], []
    other_args, other_argnums = [], []
    for i, (arg, axis) in enumerate(zip(args, in_axes)):
        if axis is not None:
            mapped_args.append(arg)
            mapped_argnums.append(i)
        else:
            other_args.append(arg)
            other_argnums.append(i)
    return mapped_args, mapped_argnums, other_args, other_argnums


def mixed_map(fun, in_axes=None, out_axes=None):
    """
    Combine jax.pmap, jax.vmap and jax.lax.map for parallelization.

    This function is similar to `jax.vmap`, except that it mixes together
    `jax.pmap`, `jax.vmap` and `jax.lax.map` to prevent OOM errors and allow
    for parallelization across multiple GPUs. The behavior is determined by
    the global variables `_MIXED_MAP_ITERS` and `_MIXED_MAP_GPUS`, which can be
    set using :py:func:`jax_moseq.utils.set_mixed_map_iters` and
    py:func:`jax_moseq.utils.set_mixed_map_gpus` respectively.

    Given an axis size of N to map, the data is padded such that the axis size
    is a multiple of the number of `_MIXED_MAP_ITERS * _MIXED_MAP_GPUS`. The
    data is then processed serially chunks, where the number of chunks is
    determined by `_MIXED_MAP_ITERS`. Each chunk is processed in parallel
    using jax.pmap to distribute across `_MIXED_MAP_GPUS` devices and jax.vmap
    within each device.
    """

    @functools.wraps(fun)
    def mixed_map_f(*args):
        nonlocal in_axes
        nonlocal out_axes

        if in_axes is None:
            in_axes = tuple([0] * len(args))
        else:
            assert len(in_axes) == len(
                args
            ), "`in_axes` should be a tuple with the same length as the number of arguments"

        mapped_args, mapped_argnums, other_args, other_argnums = _sort_args(
            args, in_axes
        )
        mapped_args, axis_size = _reshape_args(
            mapped_args, [in_axes[i] for i in mapped_argnums]
        )
        f = _partial(fun, other_args, mapped_argnums, other_argnums)
        outputs = jax.lax.map(jax.pmap(jax.vmap(f)), mapped_args)

        if not isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = (outputs,)
        if out_axes is None:
            out_axes = tuple([0] * len(outputs))
        else:
            assert len(out_axes) == len(
                outputs
            ), "`out_axes` should be a tuple with the same length as the number of function outputs"

        outputs = _reshape_outputs(outputs, out_axes, axis_size)
        return outputs

    return mixed_map_f
