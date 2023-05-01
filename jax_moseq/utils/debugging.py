import sys
import jax
import inspect
import traceback
import functools
import contextlib
import jax.numpy as jnp

# JAX has its own implementation of `tree_flatten_with_path` 
# but it's only in version of 0.4.6 or later, and currently 
# some platforms (e.g. Windows) require installing < 0.4.6
from optree import tree_flatten_with_path



def _check_array_precision(arg, x64):
    """Checks if precision of `arg` matches `x64`"""
    permitted_dtypes = [
        (np.int64 if x64 else np.int32),
        (jnp.int64 if x64 else jnp.int32),
        (np.float64 if x64 else np.float32),
        (jnp.float64 if x64 else jnp.float32)]
    if isinstance(arg, jnp.ndarray) or isinstance(arg, np.ndarray):
        if not arg.dtype in permitted_dtypes:  return False   
    return True

def check_precision(fn):
    """
    Decorator to check that the precision of the arguments matches the
    precision of the jax configuration.
    """
    arg_names = inspect.getfullargspec(fn).args
   
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        x64 = jax.config.x64_enabled
        args_with_wrong_precision = []
        for name,arg in list(zip(arg_names,args)) + list(kwargs.items()):
            check_fn = functools.partial(_check_array_precision, x64=x64)
            if not jax.tree_util.tree_all(jax.tree_map(check_fn, arg)):
                args_with_wrong_precision.append(name)
                
        if len(args_with_wrong_precision) > 0:
            msg = f'JAX is configured to use {"64" if x64 else "32"}-bit precision, '
            msg += f'but following arguments contain {"32" if x64 else "64"}-bit arrays: '
            msg += ', '.join([f'"{name}"' for name in args_with_wrong_precision])
            msg += '. Either change the JAX config using `jax.config.update("jax_enable_x64", True/False)` '
            msg += 'or convert the arguments to the correct precision using `jax_moseq.utils.utils.convert_data_precision`.'
            raise ValueError(msg)

        return fn(*args, **kwargs)
    return wrapper


def convert_data_precision(data, x64=None):
    """
    Convert all numerical data in a pytree to the specified precision.
    
    Note that converting to 64-bit precision is only possible if 
    ``jax.config.x64_enabled`` is ``True``. To update this setting, use
    ``jax.config.update('jax_enable_x64', True)``.

    Parameters
    ----------
    data: pytree (dict, list, tuple, array, or any nested combination thereof)
        The data to convert.
    x64: bool, default=None
        If ``x64=True``, convert to 64-bit precision. If ``x64=False``,
        convert to 32-bit precision. If ``x64=None``, infer the desired
        precision from ``jax.config.x64_enabled``. 

    Returns
    -------
    data: pytree
        The converted data.
    """
    if x64 is None: x64 = jax.config.x64_enabled
    elif x64==True and not jax.config.x64_enabled:
        raise ValueError(
            'Cannot convert to 64-bit precision because jax.config.x64_enabled==False '
            'Use jax.config.update("jax_enable_x64", True) to enable 64-bit precision.')
    
    def convert(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.int64 if x64 else jnp.int32)
        elif jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64 if x64 else jnp.float32)
    
    return jax.tree_map(convert, data)


def check_for_nans(data):
    """
    Check for NaNs in all arrays of a pytree.

    Parameters
    ----------
    data: pytree (dict, list, tuple, array, or any nested combination thereof)
        The data to check for NaNs in.
    
    Returns
    -------
    any_nans: bool
        Whether any of the arrays in ``data`` contain a NaN.

    nan_info: list of tuples
        List of arrays containing a NaN, in the form of pairs
        ``(path, number_of_nans)`` where ``path`` is a sequence of
        keys that define the location of the array in the pytree.

    messages: list of str
        List of messages; one for each elements of ``nan_info``.
    """

    def _format(path, num_nan):
        path = '/'.join(map(str,path))
        msg  = f"{num_nan} NaNs found in {path}"
        return msg
        
    nan_info = []
    messages = []
    for path,value in zip(*tree_flatten_with_path(data)[:2]):
        if isinstance(value, jnp.ndarray):
            if jnp.isnan(value).any():
                num_nans = jnp.isnan(value).sum().item()
                nan_info.append((path, num_nans))
                messages.append(_format(path, num_nans))
    
    any_nans = len(nan_info)>0
    return any_nans, nan_info, messages


class CheckOutputsError(Exception):
    pass


class checked_function_inputs:
    def __init__(self):
        self.inputs_dict = {}
        self.active = False
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.active = True
        sys._checked_function_inputs = self
        self.exit_stack.enter_context(jax.disable_jit())
        return self.inputs_dict

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.active = False
        del sys._checked_function_inputs
        self.exit_stack.close()

        if isinstance(exc_value, CheckOutputsError):
            print(exc_value)  # Print the exception message

            # Extract the traceback and filter out frames from the 'wrapper'
            tb_frames = traceback.extract_tb(exc_traceback)
            filtered_frames = [frame for frame in tb_frames if frame.name != 'wrapper']

            # Format the filtered frames in a pretty way
            formatted_frames = traceback.format_list(filtered_frames)

            # Print the formatted frames
            for frame in formatted_frames:
                print(frame)

            return True  # Suppress the exception from propagating further
        

def check_output_decorator(check_outputs, message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if not hasattr(sys, '_checked_function_inputs') or not sys._checked_function_inputs.active:
                    return result

                if check_outputs(*result):
                    sys._checked_function_inputs.inputs_dict[func.__name__] = (args, kwargs)
                    raise CheckOutputsError("Stopping execution due to check_outputs condition being True.")
                return result
            
            except CheckOutputsError as e:
                if hasattr(sys, '_checked_function_inputs') and sys._checked_function_inputs.active:
                    sys._checked_function_inputs.inputs_dict[func.__name__] = (args, kwargs)
                raise e
        return wrapper
    return decorator
