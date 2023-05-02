import sys
import jax
import inspect
import traceback
import functools
import contextlib
import jax.numpy as jnp
import numpy as np

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


class checked_function_args:
    """
    Context manager that activates the :py:func`check_output` decorator
    and captures the inputs of the decorated function. 

    The `checked_function_args` context manager is a debugging tool
    that identifies when one or more functions in a call stack are
    producing outputs with an undesired property (e.g. NaNs), and 
    what the inputs to those functions were.
    
    Examples
    --------
    Define a decorator called `nan_check` and use it to check for NaNs
    in the outputs of `func`. The inputs that caused `func` to produce
    NaNs are captured by the `checked_function_args` context manager.::

        >>> from jax_moseq.utils import check_for_nans
        >>> import jax.numpy as jnp
        >>> nan_check = check_output(check_for_nans, 'NaNs detected')
        >>> @nan_check
        ... def func(a, b):
        ...     return jnp.log(a), jnp.log(b)
        >>> with checked_function_args() as args:
        ...     func(1, 2)
        ...     func(0, 2)
        NaNs detected. Execution trace:
        File "<module>", line 81, in <module>
            func(0, 2)
        >>> print(args)
        {'func': ((0, 2), {})}

    When multiple decorated functions occur within the same call stack,
    the inputs to all of them are captured.::

        >>> @nan_check
        ... def func(a, b):
        ...     return jnp.log(a), jnp.log(b)
        >>> @nan_check
        ... def caller_of_func(a, b):
        ...     func(a, b)
        >>> with checked_function_args() as args:
        ...     caller_of_func(0, 2)
        NaNs detected. Execution trace:
        File "<module>", line 92, in <module>
            caller_of_func(0, 2)
        File "<module>", line 89, in caller_of_func
            func(a, b)
        >>> print(args)
        {'func': ((0, 2), {}), 'caller_of_func': ((0, 2), {})}
    """
    def __init__(self):
        self.inputs_dict = {}
        self.active = False
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.active = True
        sys._checked_function_args = self
        self.exit_stack.enter_context(jax.disable_jit())
        return self.inputs_dict

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.active = False
        del sys._checked_function_args
        self.exit_stack.close()

        if isinstance(exc_value, CheckOutputsError):
            print(exc_value, end='\n')  # Print the exception message

            # Extract the traceback and filter out frames from the 'wrapper'
            tb_frames = traceback.extract_tb(exc_traceback)
            filtered_frames = [frame for frame in tb_frames if frame.name != 'wrapper']

            # Format the filtered frames in a pretty way
            formatted_frames = traceback.format_list(filtered_frames)

            # Print the formatted frames
            for frame in formatted_frames:
                print(frame)

            return True  # Suppress the exception from propagating further
        

def check_output(checker, error_message):
    """
    Creates a decorator that applies `checker` to the outputs of a function.

    This decorator is intended to be used in conjunction with the
    :py:class:`checked_function_args` context manager, and is only
    active when the context manager is active. See 
    :py:class:`checked_function_args` for example usage.

    Parameters
    ----------
    checker : callable
        A function that takes the outputs of the decorated function and 
        returns a boolean value.

    error_message : str
        The error message to be displayed when raising a CheckOutputsError.

    Returns
    -------
    decorator : callable
        The generated decorator that checks the function output.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if not hasattr(sys, '_checked_function_args') or not sys._checked_function_args.active:
                    return result

                if checker(result):
                    sys._checked_function_args.inputs_dict[func.__name__] = (args, kwargs)
                    raise CheckOutputsError(error_message)
                return result
            
            except CheckOutputsError as e:
                if hasattr(sys, '_checked_function_args') and sys._checked_function_args.active:
                    sys._checked_function_args.inputs_dict[func.__name__] = (args, kwargs)
                raise e
        return wrapper
    return decorator

nan_check = check_output(check_for_nans, 'NaNs detected')