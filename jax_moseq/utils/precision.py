import functools

import jax

MATMUL_PRECISION = "float32"


def full_float32_matmuls(fn):
    """Decorator that runs ``fn`` with its matrix products at full float32
    precision.

    jax-moseq's float32 arithmetic (the Joseph-form Kalman steps, the AR
    refinement, the location and heading steps) assumes matrix products that
    round like float32. On GPUs with TensorFloat-32 (Ampere and newer), a
    float32 product at JAX's default precision keeps a 10-bit mantissa, with
    relative errors near 1e-3 instead of 6e-8. While ``fn`` runs,
    ``jax_default_matmul_precision`` is ``"float32"``, so every matrix
    product traced inside it, including those of the jitted functions it
    calls, runs at full precision whatever the global default is. Products
    that name their own precision keep it, and float64 products are
    unaffected.

    The setting is part of JAX's jit cache key, so a jitted function traced
    under another default outside ``fn`` is traced again inside it rather
    than reused. Applied between ``jax.jit`` and the function it compiles,
    the decorator sets the precision while the function is traced.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with jax.default_matmul_precision(MATMUL_PRECISION):
            return fn(*args, **kwargs)

    return wrapper
