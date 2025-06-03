import os
import time
from contextlib import contextmanager
from functools import wraps

from loguru import logger

# Get profile enabled flag from environment variable, default to False if not set
PROFILING_ENABLED = os.environ.get("ENABLE_PROFILING", "False").lower() in (
    "true",
    "1",
    "yes",
)


def profile(label: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROFILING_ENABLED:
                return func(*args, **kwargs)

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if label:
                logger.debug(f"> {label} - {func.__qualname__}: {elapsed:.4f}s")
            else:
                print_name = getattr(func.__self__, "get_name", lambda: None)()
                if print_name is None:
                    print_name = getattr(func, "__qualname__", func.__name__)
                logger.debug(f"> {print_name}: {elapsed:.4f}s")
            return result

        return wrapper

    return decorator


@contextmanager
def profile_block(label):
    if not PROFILING_ENABLED:
        yield
        return

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.debug(f">> {label}: {elapsed:.4f}s")


def enable_profiling(enable=True):
    """
    Programmatically enable or disable profiling during runtime.
    This will affect all subsequent profiling decorators.
    """
    global PROFILING_ENABLED
    PROFILING_ENABLED = enable
    return PROFILING_ENABLED


def is_profiling_enabled():
    """
    Returns whether profiling is currently enabled.
    """
    return PROFILING_ENABLED
