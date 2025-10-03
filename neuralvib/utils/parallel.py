"""Parallel functionalities"""

from typing import Callable

import joblib


def joblib_parallel(fun: Callable):
    """Joblib Parallel Wrapper

    Args:
        fun: the function to be paralleled

    Returns:
        _parallel_fun: the joblib paralleled
            function.
    """

    def _paralleled_func(batched_input):
        """The joblib paralleled function
        Return paralleled execution on batched_input
        """
        batch_out = joblib.Parallel(
            n_jobs=-1,
        )(joblib.delayed(fun)(single_input) for single_input in batched_input)
        return batch_out

    return _paralleled_func
