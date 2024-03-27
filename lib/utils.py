from dataclasses import dataclass
import json
import numpy as np


@dataclass
class Config:
    PLOT_RANGE_IN_METER: int = 5
    RADAR_HEIGHT_IN_METER: float = 1.83


def default_kwargs(**default_kwargs_decorator):
    def actual_decorator(fn):
        @functools.wraps(fn)
        def g(*args, **kwargs):
            default_kwargs_decorator.update(kwargs)
            return fn(*args, **default_kwargs_decorator)

        return g

    return actual_decorator


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)
