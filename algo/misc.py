import numpy as np
import torch
import gymnasium as gym
from typing import Callable

# Linear schedule for use in learning rate schedule and domain randomization
def linear_schedule(initial_value: float,
                    final_value: float,
                    init_step: int = 0,
                    end_step: int = 2e7,
                    total_steps: int = None) -> Callable[[float], float]:
    """
    Linear learning rate schedule. 'anneal_percent' goes from 0 (beginning) to 1 (end).
    Param:
        initial_value: initial learning rate.
    Return:
        func: schedule that computes current learning rate depending on remaining progress
    Example:
        when initial_value = 1, final_value = 2, the diff is -1
        when anneal_percent = 10%, current_value = 1 - 0.1 * (-1) = 1.10
        when anneal_percent = 99%, current_value = 1 - 0.99 * (-1) = 1.99
    """

    def func(progress_remaining: float = None, elapsed_steps=None) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if elapsed_steps is None:
            elapsed_steps = total_steps * (1 - progress_remaining)
        if elapsed_steps < init_step:
            return initial_value
        anneal_percent = min(elapsed_steps / end_step, 1.0)
        return initial_value - anneal_percent * (initial_value - final_value)

    return func

# Data structure used as buffer (to store RGBD observations)
class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)