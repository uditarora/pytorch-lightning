"""
Generates a summary of a model's layers and dimensionality
"""

import os
import subprocess
from collections import OrderedDict
from subprocess import PIPE
from typing import Tuple, Dict, Union, List, Any

import numpy as np
import torch
from torch.nn import Module

import pytorch_lightning as pl


class LayerSummary:
    """
    Summary class for a single layer in a :class:`~pytorch_lightning.core.lightning.LightningModule`.
    It collects the following information:

    - Type of the layer (e.g. Linear, BatchNorm1d, ...)
    - Input shape
    - Output shape
    - Number of parameters

    The input and output shapes are only known after the example input array was
    passed through the model.
    """

    def __init__(self, module: Module):
        super().__init__()
        self._module = module
        self._hook_handle = self._register_hook()
        self.in_size = None
        self.out_size = None

    def _register_hook(self):
        """
        Registers a hook on the module that computes the input- and output size(s)
        on the first forward pass. The hook will remove itself from the module, meaning that
        recursive models will only record their input- and output shapes once.
        """
        def hook(module, inp, out):
            if len(inp) == 1:
                inp = inp[0]
            self.in_size = parse_batch_shape(inp)
            self.out_size = parse_batch_shape(out)
            self._hook_handle.remove()  # hook detaches itself from module
        return self._module.register_forward_hook(hook)

    @property
    def layer_type(self) -> str:
        """ Returns the class name of the module. """
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """ Returns the number of parameters in this module. """
        return sum(np.prod(p.shape) for p in self._module.parameters())


class ModelSummary(object):

    def __init__(self, model: 'pl.LightningModule', mode: str = 'full'):
        """ Generates summaries of model layers and dimensions. """
        self._model = model
        self._mode = mode
        self._layer_summary = self.summarize()

    @property
    def named_modules(self) -> List[Tuple[str, Module]]:
        if self._mode == 'full':
            mods = self._model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        elif self._mode == 'top':
            # the children are the top-level modules
            mods = self._model.named_children()
        else:
            mods = []
        return list(mods)

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> List[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> List:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> List:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> List[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    def summarize(self) -> Dict[str, LayerSummary]:
        summary = OrderedDict()
        for name, module in self.named_modules:
            summary.update({name: LayerSummary(module)})

        if self._model.example_input_array is not None:
            self._forward_example_input()

        return summary

    def _forward_example_input(self) -> None:
        """ Run sample input through each layer to get output sizes. """

        input_ = self._model.example_input_array

        # TODO: should rethink this to add support for GPU, TPU, AMP, ... and avoid code duplication
        # or should it always be done on cpu?
        if self._model.on_gpu:
            device = next(self._model.parameters()).device
            # test if input is a list or a tuple
            if isinstance(input_, (list, tuple)):
                input_ = [input_i.to(device) if torch.is_tensor(input_i) else input_i
                          for input_i in input_]
            else:
                input_ = input_.to(device)

        # if model.trainer.use_amp and self.use_native_amp:
        #     model.forward = torch.cuda.amp.autocast()(model.forward)

        if self._model.trainer.use_amp:
            # test if it is not a list or a tuple
            if isinstance(input_, (list, tuple)):
                input_ = [input_i.half() if torch.is_tensor(input_i) else input_i
                          for input_i in input_]
            else:
                input_ = input_.half()

        with torch.no_grad():
            # let the model hooks collect the input- and output shapes
            if isinstance(input_, (list, tuple)):
                self._model(*input_)
            else:
                self._model(input_)

    def __str__(self):
        """
        Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes
        """
        arrays = [[' ', list(map(str, range(len(self._layer_summary))))],
                  ['Name', self.layer_names],
                  ['Type', self.layer_types],
                  ['Params', list(map(get_human_readable_count, self.param_nums))]]
        if self._model.example_input_array is not None:
            arrays.append(['In sizes', self.in_sizes])
            arrays.append(['Out sizes', self.out_sizes])

        return _format_summary_table(*arrays)

    def __repr__(self):
        return str(self)


def parse_batch_shape(batch: Any) -> List:
    if hasattr(batch, 'shape'):
        return list(batch.shape)

    if isinstance(batch, (list, tuple)):
        return [parse_batch_shape(el) for el in batch]

    # TODO: what do we show if type of input not recognized?
    return []


def _format_summary_table(*cols) -> str:
    """
    Takes in a number of arrays, each specifying a column in
    the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted.
    """
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1])
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = '{:<{}}'
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = ' | '.join(header) + '\n' + '-' * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += '\n' + ' | '.join(line)

    return summary


def get_memory_profile(mode: str) -> Union[Dict[str, int], Dict[int, int]]:
    """ Get a profile of the current memory usage.

    Args:
        mode: There are two modes:

            - 'all' means return memory for all gpus
            - 'min_max' means return memory for max and min

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
        If mode is 'min_max', the dictionary will also contain two additional keys:

        - 'min_gpu_mem': the minimum memory usage in MB
        - 'max_gpu_mem': the maximum memory usage in MB
    """
    memory_map = get_gpu_memory_map()

    if mode == 'min_max':
        min_index, min_memory = min(memory_map.items(), key=lambda item: item[1])
        max_index, max_memory = max(memory_map.items(), key=lambda item: item[1])

        memory_map = {'min_gpu_mem': min_memory, 'max_gpu_mem': max_memory}

    return memory_map


def get_gpu_memory_map() -> Dict[str, int]:
    """Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    """
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used',
            '--format=csv,nounits,noheader',
        ],
        encoding='utf-8',
        # capture_output=True,          # valid for python version >=3.7
        stdout=PIPE, stderr=PIPE,       # for backward compatibility with python version 3.6
        check=True)
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f'gpu_{index}': memory for index, memory in enumerate(gpu_memory)}
    return gpu_memory_map


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = [' ', 'K', 'M', 'B', 'T']
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    return f'{int(number):,d} {labels[index]}'
