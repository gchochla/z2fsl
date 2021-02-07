"""General utilities."""

import argparse
import os
import random

import numpy as np
import torch


def str2bool(arg):
    """CL bool arguments to bools"""
    if arg.lower() == 'true':
        return True
    if arg.lower() == 'false':
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_parameter_requires_grad(model, requires_grad=False):
    """Sets requires_grad for all the parameters in a model.

    Args:
        model (`nn model`): model to alter.
        requires_grad (`bool`): whether the model
            requires grad.
    """
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def is_grayscale(image):
    """Return if image is grayscale.

    Assert if image only has 1 channel.

    Args:
        image (`PIL.Image`): image to check.

    Returns:
        bool indicating whether image is grayscale.
    """

    try:
        # channel==1 is 2nd channel
        image.getchannel(1)
        return False
    except ValueError:
        return True


def configuration_filename(feature_dir, proposed_splits, split, generalized):
    """Calculates configuration specific filenames.

    Args:
        feature_dir (`str`): directory of features wrt
            to dataset directory.
        proposed_splits (`bool`): whether using proposed splits.
        split (`str`): train split.
        generalized (`bool`): whether GZSL setting.

    Returns:
        `str` containing arguments in appropriate form.
    """
    return '{}{}_{}{}.pt'.format(
        feature_dir,
        ('_proposed_splits' if proposed_splits else ''),
        split,
        '_generalized' if generalized else '',
    )


def slice_fn(tuple_slice):
    """Makes a function that slices a list.

    Args:
        tuple_slice (`tuple`): tuple of 2 floats denoting
            the percentage at which to split the argument
            of the function.

    Returns:
        A function that slices an iterable.
    """
    return lambda l: l[slice(int(len(l) * tuple_slice[0]), int(len(l) * tuple_slice[1]))]


def dataset_name(dataset_dir):
    """Gets standard dataset name from dataset directory.

    Args:
        dataset_dir (`str`): absolute dataset directory.

    Returns:
        `'cub'`, `'awa2'` or `'sun'`.
    """

    benchmark = os.path.split(dataset_dir)[1]
    if not benchmark:
        benchmark = os.path.split(dataset_dir[:-1])[1]
    benchmark = benchmark.lower()
    if 'cub' in benchmark:
        return 'cub'
    elif 'awa2' in benchmark or 'attributes2' in benchmark:
        return 'awa2'
    elif 'sun' in benchmark:
        return 'sun'
    else:
        ValueError('Dataset name was not recognised')


def manual_seed(seed):
    """Sets all possible seeds and variables
    for reproducibility in PyTorch.

    Args:
        seed (`int`): seed to use.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
