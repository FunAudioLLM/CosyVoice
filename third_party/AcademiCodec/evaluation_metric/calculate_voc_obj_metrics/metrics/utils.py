import os
import random
from os.path import join as opj

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_params(args):

    params = {}
    args_ref = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params

    return params


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def write_result(estimate, noise, file, args):
    if not os.path.exists(args.enhanced_path):
        os.makedirs(args.enhanced_path)
    file_name = opj(args.enhanced_path,
                    file[0].rsplit('.', 1)[0].replace('\\', '/').split('/')[-1])
    noise_path = file_name + '_noise.wav'
    enhanced_path = file_name + '_enhanced.wav'

    torchaudio.save(noise_path, noise.squeeze(1), args.sample_rate)
    torchaudio.save(enhanced_path, estimate.squeeze(1), args.sample_rate)


def seed_init(seed=100):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def args_dict(args):
    """
    Get your arguments and make dictionary.
    If you add some arguments in the model, you should edit here also.
    """
    args.dataset = {
        'train': args.train,
        'val': args.val,
        'test': args.test,
        'matching': args.matching
    }
    args.setting = {
        'sample_rate': args.sample_rate,
        'segment': args.segment,
        'pad': args.pad,
        'stride': args.set_stride
    }
    args.manner = {
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'hidden': args.hidden,
        'depth': args.depth,
        'kernel_size': args.kernel_size,
        'stride': args.stride,
        'growth': args.growth,
        'head': args.head,
        'segment_len': args.segment_len
    }

    args.ex_name = os.getcwd().replace('\\', '/').split('/')[-1]

    return args
