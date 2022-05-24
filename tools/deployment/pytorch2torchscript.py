import argparse
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from torch import nn

from mmdet.models import build_detector

torch.manual_seed(3)


def _demo_mm_inputs(input_shape: tuple, num_classes: int):
    """Create a superset of inputs needed to run test or train batches.
    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    #gt_labels = rng.randint(
    #    low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(False),
    #    'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2torchscript(model: nn.Module, input_shape: tuple, show: bool,
                        output_file: str, verify: bool):
    """Export Pytorch model to TorchScript model through torch.jit.trace and
    verify the outputs are same between Pytorch and TorchScript.
    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output
            TorchScript model.
        verify (bool): Whether compare the outputs between Pytorch
            and TorchScript through loading generated output_file.
    """
    #model.cpu().eval()

    #num_classes = model.head.num_classes
    num_classes = None
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)

    with torch.no_grad():

        print(type(model), model)

        trace_model = torch.jit.trace(model, img_list)
        save_dir, _ = osp.split(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        trace_model.save(output_file)
        print(f'Successfully exported TorchScript model: {output_file}')
    model.forward = origin_forward

    if verify:
        # load by torch.jit
        jit_model = torch.jit.load(output_file)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get jit output
        jit_result = jit_model(img_list[0])[0].detach().numpy()
        if not np.allclose(pytorch_result, jit_result):
            raise ValueError(
                'The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDet to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument(
        '--show', action='store_true', help='show TorchScript graph')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the TorchScript model',
        default=False)
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size (height, width)')
    args = parser.parse_args()
    return args

def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version

def check_torch_version():
    torch_minimum_version = '1.8.0'
    torch_version = digit_version(torch.__version__)

    assert (torch_version >= digit_version(torch_minimum_version)), \
        f'Torch=={torch.__version__} is not support for converting to ' \
        f'torchscript. Please install pytorch>={torch_minimum_version}.'

if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    sys.argv += ['/home/hsiehpinghan/git/mmdetection/configs/yolox/yolox_finetune_back_id_card_detect.py']
    sys.argv += ['--checkpoint', '/home/hsiehpinghan/git/mmdetection/checkpoint/best_bbox_mAP_epoch_9.pth']
    sys.argv += ['--show']
    sys.argv += ['--verify']
    sys.argv += ['--output-file', '/tmp/yolox_finetune_back_id_card_detect.pt']
    args = parse_args()

    check_torch_version()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    detector = build_detector(cfg.model,
                              train_cfg=None,
                              test_cfg=cfg.get('test_cfg'))

    if args.checkpoint:
        load_checkpoint(detector, args.checkpoint, map_location='cpu')

    # convert model to TorchScript file
    pytorch2torchscript(
        detector,
        input_shape,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)