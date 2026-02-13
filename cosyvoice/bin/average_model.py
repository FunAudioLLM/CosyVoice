# Copyright (c) 2020 Mobvoi Inc (Di Wu)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import glob

import yaml
import torch
from hyperpyyaml import load_hyperpyyaml


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    val_scores = []
    if args.val_best:
        yamls = glob.glob('{}/*.yaml'.format(args.src_path))
        yamls = [
            f for f in yamls
            if os.path.basename(f).startswith('epoch_')
        ]
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = load_hyperpyyaml(f)
                if dic_yaml is None or 'loss_dict' not in dic_yaml:
                    continue
                loss = float(dic_yaml['loss_dict']['loss'])
                epoch = int(dic_yaml['epoch'])
                step = int(dic_yaml['step'])
                tag = dic_yaml.get('tag', 'unknown')
                val_scores += [[epoch, step, loss, tag]]
        sorted_val_scores = sorted(val_scores,
                                   key=lambda x: x[2],
                                   reverse=False)
        print("best val (epoch, step, loss, tag) = " +
              str(sorted_val_scores[:args.num]))
        path_list = [
            args.src_path + '/epoch_{}_whole.pt'.format(score[0])
            for score in sorted_val_scores[:args.num]
        ]
    else:
        # Default behavior: take the last N checkpoints if val_best is not specified
        path_list = glob.glob('{}/*_whole.pt'.format(args.src_path))
        path_list = sorted(path_list, key=os.path.getmtime, reverse=True)[:args.num]

    if not path_list:
        print("Error: No models found to average in {}".format(args.src_path))
        return

    print(path_list)
    avg = {}
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # If the checkpoint is a dict containing 'model', use checkpoint['model']
        states = checkpoint.get('model', checkpoint)
        
        for k, v in states.items():
            if k not in ['step', 'epoch']:
                if k not in avg:
                    if isinstance(v, torch.Tensor):
                        avg[k] = v.clone()
                    else:
                        avg[k] = v
                else:
                    if isinstance(v, torch.Tensor):
                        avg[k] += v

    # average
    for k in avg.keys():
        if avg[k] is not None and isinstance(avg[k], torch.Tensor):
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)


if __name__ == '__main__':
    main()
