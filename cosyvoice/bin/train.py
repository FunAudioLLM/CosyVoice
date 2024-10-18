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

from __future__ import print_function
import argparse
import logging
import os
import torch
import torch.distributed as dist
import deepspeed
from hyperpyyaml import load_hyperpyyaml
from copy import deepcopy
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config
)
from torch.distributed.elastic.multiprocessing.errors import record

def get_args():
    parser = argparse.ArgumentParser(description='Training your network')
    parser.add_argument('--train_engine', default='torch_ddp', choices=['torch_ddp', 'deepspeed'], help='Engine for parallelized training')
    parser.add_argument('--model', required=True, help='Model to be trained')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--train_data', required=True, help='Training data file')
    parser.add_argument('--cv_data', required=True, help='CV data file')
    parser.add_argument('--checkpoint', help='Checkpoint model path')
    parser.add_argument('--model_dir', required=True, help='Directory to save the model')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='Tensorboard log directory')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl', choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of subprocess workers for reading')
    parser.add_argument('--prefetch', default=100, type=int, help='Prefetch number')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory buffers for reading')
    parser.add_argument('--deepspeed.save_states', dest='save_states', default='model_only', choices=['model_only', 'model+optimizer'], help='Save model/optimizer states')
    parser.add_argument('--timeout', default=60, type=int, help='Timeout (in seconds) for cosyvoice_join')
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    gan = True if args.model == 'hifigan' else False
    override_dict = {k: None for k in ['llm', 'flow', 'hift', 'hifigan'] if k != args.model}
    if gan: override_dict.pop('hift')

    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    
    if gan:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    init_distributed(args)

    train_dataset, cv_dataset, train_data_loader, cv_data_loader = init_dataset_and_dataloader(args, configs, gan)
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    model = configs[args.model]
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=False)
    
    model = wrap_cuda_model(args, model)
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)

    info_dict = deepcopy(configs['train_conf'])
    save_model(model, 'init', info_dict)

    executor = Executor(gan=gan)

    for epoch in range(info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        
        if gan:
            executor.train_one_epoc_gan(model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader, writer, info_dict, group_join)
        else:
            executor.train_one_epoc(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, group_join)
        
        dist.destroy_process_group(group_join)

if __name__ == '__main__':
    main()
