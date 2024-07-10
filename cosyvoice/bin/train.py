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
import datetime
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from copy import deepcopy
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter,
    save_model,
    wrap_cuda_model,
    check_modify_and_save_config,
)


def get_args():
    parser = argparse.ArgumentParser(description="训练网络")
    parser.add_argument(
        "--train_engine",
        default="torch_ddp",
        choices=["torch_ddp", "deepspeed"],
        help="并行训练引擎",
    )
    parser.add_argument("--model", required=True, help="将要训练的模型")
    parser.add_argument("--config", required=True, help="配置文件")
    parser.add_argument("--train_data", required=True, help="训练数据文件")
    parser.add_argument("--cv_data", required=True, help="验证数据文件")
    parser.add_argument("--checkpoint", help="模型检查点")
    parser.add_argument("--model_dir", required=True, help="保存模型的目录")
    parser.add_argument(
        "--tensorboard_dir", default="tensorboard", help="Tensorboard 日志目录"
    )
    parser.add_argument(
        "--ddp.dist_backend",
        dest="dist_backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="分布式后端",
    )
    parser.add_argument(
        "--num_workers", default=0, type=int, help="读取数据的子进程数量"
    )
    parser.add_argument("--prefetch", default=100, type=int, help="预取数量")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="使用固定内存缓冲区进行读取",
    )
    parser.add_argument(
        "--deepspeed.save_states",
        dest="save_states",
        default="model_only",
        choices=["model_only", "model+optimizer"],
        help="保存模型/优化器状态",
    )
    parser.add_argument(
        "--timeout", default=30, type=int, help="cosyvoice_join 的超时时间（秒）"
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )

    # 忽略不需要的模型配置
    override_dict = {k: None for k in ["llm", "flow", "hift"] if k != args.model}
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs["train_conf"].update(vars(args))

    # 初始化分布式环境
    init_distributed(args)

    # 获取数据集和数据加载器
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = (
        init_dataset_and_dataloader(args, configs)
    )

    # 进行一些检查并保存配置到 args.model_dir
    configs = check_modify_and_save_config(args, configs)

    # 初始化 Tensorboard 摘要写入器
    writer = init_summarywriter(args)

    # 加载检查点
    model = configs[args.model]
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # 将模型从 CPU 移动到 GPU
    model = wrap_cuda_model(args, model)

    # 获取优化器和调度器
    model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)

    # 保存初始检查点
    info_dict = deepcopy(configs["train_conf"])
    save_model(model, "init", info_dict)

    # 获取执行器
    executor = Executor()

    # 开始训练循环
    for epoch in range(info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout)
        )
        executor.train_one_epoc(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            cv_data_loader,
            writer,
            info_dict,
            group_join,
        )
        dist.destroy_process_group(group_join)


if __name__ == "__main__":
    main()
