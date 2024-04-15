#!/bin/bash

# deepspeed : 用于启动和管理DeepSpeed训练任务
# cifar10_deepspeed.py : 将要运行的Python脚本
# --deepspeed : 使用DeepSpeed进行训练
# --deepspeed_config : 指定DeepSpeed的配置文件
# $@ : 表示把所有的命令行参数传递给下一个命令
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@
