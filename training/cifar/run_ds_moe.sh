#!/bin/bash

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=1
# Size of expert parallel world (should be less than total world size)
EP_SIZE=1
# Number of total experts
EXPERTS=2
# --log-interval : 每训练100个mini-batch后记录一次训练日志
# --deepspeed : 使用DeepSpeed进行训练
# -- deepspeed_config : 指定了DeepSpeed的配置文件
# --moe : 这个参数指示使用混合专家模型。MoE模型是一种并行化策略，它可以将模型的某些部分（专家）并行运行在多个设备上，从而提高训练速度。
# --ep-world-size : 指定了专家并行世界的大小，即专家之间可以并行处理的任务数量。
# --num-experts : 定义了专家的数量。在这里，我们指定了两组专家，一个有2个专家，另一个有4个专家。
# --top-k : 选择k个最可能的专家进行处理。这是一种动态路由策略，它可以根据输入数据的特性，将任务分配给最可能处理好这个任务的专家。
# --noisy-gate-policy : 控制了如何在专家之间分配任务。在这里，我们使用"RSample"策略，即根据任务分配给专家的概率进行随机采样。
# --moe-param-group : 使得MoE参数在优化器中被单独分组，这可能有助于提高训练性能。
deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group
