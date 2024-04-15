#!/bin/bash

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=2
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts, note here we need to pass >= two numbers (numbers can be different)
EXPERTS="2 4"
# 下面参数说明：
# --moe : 启用混合专家（MoE）
# --ep-world-size : 指定了专家并行世界的大小，即专家之间可以并行处理的任务数量。
# --num-experts : 定义了专家的数量。在这里，我们指定了两组专家，一个有2个专家，另一个有4个专家。
# --top-k : 控制了选择最佳专家的策略
# --mlp-type : 控制了模型中多层感知机（MLP）的类型，这里使用的是残差类型（residual）。
# --noisy-gate-policy : 控制了如何在专家之间分配任务。在这里，我们使用"RSample"策略，即根据任务分配给专家的概率进行随机采样。这种策略可以增加模型的多样性和健壮性。
# --moe-param-group : 使得MoE参数在优化器中被单独分组，这可能有助于提高训练性能。
deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--mlp-type 'residual' \
	--noisy-gate-policy 'RSample' \
	--moe-param-group
