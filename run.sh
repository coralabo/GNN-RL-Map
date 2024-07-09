#!/bin/bash
op=$1
if [ "$op" == "demo" ]; then
  python3 main.py --src_file_path=data/cholesky.txt --actor_lr=0.00003 --gcn_dims=128 \
  --max_iteration=300000 --warmup_iteration=100 --batch_size=24 --temperatre=1 --pea_width=4 --mii=2 \
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=4 --reward_mode=2
 
elif [ "$op" == "demo2" ]; then
  python3 main.py --src_file_path=data/gemm.txt --actor_lr=0.00001 --gcn_dims=128 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=24 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=1 --max_GRF=4 --max_memory=4 --reward_mode=4

elif [ "$op" == "atax_unroll" ]; then
  nohup python3 main.py --src_file_path=data/atax_unroll.txt --actor_lr=0.00005 --gcn_dims=128 \
   --max_iteration=200000 --warmup_iteration=100 --batch_size=32 --temperatre=10 --pea_width=6 --ii=2 \
   --beta=0.2 --layer_nums=5 --max_LRF=2 --max_GRF=2 --max_memory=4 --reward_mode=3  >> saving_log/atax_unroll.log 2>&1 & \


elif [ "$op" == "clean" ]; then
  rm nohup.out train_reward.log test_reward.log max_train_reward.log
  rm -rf saving_log/*
else
  echo "请输入 bash run.sh (想要测试的模型 || clean) 示例: bash run.sh demo"
fi
