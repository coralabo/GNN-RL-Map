#!/bin/bash
op=$1
if [ "$op" == "demo2" ]; then
  nohup python3 main.py --src_file_path=data/o29m.txt --actor_lr=0.00003 --gcn_dims=128 \
  --max_iteration=300000 --warmup_iteration=100 --batch_size=24 --temperatre=1 --pea_width=3 --mii=2 \
  --beta=0 --layer_nums=6 --max_LRF=4 --max_GRF=4 --max_memory=4 --reward_mode=2 >saving_log/o29_6_6_ii_2_lr_3e5_256_3.log 2>&1 & 
 
elif [ "$op" == "demo" ]; then
  python3 main.py --src_file_path=data/o11m.txt --actor_lr=0.00001 --gcn_dims=128 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=24 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=1 --max_GRF=4 --max_memory=4 --reward_mode=4

elif [ "$op" == "o10" ]; then
  nohup python3 main.py --src_file_path=data/o10.txt --actor_lr=0.0001 --gcn_dims=128 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=32 --temperatre=10 --pea_width=4 --ii=2 \
  --beta=0 --c=5 \

elif [ "$op" == "o14" ]; then
  nohup python3 main.py --src_file_path=data/o14.txt --actor_lr=0.00005 --gcn_dims=128 \
   --max_iteration=2000 --warmup_iteration=100 --batch_size=32 --temperatre=3 --pea_width=4 --ii=4 \
   --beta=0.2 >> log/o14.log 2>&1 & \

elif [ "$op" == "o21" ]; then
  nohup python3 main.py --src_file_path=data/o21.txt --actor_lr=0.0001 --gcn_dims=128 \
  --max_iteration=2500 --warmup_iteration=100 --batch_size=32 --temperatre=1.5 --pea_width=4 --ii=3 \
  --beta=0.2 \

elif [ "$op" == "test40" ]; then
  nohup python3 main.py --src_file_path=test/si2_r005_s40.A28.out --tgt_file_path=test/si2_r005_s40.B28.out \
  --batch_size=64 --max_iteration=2000 --warmup_iteration=64 --actor_lr=0.0001 --critic_lr=0.0002 --temperature=10\

elif  [ "$op" = "test100" ]; then
  nohup python3 main.py --src_file_path=test/si2_r001_s100.A00.out --tgt_file_path=test/si2_r001_s100.B00.out \
  --batch_size=64 --max_iteration=5000 --warmup_iteration=2000 --actor_lr=0.0001 --critic_lr=0.0002 \

elif [ "$op" == "clean" ]; then
  rm nohup.out train_reward.log test_reward.log max_train_reward.log
  rm -rf log/*
else
  echo "请输入 bash run.sh (想要测试的模型 || clean) 示例: bash run.sh demo"
fi
