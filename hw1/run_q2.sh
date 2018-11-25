#!/bin/bash
set -eux
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python q2.py expert_data/$e.pkl $e --max_timesteps=1000 --num_rollouts=20
done