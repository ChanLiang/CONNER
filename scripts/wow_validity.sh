#!/bin/bash

# Toggle debug mode
debug=False

# Number of evaluations
eval_num=500

# List of experiment directories (update with actual directory names)
your_experiment_dir_list=(dir1 dir2 dir3)

for name in $your_experiment_dir_list
do

ref=/misc/kfdata01/kf_grp/lchen/EMNLP23/experiments/emnlp_data/wow/random_testset/seen_random_testset.txt
hyp=wow_answer/wow_answer_for_${name}/wow_answer

exp_name=${name}-answer
echo $exp_name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/wow_validity.py  \
--use_IR_eval \
--retrieved_num 5 \
--hyp_path $hyp \
--ref_path $ref \
--debug $debug \
--eval_num $eval_num 1>log/log-${exp_name} 2>&1 

wait
done
