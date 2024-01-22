#!/bin/bash
# This script evaluates the knowledge predictions for various models and strategies.

# Set the debug mode (use 'True' to enable debugging).
debug=False

# Number of evaluations to perform.
eval_num=500

# Number of information retrieval (IR) evidence to consider.
IR_num=10

# Whether to use ground truth knowledge or not.
wo_ground_truth_knowledge=False

# Error tolerance.
outer_strategy=max

for name in random_prompt_llama_65b_T100
do

# seen split
ref=./emnlp_data/wow/random_testset/seen_random_testset.txt
hyp=${name}/seen_knowledge

exp_name=${name}_IR${IR_num}_seen_knowledge_$outer_strategy
echo $exp_name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/eval_exp.py  \
--hyp_path $hyp \
--ref_path $ref \
--use_IR_eval \
--debug $debug \
--eval_num $eval_num \
--wo_ground_truth_knowledge $wo_ground_truth_knowledge \
--retrieved_num $IR_num  1>log/log-${exp_name} 2>&1 

wait


# unseen split
ref=./emnlp_data/testsets500/unseen_random_testset.txt
hyp=${name}/unseen_knowledge

exp_name=${name}_IR${IR_num}_unseen_knowledge
echo $exp_name

export CUDA_VISIBLE_DEVICES=3
PYTHONPATH=. python -u src/eval_exp.py  \
--hyp_path $hyp \
--ref_path $ref \
--use_IR_eval \
--debug $debug \
--eval_num $eval_num \
--wo_ground_truth_knowledge $wo_ground_truth_knowledge \
--retrieved_num $IR_num  1>log/log-${exp_name} 2>&1 

wait

done 