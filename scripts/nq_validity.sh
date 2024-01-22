#!/bin/bash

# Toggle debug mode
debug=False

# Number of evaluations
eval_num=500

# List of experiment directories (update with actual directory names)
your_experiment_dir_list=(dir1 dir2 dir3)

for name in "${your_experiment_dir_list[@]}"; do
    ref="./emnlp_data/nq/random_testset/nq_test_random_testset.txt"
    hyp="./answers/nq_answer_for_${name}/nq_answer"

    echo "Running experiment: ${name}"

    export CUDA_VISIBLE_DEVICES=0

    PYTHONPATH=. python -u src/nq_validity.py \
    --hyp_path "$hyp" \
    --ref_path "$ref" \
    --debug "$debug" \
    --eval_num "$eval_num" 1>"log/log-${name}" 2>&1 

    wait
done