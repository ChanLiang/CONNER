#!/bin/bash
# This script evaluates the model predictions using various parameters.

# Set the debug mode. If true, additional debugging information will be printed.
debug=False

# Number of evaluations to perform.
eval_num=500

# Number of information retrieval (IR) evidence to consider.
IR_num=10

# Whether to use ground truth knowledge or not.
wo_ground_truth_knowledge=False

# Error tolerance.
outer_strategy=max 

# Loop through all the predictions of your model.
for name in model_prediction_dir; do
    # Define the reference and hypothesis paths.
    ref="emnlp_data/nq/random_testset/nq_test_random_testset.txt"
    hyp="${name}/nq_knowledge"

    # Construct the experiment name based on the current configuration.
    exp_name="${name}_IR${IR_num}_${outer_strategy}"
    echo "Experiment Name: $exp_name"

    # Set the CUDA device.
    export CUDA_VISIBLE_DEVICES=0

    # Run the evaluation script with the specified parameters.
    PYTHONPATH=. python -u src/eval_exp.py \
    --hyp_path "$hyp" \
    --ref_path "$ref" \
    --use_IR_eval \
    --debug "$debug" \
    --eval_num "$eval_num" \
    --wo_ground_truth_knowledge "$wo_ground_truth_knowledge" \
    --outer_strategy "$outer_strategy" \
    --retrieved_num "$IR_num" \
    1> "log/log-${exp_name}" 2>&1 

    # Wait for the process to finish before continuing with the next prediction.
    wait
done