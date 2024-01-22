
exp_name="YOUR_EXP_NAME"
task=nq

# debug=True
debug=False
testfile=../emnlp23/emnlp_data/nq/random_testset/nq_test_random_testset.txt
promptfile=../emnlp23/emnlp_data/nq/random_prompts/nq_test_random_prompt.txt

# downstream_model=flan-t5-xxl
downstream_model=llama-65B
zero_shot=False
knowledge_type=wo_knowledge

export TRANSFORMERS_CACHE='YOUR_DIR'
export HF_HOME='YOUR_DIR'
export HUGGINGFACE_HUB_CACHE='YOUR_DIR'

python3 -u helpfulness.py \
--exp_name $exp_name \
--task $task \
--zero_shot $zero_shot \
--debug $debug \
--testfile $testfile \
--promptfile $promptfile \
--downstream_model $downstream_model \
--knowledge_type $knowledge_type 1>log/$exp_name.log 2>&1

