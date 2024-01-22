exp_name=wow_helpfulness_random_knowledge
task=wow

# debug=True
debug=False
testfile=../emnlp23/emnlp_data/wow/random_testset/seen_random_testset.txt
promptfile=../emnlp23/emnlp_data/wow/random_prompts/seen_random_prompt.txt

downstream_model=llama-65B
zero_shot=False
knowledge_type=random_knowledge

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
# --knowledge_type $knowledge_type 


