for name in your_predictions_dir
do

exp_name=${name}_w_hyp_knowledge
# debug=True
debug=False

testfile=emnlp_data/nq/random_testset/nq_test_random_testset.txt
promptfile=./emnlp_data/nq/random_prompts/nq_test_random_prompt.txt
hyp_knowledge="${name}_w_hyp_knowledge"

# downstream_model=flan-t5-xxl
downstream_model=llama-65B
knowledge_type=w_hyp_knowledge
zero_shot=False

export TRANSFORMERS_CACHE='YOUR_DIR'
export HF_HOME='YOUR_DIR'
export HUGGINGFACE_HUB_CACHE='YOUR_DIR'

python3 -u helpfulness.py \
--exp_name $exp_name \
--task nq \
--zero_shot $zero_shot \
--debug $debug \
--testfile $testfile \
--hyp_knowledge $hyp_knowledge \
--promptfile $promptfile \
--downstream_model $downstream_model \
--knowledge_type $knowledge_type 1>log/$exp_name.log 2>&1

wait

done 