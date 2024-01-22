REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data

IR_num=3
exp_name=OPT-IR${IR_num}_eval

# for model in opt-13b opt-1.3b
for model in opt-13b opt-iml-1.3b opt-1.3b
# for model in opt-6.7b
do
echo $model
# seen + few-shot
split=seen
data=few-shot
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_extract \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait

# unseen + few-shot
split=unseen
data=few-shot
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_extract \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait


done





# GEN_TO_EVALUATE_NAME=./wow/zero-shot/wizard-test-p1.jsonl

# PYTHONPATH=. python src/evaluate_generated_knowledge.py  \
# --gen_path ${GEN_TO_EVALUATE_NAME} 1>log/res.txt 2>log/err.txt


