REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data

IR_num=3
# exp_name=IR${IR_num}_eval_backup
exp_name=IR${IR_num}_eval_filter_know

# for model in flan-t5-11B flan-t5-xl flan-t5-large flan-t5-base flan-t5-small
for model in flan-t5-xxl
do
echo $model
# seen + few-shot
split=seen
data=few-shot
# hyp=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge__clean
hyp=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/filter_know_${split}_knowledge

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python src/eval_401.py  \
--hyp_path $hyp \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait


# # seen + zero-shot
# split=seen
# data=zero-shot
# export CUDA_VISIBLE_DEVICES=1
# PYTHONPATH=. python src/eval_401.py  \
# --hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter_ \
# --sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
# --use_IR_eval \
# --retrieved_num $IR_num \
# --doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

# wait

# unseen + few-shot
split=unseen
data=few-shot
hyp=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/filter_know_${split}_knowledge

export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_401.py  \
--hyp_path $hyp \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait

# # unseen + zero-shot
# split=unseen
# data=zero-shot
# export CUDA_VISIBLE_DEVICES=0
# PYTHONPATH=. python src/eval_401.py  \
# --hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter_ \
# --sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
# --use_IR_eval \
# --retrieved_num $IR_num \
# --doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

# wait

done





# GEN_TO_EVALUATE_NAME=./wow/zero-shot/wizard-test-p1.jsonl

# PYTHONPATH=. python src/evaluate_generated_knowledge.py  \
# --gen_path ${GEN_TO_EVALUATE_NAME} 1>log/res.txt 2>log/err.txt


