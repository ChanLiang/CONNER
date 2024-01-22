REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data

IR_num=3
exp_name=IR${IR_num}_eval_refinement

for model in flan-t5-xxl
do
echo $model

# seen + few-shot
split=seen
data=few-shot
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_fewshot_refinement \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait

# unseen + few-shot
split=unseen
data=few-shot
export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_fewshot_refinement \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${exp_name}-${model}-${data}-${split}-res.txt 2>log/${exp_name}-${model}-${data}-${split}-err.txt &

wait


done


