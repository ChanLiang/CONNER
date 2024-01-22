REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data

# for model in flan-t5-11B flan-t5-xl flan-t5-large flan-t5-base flan-t5-small
# for model in flan-t5-11B
for model in flan-t5-xxl
do
echo $model


# seen + few-shot
split=seen
data=few-shot
# hyp_file=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge__clean_refinement
hyp_file=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_fewshot_refinement

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path $hyp_file \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-fewshot-refine-knowledge-res.txt 2>log/${model}-${data}-${split}-fewshot-refine-knowledge-err.txt &

# unseen + few-shot
split=unseen
data=few-shot
hyp_file=/misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_fewshot_refinement

export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path $hyp_file \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-fewshot-refine-knowledge-res.txt 2>log/${model}-${data}-${split}-fewshot-refine-knowledge-err.txt &

# # seen + zero-shot
# split=seen
# data=zero-shot
# export CUDA_VISIBLE_DEVICES=1
# PYTHONPATH=. python src/eval_NE_NLI.py  \
# --hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter__refinement \
# --sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
# --doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-zeroshot-refine-knowledge-res.txt 2>log/${model}-${data}-${split}-zeroshot-refine-knowledge-err.txt &


# # unseen + zero-shot
# split=unseen
# data=zero-shot
# export CUDA_VISIBLE_DEVICES=3
# PYTHONPATH=. python src/eval_NE_NLI.py  \
# --hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_knowledge_last_utter__refinement \
# --sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
# --doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-zeroshot-refine-knowledge-res.txt 2>log/${model}-${data}-${split}-zeroshot-refine-knowledge-err.txt &

wait

done


