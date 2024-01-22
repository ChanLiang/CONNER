REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data


# for model in flan-t5-11B flan-t5-xl flan-t5-large flan-t5-base flan-t5-small
for model in flan-t5-11B
do
echo $model
# seen + few-shot
split=seen
data=few-shot
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_response \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_response_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-response-res.txt 2>log/${model}-${data}-${split}-response-err.txt &
# --doc_ref_path $REF_PATH/output_test${split}_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-response-res.txt 2>log/${model}-${data}-${split}-response-err.txt &

# seen + zero-shot
split=seen
data=zero-shot
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_response \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_response_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-response-res.txt 2>log/${model}-${data}-${split}-response-err.txt &


# unseen + few-shot
split=unseen
data=few-shot
export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_response \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_response_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-response-res.txt 2>log/${model}-${data}-${split}-response-err.txt &


# unseen + zero-shot
split=unseen
data=zero-shot
export CUDA_VISIBLE_DEVICES=3
PYTHONPATH=. python src/eval_NE_NLI.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/opt/output/$model/$data/${split}_response \
--sent_ref_path $REF_PATH/output_test${split}_knowledge_sentence_reference.txt \
--doc_ref_path $REF_PATH/output_test${split}_response_knowledge_doc_reference.txt 1>log/${model}-${data}-${split}-response-res.txt 2>log/${model}-${data}-${split}-response-err.txt &

wait

done





# GEN_TO_EVALUATE_NAME=./wow/zero-shot/wizard-test-p1.jsonl

# PYTHONPATH=. python src/evaluate_generated_knowledge.py  \
# --gen_path ${GEN_TO_EVALUATE_NAME} 1>log/res.txt 2>log/err.txt


