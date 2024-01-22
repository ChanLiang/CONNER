REF_PATH=/misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/processed_data


# w IR

IR_num=3
exp_name=IR${IR_num}_eval_backup

export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/DPR_top1_knowledge_seen \
--sent_ref_path $REF_PATH/output_testseen_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_testseen_knowledge_doc_reference.txt 1>log/DPR-${exp_name}-zero-shot-res.txt 2>log/DPR-${exp_name}-zero-shot-err.txt

wait

export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=. python src/eval_401.py  \
--hyp_path /misc/kfdata01/kf_grp/lchen/ParlAI/data/wizard_of_wikipedia/DPR_top1_knowledge_unseen \
--sent_ref_path $REF_PATH/output_testunseen_knowledge_sentence_reference.txt \
--use_IR_eval \
--retrieved_num $IR_num \
--doc_ref_path $REF_PATH/output_testunseen_knowledge_doc_reference.txt 1>log/DPR-${exp_name}-zero-shot-unseen-res.txt 2>log/DPR-${exp_name}-zero-shot-unseen-err.txt



