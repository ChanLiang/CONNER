# for name in nq_DPR nq_random_prompt_flan_xxl nq_zeroshot_prompt4_flan_xxl nq_random_prompt_llama_65b_T100 nq_zeroshot_prompt4_llama_65b_T100 
for name in nq_random_prompt_chatgpt_T100 nq_zeroshot_prompt4_chatgpt_T100

do

echo $name
res=/misc/kfdata01/kf_grp/lchen/FactualityPrompt/output/${name}/nq_knowledge_avg_sent_ppl
tail -1 $res
echo
echo

done


# for name in wow_DPR zeroshot_prompt2_flan_xxl zeroshot_prompt4_llama_65b random_prompt_flan_xxl random_prompt_llama_65b_T100
for name in random_prompt_chatgpt zeroshot_prompt4_chatgpt_T100
do

echo $name
# res=/misc/kfdata01/kf_grp/lchen/FactualityPrompt/output_emnlp/${name}/seen_knowledge_avg_sent_ppl
res=/misc/kfdata01/kf_grp/lchen/FactualityPrompt/output/${name}/seen_knowledge_avg_sent_ppl
tail -1 $res
echo
echo

done
