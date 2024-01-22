for name in your_prediction_dir
do

hyp=/misc/kfdata01/kf_grp/lchen/FactualityPrompt/model_scale_exp/knowledge/${name}/nq_knowledge

ref=/misc/kfdata01/kf_grp/lchen/EMNLP23/experiments/emnlp_data/nq/random_testset/nq_test_random_testset.txt

exp_name=info_${name}
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u info.py  \
--task nq \
--ref_path $ref \
--hyp_path $hyp 1>log/log-${exp_name} 2>&1 

echo 

wait
done
