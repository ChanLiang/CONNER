for name in your_prediction_dir
do

hyp=${name}/seen_knowledge
ref=./emnlp_data/wow/random_testset/seen_random_testset.txt

exp_name=info_${name}
echo $name

export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=. python -u info.py  \
--task wow \
--ref_path $ref \
--hyp_path $hyp 1>log/log-${exp_name} 2>&1 

echo 

wait
done
