# env: base

for name in your_hyper_dir
do

hyp=${name}/nq_ref

exp_name=ppl_${name}
echo $name

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. python -u src/ppl.py  \
--hyp_path $hyp 1>log/log-${exp_name} 2>&1 

echo 

wait
done
