
for name in "YOUR_EXP_DIR"
do

echo $name
exp_name=${name}_w_hyp_knowledge
tail -2 helpfulness_results/${exp_name}.txt
echo

done

