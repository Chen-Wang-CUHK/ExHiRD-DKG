# /bin/bash

# change to the absolute path to HRED-DKG
home_dir=/apdcephfs/common/rickywchen/code/HRED_KG_SUM

cd ${home_dir}

for K in 0
do
  model="ExHiRD_s"
  # 34 343 3435
  for seed in 34 343 3435
  do
    LOG_DIR="logs/full_data_logs/translate/log/ACL2020_reported_new_testing_results_log/${model}/history${K}/seed${seed}/"
    mkdir -p ${LOG_DIR}
    # "inspec" "krapivin" "semeval" "kp20k"
    for dataset in "inspec" "krapivin" "semeval" "kp20k"
    do
      # output file e.g., ExHiRD_s_seed34_inspec_history0.out
      output="logs/full_data_logs/translate/out/ACL2020_reported_new_testing_results_out/${model}/history${K}/seed${seed}/${model}_seed${seed}_${dataset}_history${K}.out"
      log_file="logs/full_data_logs/translate/log/ACL2020_reported_new_testing_results_log/${model}/history${K}/seed${seed}/${model}_seed${seed}_${dataset}_history${K}.log"
      ../../anaconda3/envs/pytorch1.2_cuda10.0/bin/python -u evaluation_utils.py \
      -src=data/test_datasets/processed_${dataset}_testing_context.txt \
      -tgt=data/test_datasets/processed_${dataset}_testing_keyphrases.txt \
      -output=${output} \
      -log_file=${log_file}
    done  
  done
done
