#!/bin/bash
#SBATCH --job-name=ExHiRD_h_seed3435_history1_4_PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk_translate
#SBATCH --mail-user=wangzaicuhk@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/king3/wchen/Code4Git/ExHiRD-DKG/sh/ExHiRD/sh_log/ExHiRD_h_seed3435_history1_4_PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk_translate.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu_24h
#SBATCH -w gpu9

home_dir=/research/king3/wchen/Code4Git/ExHiRD-DKG/
cd ${home_dir}

#export CUDA_VISIBLE_DEVICES=2

model_name="ExHiRD_h"
min_kp_num=1
max_kp_num=20

seed=3435
processed_type="PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk"

model_save_dir="saved_models/${model_name}_seed${seed}_${processed_type}/"
SAVED_MODEL="${model_name}_seed${seed}_${processed_type}_ppl_4.920_acc_64.467_step_150000.pt"


for data in "inspec" "krapivin" "semeval" "kp20k"
do
  win_size=1
  if [ "${data}" == "inspec" ]
  then
    win_size=4
  fi

  history="history${win_size}"
  LOG_SAVED_DIR="logs/translate/${model_name}/log/${history}/seed${seed}/"
  mkdir -p ${LOG_SAVED_DIR}
  OUT_SAVED_DIR="logs/translate/${model_name}/out/${history}/seed${seed}/"
  mkdir -p ${OUT_SAVED_DIR}
  
  echo "===================${data}_${model_name}_${history}_translate========================="
  /research/king3/wchen/Anaconda3/envs/py3.6_th1.0_cuda9.0/bin/python3.6 -u translate.py \
  -min_kp_length=1 \
  -max_kp_length=6 \
  -min_kp_num=${min_kp_num} \
  -max_kp_num=${max_kp_num} \
  -src=data/test_datasets/processed_${data}_testing_context.txt \
  -tgt=data/test_datasets/processed_${data}_testing_keyphrases.txt \
  -beam_size=1 \
  -n_best=1 \
  -batch_size=6 \
  -gpu=0 \
  -single_word_maxnum=-1 \
  -ap_splitter="<absent_end>" \
  -translate_report_num=500 \
  -first_valid_word_exclusive_search \
  -exclusive_window_size=${win_size} \
  -output=${OUT_SAVED_DIR}/${model_name}_seed${seed}_${data}_${history}.out \
  -log_file=${LOG_SAVED_DIR}/${model_name}_seed${seed}_${data}_${history}_translate.log \
  -model=${model_save_dir}/${SAVED_MODEL}
done

