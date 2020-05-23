#!/bin/bash
#SBATCH --job-name=ExHiRD_s_seed3435_PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk_train
#SBATCH --mail-user=wangzaicuhk@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/king3/wchen/Code4Git/ExHiRD-DKG/sh/ExHiRD/sh_log/ExHiRD_s_seed3435_PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk_train.log
#SBATCH --gres=gpu:1
#SBATCH -p gpu_24h
#SBATCH -w gpu26

home_dir=/research/king3/wchen/Code4Git/ExHiRD-DKG/
cd ${home_dir}

#export CUDA_VISIBLE_DEVICES=3

model_name="ExHiRD_s"
processed_type="PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_RmKeysAllUnk"
seed=3435
echo "=============================ExHiRD_s_seed${seed}================================="

model_save_dir="saved_models/${model_name}_seed${seed}_${processed_type}/"
mkdir -p ${model_save_dir}
log_dir="logs/train/${model_name}/"
mkdir -p ${log_dir}

/research/king3/wchen/Anaconda3/envs/py3.6_th1.0_cuda9.0/bin/python3.6 -u train.py \
-copy_attn \
-reuse_copy_attn \
-word_vec_size=100 \
-share_embeddings \
-rnn_size=300 \
-input_feed=1 \
-rnn_type=GRU \
-global_attention=general \
-data=data/train_valid_dataset/onmt_processed_dataset/with_copy_seqE_HRD_${processed_type}/full_processed_kp20k  \
-batch_size=10 \
-train_steps=300000 \
-save_checkpoint_steps=2500 \
-valid_steps=2500 \
-valid_batch_size=10 \
-gpu_ranks=0 \
-world_size=1 \
-seed=${seed} \
-optim=adam \
-max_grad_norm=1 \
-dropout=0.0 \
-learning_rate=0.001 \
-learning_rate_decay=0.5 \
-report_every=50 \
-max_generator_batches=0 \
-encoder_type=brnn \
-decoder_type=hrd_rnn \
-word_dec_init_type=hidden_vec \
-sent_dec_input_feed_w_type=attn_vec \
-remove_input_feed_h \
-seqE_HRD_rescale_attn \
-sent_dec_init_type=enc_last \
-enc_layers=2 \
-dec_layers=1 \
-exclusive_loss \
-lambda_ex=1 \
-ex_loss_win_size=4 \
-save_model=${model_save_dir}/${model_name}_seed${seed}_${processed_type} \
-log_file=${log_dir}/${model_name}_seed${seed}_${processed_type}_train.log

# select the best model and remove others
# if you do not want to remove the intermediate models, please comment the following lines
/research/king3/wchen/Anaconda3/envs/py3.6_th1.0_cuda9.0/bin/python3.6 -u saved_model_utils.py \
-model_dir=${model_save_dir}/
