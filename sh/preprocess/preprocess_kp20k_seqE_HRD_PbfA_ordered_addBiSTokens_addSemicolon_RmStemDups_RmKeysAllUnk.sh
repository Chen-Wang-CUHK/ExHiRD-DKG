# !/bin/bash


# change to the absolute path to ExHiRD-DKG
home_dir="/research/king3/wchen/Code4Git/ExHiRD-DKG/"

dataset="processed_kp20k"
DATADIR="${home_dir}/data/train_valid_dataset"

cd ${home_dir}

data_type="seqE_HRD"

unk_rm_type="RmKeysAllUnk"

processed_type="PbfA_ordered_addBiSTokens_addSemicolon_RmStemDups_${unk_rm_type}"
SAVED_DIR="${DATADIR}/onmt_processed_data/with_copy_${data_type}_${processed_type}"
mkdir -p ${SAVED_DIR}

/research/king3/wchen/Anaconda3/envs/py3.6_th1.0_cuda9.0/bin/python3.6 -u preprocess.py \
-train_src=${DATADIR}/processed_raw_data/${dataset}_training_context_filtered_${unk_rm_type}.txt \
-train_tgt=${DATADIR}/processed_raw_data/${dataset}_training_keyphrases_filtered_${processed_type}.txt \
-valid_src=${DATADIR}/processed_raw_data/${dataset}_validation_context_filtered_${unk_rm_type}.txt \
-valid_tgt=${DATADIR}/processed_raw_data/${dataset}_validation_keyphrases_filtered_${processed_type}.txt \
-save_data=${SAVED_DIR}/full_${dataset} \
-shard_size=20000 \
-src_vocab_size=50000 \
-share_vocab \
-src_seq_length=400 \
-tgt_seq_length=80 \
-seed=3435 \
-report_every=20000 \
-filter_valid \
-log_file=${DATADIR}/logs/${dataset}_with_copy_${data_type}_${processed_type}_preprocess.log \
-dynamic_dict \
-use_existing_vocab \
-hr_tgt
