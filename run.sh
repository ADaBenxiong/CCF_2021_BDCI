#!/bin/bash
export PRETRAIN_DIR=init_model
export MODEL_TYPE=MacBert_large
export CACHE_DIR=cache
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 python train_pytorch.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $PRETRAIN_DIR/$MODEL_TYPE/pytorch_model.bin \
    --config_name $PRETRAIN_DIR/$MODEL_TYPE/config.json \
    --tokenizer_name $PRETRAIN_DIR/$MODEL_TYPE/vocab.txt \
    --do_train \
    --do_test \
    --do_eval \
    --do_misspelling \
    --data_dir $DATA_DIR \
    --train_set $DATA_DIR/train.txt \
    --dev_set $DATA_DIR/dev.txt \
    --test_set $DATA_DIR/test_A.tsv \
    --logging_steps 2000 \
    --save_steps 20000 \
    --output_dir Checkpoints/$MODEL_TYPE/baseline_last2cls_seed_42_2080ti \
    --result_file ccf_qianyan_qm_result_A.csv \
    --seed 42 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --rdrop_coef 0.0 \
    --overwrite_cache \
    --overwrite_output_dir 
