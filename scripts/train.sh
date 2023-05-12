#!/bin/bash
DATASET=dd

temp=0.1
weight=1
lr=5e-5


python reranker/train_reranker.py --dataset $DATASET --batch_size 8 --accumulation_step 2 --lr $lr --scheduler_type linear --exp_name reranker.scl-temp"$temp"-coeff"$weight".epoch10.lr"$lr" --epochs 10 --use_scl_loss --scl_coeff $weight --scl_temp $temp 
