eval_batch_size=128
ref_type=train
expname=reranker.scl-temp0.1-coeff1.epoch10.lr5e-5
distance_method=maha

train_dtaset=dd
for eval_dataset in dd grade_eval_dd
do
    python evaluators/run_evaluation.py --reranker_exp_name $expname --distance_function $distance_method --ref_type $ref_type --eval_dataset $eval_dataset --dataset $train_dtaset  --eval_batch_size $eval_batch_size
done
