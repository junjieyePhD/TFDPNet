export CUDA_VISIBLE_DEVICES=0


model_name=TFDPNet


for pred_len in 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 256 \
      --batch_size 32 \
      --dropout 0.5 \
      --train_epochs 10 \
      --learning_rate 0.002 \
      --n_heads 2
done


for pred_len in 192
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 64 \
      --dropout 0.3 \
      --train_epochs 10 \
      --learning_rate 0.01 \
      --n_heads 4
done



for pred_len in 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 32 \
      --dropout 0.8 \
      --train_epochs 10 \
      --learning_rate 0.001 \
      --n_heads 4
done


