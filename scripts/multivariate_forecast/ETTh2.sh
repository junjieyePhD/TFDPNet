#export CUDA_VISIBLE_DEVICES=1

model_name=TFDPNet

seq_len=96

for pred_len in 96 192
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 64 \
      --batch_size 32 \
      --dropout 0.5 \
      --learning_rate 0.01 \
      --d_model_frequency 32 \
      --n_heads 4 \
      --train_epochs 10
done

for pred_len in 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 64 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --d_model_frequency 64 \
      --train_epochs 10 \
      --n_heads 2
done


